import json
import os
import pickle
from itertools import cycle

import context
import config as cfg
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import torch

import yastn.yastn as yastn
from yastn.yastn.sym import sym_Z2, sym_U1
from ipeps.integration_yastn import PepsAD, load_PepsAD
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.mps import measure_mpo
from yastn.yastn.tn.mps import measure_overlap

from edge_spec import load_env_from_dict


def Ts_to_MPO(env, L, start_site=0, dirn="x"):
    """
    Reuturn an MPO made of T tensors.
    To represent the MPO with PBC in OBC form, we need to double the bond dimensions:

       -|---|--- ... ---|-
      / |   |           | \
      --O---O--- ... ---O--
        |   |           |
        |   |           |

    Parameters
    ----------
    env (EnvCTM)
    L (int): length of the MPO
    start_site (Site): the site where the entanglement cut locates.
        When dirn='x', the cut intersects the left bond of the start_site.
        When dirn='y', the cut intersects the upper bond of the start_site.
    dirn (str): direction of the entanglement cut.
    """
    if dirn == "x":
        TLs, TRs = [], []
        site = start_site
        for _ in range(env.psi.dims[0]): # iterate in x-direction
            r_site = env.nn_site(site, "l") # env[r_site].r matches with env[site].l
            TL, TR = env[site].l, env[r_site].r
            TL = TL.transpose(axes=(2, 1, 0))
            TL = TL.unfuse_legs(axes=(1,))
            TR = TR.unfuse_legs(axes=(1,))
            TLs.append(TL)
            TRs.append(TR)
            site = env.nn_site(site, "b")

        #  0   __ 1 (top)    (top) 1__   0
        #  | /                        \  |
        #  TL----2 (bot)     (bot) 2----TR
        #  |                            |
        #  3                            3

        # TODO: sign
        leg = TRs[0].get_legs(axes=0) # leg coming from PBC
        TR_start = TRs[0].swap_gate(axes=(2, (0, 3)))
        TR_start = TR_start.fuse_legs(axes=(1, (0, 3), 2), mode='hard')
        TR_start = TR_start.add_leg(axis=0, s=-1)
        #
        #           0 (dim=1)
        #           | --
        # 1  (top)--TR |
        #        /__|__|_(bot) 3
        #           |  |
        #            | 2

        TR_cycle = [None]*len(TRs)
        for i in range(0, env.psi.dims[0]):
            I = yastn.eye(config=TRs[i].config, legs=(leg.conj(), leg), isdiag=False)
            TR_middle = TRs[i].tensordot(I, axes=((), ()))
            TR_middle = TR_middle.swap_gate(axes=(2, (3, 5)))
            TR_cycle[i] = TR_middle.fuse_legs(axes=((4, 0), 1, (5, 3), 2), mode='hard')
            #       0
            #       |
            #	   |  |
            #   1--TR |
            #    /-|--|-3
            #       |
            #       2

        TR_end = TRs[-1].add_leg(axis=-1, s=1)
        TR_end = TR_end.fuse_legs(axes=((3, 0), 1, 4, 2), mode='hard')

        #            0
        #            |
        #           |  |
        # 1  (top)--TR_|
        #        /__|___(bot) 3
        #           |
        #           2 (dim=1)

        leg = TLs[0].get_legs(axes=0) # leg coming from PBC
        TL_start = TLs[0].swap_gate(axes=(2, (0, 3)))
        TL_start = TL_start.fuse_legs(axes=(2, (0, 3), 1), mode='hard')
        TL_start = TL_start.add_leg(axis=0, s=-1)
        #           0 (dim=1)
        #         __|
        #         | TL---(top) 3
        #  1(bot)_|_|_\
        #	      | |
        #          |
        #          2


        TL_cycle = [None]*len(TLs)
        for i in range(0, env.psi.dims[0]):
            #    4 0
            #	 | |
            #    | TL---1
            #  2-|-|-\
            #    5 3
            I = yastn.eye(config=TLs[i].config, legs=(leg.conj(), leg), isdiag=False)
            TL_middle = TLs[i].tensordot(I, axes=((), ()))
            TL_middle = TL_middle.swap_gate(axes=(2, (3, 5)))
            TL_cycle[i] = TL_middle.fuse_legs(axes=((4, 0), 2, (5, 3), 1), mode='hard')

        TL_end = TLs[-1].add_leg(axis=-1, s=1)
        TL_end = TL_end.fuse_legs(axes=((3, 0), 2, 4, 1), mode='hard')
            #     0
            #     |
            #	 | |
            #    |_TL---3
            # 1____|_\
            #      |
            #      2 (dim=1)

        mpo_R = yastn.tn.mps.MpsMpoOBC(N=env.psi.dims[0]*L, nr_phys=2)
        mpo_L = yastn.tn.mps.MpsMpoOBC(N=env.psi.dims[0]*L, nr_phys=2)

        TL_gen = cycle(TL_cycle)
        TR_gen = cycle(TR_cycle)
        next(TL_gen)
        next(TR_gen)

        mpo_L[0], mpo_R[0] = TL_start, TR_start
        mpo_L[mpo_L.N-1], mpo_R[mpo_R.N-1] = TL_end, TR_end
        for i in range(1, env.psi.dims[0]*L-1):
            mpo_R[i] = next(TR_gen)
            mpo_L[i] = next(TL_gen)

        return mpo_R, mpo_L

def compress_mpo(mpo, Dmax=36):
    opts_svd = {'tol':1e-10, "D_total":Dmax}
    mpo_comp = yastn.tn.mps.random_mpo(mpo, D_total=Dmax, dtype='complex128')
    for output in yastn.tn.mps.compression_(mpo_comp, mpo, method='2site', max_sweeps=30, iterator_step=1, opts_svd=opts_svd, normalize=False):
        print(output)
        if output.doverlap.norm() < 1e-9:
            break
    return mpo_comp



def _get_null_space(A):
    # Find Q such that A @ Q = 0
    u, s, vh =A.svd(axes=(0, 1), sU=A.s[0], policy="fullrank", full_matrices=True)
    s = s > 1e-7
    vh = s @ vh
    leg = vh.get_legs(axes=1)
    I = yastn.eye(config=A.config, legs=(leg.conj(), leg), isdiag=False)
    P = vh.tensordot(vh, axes=(0, 0), conj=(1, 0))
    P_orth = I - P # orthogonal complement
    if P_orth.norm() < 1e-9: # No orthogonal complement, return empty tensor
        return None

    Q, R = P_orth.qr(axes=(0, 1), sQ=P_orth.s[1], Qaxis=1, Raxis=0)
    mask = R.drop_leg_history().diag().__abs__() > 1e-7

    # select entries with value True
    reduced_mask = yastn.Tensor(config=mask.config, s=mask.s, n=mask.n, isdiag=False, dtype=mask.yastn_dtype)
    for charge in mask.get_blocks_charge():
        block = mask[charge]
        reduced_block = block[block]
        if reduced_block.numel() > 0:
            Ds = (len(block), len(reduced_block))
            val = torch.zeros(*Ds, dtype=reduced_block.dtype)
            idx = torch.arange(reduced_block.numel())
            val[idx, idx] = reduced_block
            reduced_mask.set_block(ts=charge, Ds=Ds, val=val)

    Q = Q @ reduced_mask
    return Q


def get_gauge_fixing_V(mps):
    gauge_Vs = yastn.tn.mps.MpsMpoOBC(N=mps.N, nr_phys=1)
    mps.canonize_(to='last', normalize=True)
    for i in range(mps.N):
        Ai = mps[i].fuse_legs(axes=(2, (0, 1)))
        Vi = _get_null_space(Ai.conj())
        if Vi:
            gauge_Vs[i] = Vi.unfuse_legs(axes=(0,))
    return gauge_Vs


def compress_Xs(Xs):
    data_t, meta_t, slices = [], [], []
    start = 0
    for X in Xs:
        if X:
            data, meta = X.compress_to_1d()
            data_t.append(data)
            meta_t.append(meta)
            slices.append((start, start + data.numel()))
            start += data.numel()
        else:
            meta_t.append(None)
            slices.append((start, start))  # empty slice
    return torch.cat(data_t), meta_t, slices

def decompress_Xs(data_1d, meta_t, slices):
    Xs = []
    for meta, sl in zip(meta_t, slices):
        if meta:
            Xs.append(yastn.decompress_from_1d(data_1d[sl[0]:sl[1]], meta=meta))
        else:
            Xs.append(None)
    return Xs

def Heff(mps, mpo, gauge_Vs, meta_X, slices, Xs_1d):
    mps.canonize_(to='last', normalize=True)
    mps_r = mps.copy()
    mps_r.canonize_(to='first', normalize=True)
    N = mpo.N
    to_tensor= lambda x: mps[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=mps[0].yastn_dtype, device=mps[0].device)
    to_numpy= lambda x: mps[0].config.backend.to_numpy(x)
    Xs = decompress_Xs(to_tensor(Xs_1d), meta_X, slices)

    rho_l_0 = yastn.ones(config=mps.config,
                         legs=(mps[0].get_legs(0).conj(), mpo[0].get_legs(0).conj(), mps[0].get_legs(0)))
    sigma_l_0 = yastn.zeros(config=mps.config,
                            legs=(mps[0].get_legs(0).conj(), mpo[0].get_legs(0).conj(), mps[0].get_legs(0)))
    rho_r_end = yastn.ones(config=mps_r.config,
                         legs=(mps_r[N-1].get_legs(2).conj(), mpo[N-1].get_legs(2).conj(), mps_r[N-1].get_legs(2)))
    sigma_r_end = yastn.zeros(config=mps_r.config,
                              legs=(mps_r[N-1].get_legs(2).conj(), mpo[N-1].get_legs(2).conj(), mps_r[N-1].get_legs(2)))
    rho_ls, rho_rs = [rho_l_0], [rho_r_end]
    sigma_ls, sigma_rs = [sigma_l_0], [sigma_r_end]


    def contract_left_edge(edge, top, O, bottom):
        res = edge.tensordot(top, axes=(0, 0))
        res = res.tensordot(O, axes=((0, 2), (0, 3)))
        res = res.tensordot(bottom, axes=((0, 2), (0, 1)), conj=(0, 1))

        return res

    def contract_right_edge(edge, top, O, bottom):
        res = edge.tensordot(top, axes=(0, 2))
        res = res.tensordot(O, axes=((0, 3), (2, 3)))
        res = res.tensordot(bottom, axes=((0, 3), (2, 1)), conj=(0, 1))

        return res

    def Ti_contraction(l, O, r, top):
        res = l.tensordot(top, axes=(0, 0))
        res = res.tensordot(O, axes=((0, 2), (0, 3)))
        res = res.tensordot(r, axes=((1, 3), (0, 1)))

        return res

    for i in range(N):
        rho_ls.append(contract_left_edge(rho_ls[-1], mps[i], mpo[i], mps[i]))
        rho_rs.insert(0, contract_right_edge(rho_rs[0], mps_r[N-1-i], mpo[N-1-i], mps_r[N-1-i]))

        sigma_l = contract_left_edge(sigma_ls[-1], mps_r[i], mpo[i], mps[i])
        if gauge_Vs[i]:
            B = gauge_Vs[i]@Xs[i]
            sigma_l += contract_left_edge(rho_ls[-2], B, mpo[i], mps[i])
        sigma_ls.append(sigma_l)

        sigma_r = contract_right_edge(sigma_rs[0], mps[N-1-i], mpo[N-1-i], mps_r[N-1-i])
        if gauge_Vs[N-1-i]:
            B = gauge_Vs[N-1-i]@Xs[N-1-i]
            sigma_r += contract_right_edge(rho_rs[1], B, mpo[N-1-i], mps_r[N-1-i])
        sigma_rs.insert(0, sigma_r)

    rho_ls = rho_ls[:-1]
    sigma_ls = sigma_ls[:-1]
    rho_rs = rho_rs[1:]
    sigma_rs = sigma_rs[1:]

    Xps = []
    for i in range(N):
        if gauge_Vs[i]:
            B = gauge_Vs[i]@Xs[i]
            Ti = Ti_contraction(sigma_ls[i], mpo[i], rho_rs[i], mps_r[i]) \
                + Ti_contraction(rho_ls[i], mpo[i], sigma_rs[i], mps[i]) \
                + Ti_contraction(rho_ls[i], mpo[i], rho_rs[i], B)

            Xps.append(gauge_Vs[i].tensordot(Ti, axes=((0, 1), (0, 1)), conj=(1, 0)))
        else:
            Xps.append(None)
    Xp_data1d, meta_Xp, Xp_slices = compress_Xs(Xps)

    return to_numpy(Xp_data1d)

def qp_excitation(mpo, gs_mps, k=3):
    gauge_Vs = get_gauge_fixing_V(gs_mps)
    X_init = []
    for i in range(gauge_Vs.N):
        V = gauge_Vs[i]
        if V:
            X_init.append(yastn.rand(config=V.config, legs=(V.get_legs(2).conj(), gs_mps[i].get_legs(2)), n=0, dtype=gs_mps[i].yastn_dtype))
        else:
            X_init.append(None)

    Xs_1d, meta_X, slices = compress_Xs(X_init)
    _mv = lambda x: Heff(gs_mps, mpo, gauge_Vs, meta_X, slices, x)
    op = LinearOperator(shape=(len(Xs_1d), len(Xs_1d)), matvec=_mv, dtype=np.complex128)
    ws, vs = eigs(op, k=k, which='LM', v0=Xs_1d, return_eigenvectors=True)

    return ws, vs, meta_X, slices, gauge_Vs

def reconstruct_mps(v, meta_X, slices, gauge_Vs, gs_mps):
    to_tensor= lambda x: gs_mps[0].config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=gs_mps[0].yastn_dtype, device=gs_mps[0].device)
    Xs = decompress_Xs(to_tensor(v), meta_X, slices)
    res = None
    gs_mps.canonize_(to='first', normalize=True)
    tmps = []
    for j in range(len(Xs)):
        gs_mps.absorb_central_(to='last')
        gs_mps.orthogonalize_site_(j, to='last', normalize=True)
        tmp = gs_mps.copy()
        if Xs[j]:
            tmp[j] = gauge_Vs[j]
            tmp.A[tmp.pC] = Xs[j]
            tmp.absorb_central_(to='last')
            gs_mps.absorb_central_(to='last')
            tmps.append(tmp)
            # print(measure_overlap(tmp, tmp), tmp.factor)
            if res is None:
                res = tmp
            else:
                res = res + tmp
    return res

def overlap(mpo1, mpo2):
    norm = measure_overlap(mpo1, mpo1).real
    comp_norm = measure_overlap(mpo2, mpo2).real
    overlap = measure_overlap(mpo1, mpo2)
    return overlap/torch.sqrt(norm*comp_norm)

if __name__ == "__main__":
    state_file = (
        f"./t1t2t3_V1_grad/seed_123/V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu_cont4_state.json"
    )

    yastn_config = yastn.make_config(
        backend=backend,
        # sym=sym_Z2,
        sym=sym_U1,
        fermionic=True,
        default_device="cpu",
        default_dtype="complex128",
    )

    D = 6
    opt_chi = 72
    omp_cores = 16
    torch.set_num_threads(omp_cores)

    # Converge environment
    opts_svd = {
        "D_total": opt_chi,
        "tol": 1e-8,
        "eps_multiplet": 1e-8,
        "fix_signs": True,
        "truncate_multiplets": True,
    }

    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(opt_chi,))
    env_dict_filename = "./t1t2t3_V1_grad/seed_123/es/env_dict_V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu"


    start_site=0
    state = load_PepsAD(yastn_config, state_file)
    with open(env_dict_filename, "rb") as handle:
        d = pickle.load(handle)
    env = EnvCTM(state, init="eye", leg=env_leg)
    env = load_env_from_dict(env, d,  yastn_config)
    print("Loaded")

    L = 6
    start = env.sites()[0]
    mpo_R, mpo_L = Ts_to_MPO(env, L, start_site=start, dirn="x")

    Dmax = 72
    mpo_R_comp = compress_mpo(mpo_R, Dmax=Dmax)
    mpo_L_comp = compress_mpo(mpo_L, Dmax=Dmax)

    mpo_R_comp_dict = mpo_R_comp.save_to_dict()
    mpo_L_comp_dict = mpo_L_comp.save_to_dict()

    with open(f"es/mpo_R_comp_L_{L:d}_Dmax_{Dmax:d}", "wb") as f:
        pickle.dump(mpo_R_comp_dict, f)

    with open(f"es/mpo_L_comp_L_{L:d}_Dmax_{Dmax:d}", "wb") as f:
        pickle.dump(mpo_L_comp_dict, f)

    print(overlap(mpo_R_comp, mpo_R))
    print(overlap(mpo_L_comp, mpo_L))

    mpo_comp = mpo_L_comp@mpo_R_comp

    opts_svd = {'tol':1e-8, "D_total": Dmax}
    opts_eigs = {'hermitian': False, 'ncv': 20, "which": 'LM'}

    gs_mps = yastn.tn.mps.random_mps(mpo_comp, D_total=Dmax, n=0, dtype='complex128')
    gen = yastn.tn.mps.dmrg_(gs_mps, mpo_comp, project=[], method='2site', max_sweeps=20, iterator_step=1, opts_eigs=opts_eigs, opts_svd=opts_svd)

    gs_mps_dict = gs_mps.save_to_dict()
    with open(f"es/gs_mps_Dmax_{Dmax:d}", "wb") as f:
        pickle.dump(gs_mps_dict, f)

    for out in gen:
        if out.denergy < 1e-8:
            break

    leading_w = measure_mpo(gs_mps, mpo_comp, gs_mps)/mpo_comp.factor
    ws, vs, meta_X, slices, gauge_Vs = qp_excitation(mpo_comp, gs_mps, k=30)
    print(ws/leading_w)

    with open(f"es/evals_{L:d}_Dmax_{Dmax:d}", "wb") as f:
        pickle.dump(ws, f)

    with open(f"es/evecs_{L:d}_Dmax_{Dmax:d}", "wb") as f:
        pickle.dump(vs, f)

