import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_kagome import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import corrf
from ctm.generic import transferops
from models import kagome
import unittest
import numpy as np
import pickle
from math import sqrt
import sys
print(sys.path)
# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--theta", type=float, default=0.5, help="parametrization between 2- and 3-site terms. theta * pi")
parser.add_argument("--phi", type=float, default=0., help="parametrization between normal and chiral terms. phi * pi")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
# additional observables-related arguments
parser.add_argument("--corrf_n", type=int, default=10, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=10, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--input_prefix", type=str, default="theta_0_phi_0_bonddim_3_chi_16", help="parameters of input state")
parser.add_argument("--output_path", type=str, default="/scratch/yx51/kagome", help="path of output")
parser.add_argument("--restrictions", type=bool, default=False, help="restrictions on the 5 site tensors")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    param_j = np.round(np.cos(np.pi*args.theta), decimals=12)
    param_k = np.round(np.sin(np.pi*args.theta) * np.cos(np.pi*args.phi), decimals=12)
    param_h = np.round(np.sin(np.pi*args.theta) * np.sin(np.pi*args.phi), decimals=12)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    model = kagome.KAGOME(phys_dim=3, j=param_j, k=param_k, h=param_h)
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "1SITE":
        def lattice_to_site(coord):
            return (0, 0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            # vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, 0)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    # elif args.tiling == "8SITE":
    #     def lattice_to_site(coord):
    #         shift_x = coord[0] + 2 * (coord[1] // 2)
    #         vx = shift_x % 4
    #         vy = coord[1] % 2
    #         return (vx, vy)
    else:
        raise ValueError("Invalid tiling: " + str(args.tiling) + " Supported options: " \
                         + "1SITE, BIPARTITE, 2SITE, 4SITE, 8SITE")

    # initialize an ipeps
    if args.instate != None:
        state = read_ipeps_kagome(args.instate, restrictions=args.restrictions, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim_kagome(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type == 'RANDOM':
        bond_dim = args.bond_dim
        A = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        RD = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        RU = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        B = B/torch.max(torch.abs(B))
        C = C/torch.max(torch.abs(C))
        RD = RD/torch.max(torch.abs(RD))
        RU = RU/torch.max(torch.abs(RU))
        kagome_sites = {(0, 0, 0): A, (0, 0, 1): B, (0, 0, 2): C, (0, 0, 3): RD, (0, 0, 4): RU}
        if args.tiling in ["BIPARTITE", "2SITE", "4SITE"]:
            A2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            B2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            C2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            RD2 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RU2 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            kagome_sites[(1, 0, 0)] = A2/torch.max(torch.abs(A2))
            kagome_sites[(1, 0, 1)] = B2/torch.max(torch.abs(B2))
            kagome_sites[(1, 0, 2)] = C2/torch.max(torch.abs(C2))
            kagome_sites[(1, 0, 3)] = RD2/torch.max(torch.abs(RD2))
            kagome_sites[(1, 0, 4)] = RU2/torch.max(torch.abs(RU2))
        if args.tiling in ["4SITE"]:
            A3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            B3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            C3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RD3 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            RU3 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            kagome_sites[(0, 1, 0)] = A3 / torch.max(torch.abs(A3))
            kagome_sites[(0, 1, 1)] = B3 / torch.max(torch.abs(B3))
            kagome_sites[(0, 1, 2)] = C3 / torch.max(torch.abs(C3))
            kagome_sites[(0, 1, 3)] = RD3 / torch.max(torch.abs(RD3))
            kagome_sites[(0, 1, 4)] = RU3 / torch.max(torch.abs(RU3))

            A4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            B4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            C4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RD4 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            RU4 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            kagome_sites[(1, 1, 0)] = A4 / torch.max(torch.abs(A4))
            kagome_sites[(1, 1, 1)] = B4 / torch.max(torch.abs(B4))
            kagome_sites[(1, 1, 2)] = C4 / torch.max(torch.abs(C4))
            kagome_sites[(1, 1, 3)] = RD4 / torch.max(torch.abs(RD4))
            kagome_sites[(1, 1, 4)] = RU4 / torch.max(torch.abs(RU4))

        state = IPEPS_KAGOME(kagome_sites, restrictions=args.restrictions, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= " \
                         + str(args.ipeps_init_type) + " is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= kagome.KAGOME(phys_dim=3, j=param_j, k=param_k, h=param_h)

    print(state)

    # 2) select the "energy" function
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f = model.energy_1site
        eval_obs_f= model.eval_obs
    elif args.tiling == "1SITE":
        energy_f= model.energy_1site
        # TODO include eval_obs with rotation on B-sublattice
        eval_obs_f= model.eval_obs
    elif args.tiling == "4SITE":
        energy_f = model.energy_1site
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history = []
            e_curr = energy_f(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()] + obs_values)
            print(", ".join([f"{len(history)}", f"{e_curr}"] + [f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0] - history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)

    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{e_curr0}"] + [f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    e_final = energy_f(state, ctm_env_init)
    obs_values, obs_labels = model.eval_obs(state, ctm_env_init)
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{e_final}"] + [f"{v}" for v in obs_values]))

    coord = (0, 0)
    corrf_n = args.corrf_n
    pd = model.phys_dim
    # get operators for correlation function
    tz_op = model.obs_ops["tz"]
    tp_op = model.obs_ops["tp"]
    tm_op = model.obs_ops["tm"]
    vp_op = model.obs_ops["vp"]
    vm_op = model.obs_ops["vm"]
    up_op = model.obs_ops["up"]
    um_op = model.obs_ops["um"]
    y_op = model.obs_ops["y"]
    id_op = model.obs_ops["id"]
    # id_op = torch.eye(pd, dtype=cfg.global_args.torch_dtype)

    transfer_spec_x = np.array(torch.zeros(args.top_n, 2), dtype='float64')
    transfer_spec_y = np.array(torch.zeros(args.top_n, 2), dtype='float64')
    tztz_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    tptm_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    tmtp_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    upum_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    umup_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    vpvm_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    vmvp_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')
    yy_corrf = np.array(torch.zeros(corrf_n, 2), dtype='complex128')

    # correlation length extracted from transfer matrix
    direction = (1, 0)
    spec_x = transferops.get_Top_spec(args.top_n, coord, direction, state, ctm_env_init)
    direction = (0, 1)
    spec_y = transferops.get_Top_spec(args.top_n, coord, direction, state, ctm_env_init)
    for n in range(0, args.top_n):
        transfer_spec_x[n, 0] = spec_x[n, 0].numpy()
        transfer_spec_x[n, 1] = spec_x[n, 1].numpy()
        transfer_spec_y[n, 0] = spec_y[n, 0].numpy()
        transfer_spec_y[n, 1] = spec_y[n, 1].numpy()

    def _gen_op2(op):
        def dummy_op2(r):
            return op
        return dummy_op2

    # correlation functions
    corrf_n_tri = corrf_n // 2 - 1
    directions_rules_map = {(1, 0): 'il,jm,kn->kijnlm', (0, 1): 'il,jm,kn->jkimnl'}
    for direction, rule in directions_rules_map.items():
        if direction == (0, 1):
            n = 0
        elif direction == (1, 0):
            n = 1
        else:
            n = 2
        # tztz
        op1 = torch.einsum(rule, tz_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, tz_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, tz_op, id_op).reshape(pd**3, pd**3)  # odd sites
        tztz_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        tztz_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # tptm
        op1 = torch.einsum(rule, tp_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, tm_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, tm_op, id_op).reshape(pd**3, pd**3)  # odd sites
        tptm_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        tptm_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # tmtp
        op1 = torch.einsum(rule, tm_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, tp_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, tp_op, id_op).reshape(pd**3, pd**3)  # odd sites
        tmtp_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        tmtp_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # upum
        op1 = torch.einsum(rule, up_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, um_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, um_op, id_op).reshape(pd**3, pd**3)  # odd sites
        upum_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        upum_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # umup
        op1 = torch.einsum(rule, um_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, up_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, up_op, id_op).reshape(pd**3, pd**3)  # odd sites
        umup_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        umup_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)\
        # vpvm
        op1 = torch.einsum(rule, vp_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, vm_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, vm_op, id_op).reshape(pd**3, pd**3)  # odd sitesF
        vpvm_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        vpvm_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # vmvp
        op1 = torch.einsum(rule, vm_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, vp_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, vp_op, id_op).reshape(pd**3, pd**3)  # odd sites
        vmvp_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        vmvp_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)
        # yy
        op1 = torch.einsum(rule, y_op, id_op, id_op).reshape(pd**3, pd**3)  # site 0
        op2 = torch.einsum(rule, y_op, id_op, id_op).reshape(pd**3, pd**3)  # even sites
        op3 = torch.einsum(rule, id_op, y_op, id_op).reshape(pd**3, pd**3)  # odd sites
        yy_corrf[1::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op2), corrf_n_tri)
        yy_corrf[::2, n] = corrf.corrf_1sO1sO(coord, direction, state, ctm_env_init, op1, _gen_op2(op3), corrf_n_tri)

    ss_corrf = 0.5 * (tptm_corrf + tmtp_corrf + upum_corrf + umup_corrf + tptm_corrf + tmtp_corrf) + tztz_corrf + 0.75 * yy_corrf
    correlation_dict = {"tztz": tztz_corrf, "tptm": tptm_corrf, "tmtp": tmtp_corrf, "vpvm": vpvm_corrf,
                        "vmvp": vmvp_corrf, "upum": upum_corrf, "umup": umup_corrf, "yy": yy_corrf,
                        "ss": ss_corrf, "transfer_spec_x": transfer_spec_x, "transfer_spec_y": transfer_spec_y}
    print(correlation_dict)
    # filename = "correlation_theta_.pkl"

    # for nt in range(1, args.max_nt+1, 1):
    #     t_spec = boundary_spectrum(ctm_env_init, n_site=nt, top_n=args.top_n)
    #     tmp_filename = "{}/edge_spectrum_nt_{}_{}.pkl".format(args.output_path, nt, args.input_prefix)
    #     with open(tmp_filename, "wb") as fp:
    #         pickle.dump(t_spec, fp)


    # corrSS = model.eval_corrf_SS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(1,0)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSS = model.eval_corrf_SS((0, 0), (0, 1), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(0,1)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSSSS = model.eval_corrf_SSSS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSSSS[(0,0),(1,0)] r " + " ".join([label for label in corrSSSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSSSS[label][i]}" for label in corrSSSS.keys()]))
    #
    # # environment diagnostics
    # print("\n")
    # for c_loc, c_ten in ctm_env_init.C.items():
    #     u, s, v = torch.svd(c_ten, compute_uv=False)
    #     print(f"spectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")
    #
    # transfer operator spectrum
    # site_dir_list = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 1), (1, 0)), ((1, 1), (0, 1))]
    # for sdp in site_dir_list:
    #     print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
    #     l = transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
    #     for i in range(l.size()[0]):
    #         print(f"{i} {l[i, 0]} {l[i, 1]}")
    #
    # # environment diagnostics
    # for c_loc,c_ten in ctm_env_init.C.items():
    #     u,s,v= torch.svd(c_ten, compute_uv=False)
    #     print(f"\n\nspectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


