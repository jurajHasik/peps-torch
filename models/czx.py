import numpy as np
import torch
import yastn.yastn as yastn
import yastn.yastn.tn.mps as mps
import yastn.yastn.backend.backend_torch as backend_torch
from functools import reduce
import config as cfg
from ctm.generic.rdm import rdm1x1, eval_mpo_rdm2x2_oe, eval_mpo_rdm3x1_oe, eval_mpo_rdm1x3_oe

def _get_op_from_mpo(mpo,):
    #
    #
    #    -1(1)        -2(1)      -(i+1)
    # 0--A1--(2)1  1--A2--2 ... i--A3--((-1)**(i+1)//N) * (i+1 + (i+1)//N)
    #    -(N+2)(3)   -(N+3)      -(i+N+2)
    #
    N= len(mpo)
    tn_to_contract= [ [i, -(i+1) ,(-1)**((i+1)//N) * (i+1+(i+1)//N), -(i+N+2)] for i in range(N) ]
    # print(tn_to_contract)
    op=yastn.ncon( list(mpo[i] for i in range(N)), tn_to_contract)
    op= op.fuse_legs(axes=(0,tuple(i+1 for i in range(N)),N+1,tuple(i+N+2 for i in range(N))))
    op= op.remove_leg(axis=-1).remove_leg(0)
    return op

def get_U_czx_fused(yastn_cfg):
    # Lets get an algebra of local/on-site spin-1/2 ops. Here, without explicit symmetry (no block sparsity)
    S12= yastn.operators.Spin12(**yastn_cfg._asdict())
    P1= -0.5*(S12.z()-S12.I())# |1><1|<=>|down><down|

    # We build U_CZX symmetry generator, acting on four spin-1/2 DoFs on-site
    # which is defined as (see https://journals.aps.org/prb/pdf/10.1103/PhysRevB.84.235141 Eq. 1,2,3,and 4 )
    #
    # U_CZX = U_X U_CZ
    #
    # with U_X = X_0 X_1 X_2 X_3 and U_CZ = CZ_01 CZ_12 CZ_23 CZ_30
    #

    # Identity MPO and \prod_i X_i on 4 spins-1/2
    mpo_I= mps.product_mpo(S12.I(), 4)
    mpo_U_X= mps.product_mpo(S12.x(), 4)

    # Create CZ operator between site 1 and site 2 on a chain of N sites
    def CZ_N(N,site1,site2):
        mpo_I_N= mps.product_mpo(S12.I(), N)
        return mps.generate_mpo(mpo_I_N,
                [mps.Hterm(positions=tuple(i for i in range(N)), operators=(S12.I(),)*N), # pauli string
                mps.Hterm(amplitude=-2,positions=(site1,site2),operators=[P1,P1])],      # -2|11><11|
                opts_svd={'tol':1e-14})

    mps_U_CZX= reduce(mps.multiply, [CZ_N(4,i,(i+1)%4) for i in range(4)]+[mpo_U_X,])

    # Lets fuse individual spins of mps_U_CZX together to match on-site tensors of iPEPS and create simple
    # 16x16 matrix representation

    # Fusion (or coarse-graining) of MPO sites has to respect YASTN's convention for resulting MPO tensors
    # see https://yastn.github.io/yastn/mps/properties.html#index-convention
    #
    #    -1(1)          -2(1)  -3(1)  -4(1)
    # 0--A1--(2)1 1(2)--A2--2--A3--3--A4--(2)-5
    #    -6(3)          -7(3)  -8(1)  -9(1)
    #
    return _get_op_from_mpo(mps_U_CZX)

def get_H_czx_mpo_fused(yastn_cfg):

    # lets the define Hamiltonian, acting on a plaquette, which contains four spin-1/2 on each vertex/site
    #
    #  .x  x.
    #  xo--ox
    #   |  |
    #  xo--ox
    #  .x  x.
    #
    # We have to pick consistent indexing/ordering of all DoFs. Let's make a following choice
    #
    #   0,1    4,5
    #   3,2----7,6
    #     |    |
    #   8,9----12,13
    # 11,10    15,14
    #
    # For purpose of evaluating such 16-spin operator on iPEPS, its easier to work
    # with its MPO representation.
    #
    # Let us build this operator as an MPO. Here we use yastn (https://github.com/yastn/yastn),
    # although anything works as far as we get MPO matrices at the end.
    #
    # Lets get an algebra of local/on-site spin-1/2 ops. Here, without explicit symmetry (no block sparsity)
    S12= yastn.operators.Spin12(**yastn_cfg._asdict())
    P0= 0.5*(S12.z()+S12.I()) # |0><0|<=>|up><up|
    P1= -0.5*(S12.z()-S12.I())# |1><1|<=>|down><down|
    Sp= S12.sp() # |0><1|

    # Now we build the MPO
    # from https://yastn.github.io/yastn/examples/mps/build.html#spinless-fermions-with-hopping-at-arbitrary-range-hterm
    #
    # We are going to build the MPO over full space of 16 spins. This makes it easy to stick to the ordering
    # we have chosen earlier and it will not have an effect on the final computational scaling.
    #
    # i) define identity MPO over 4*4 spin-1/2 chain
    mpo_I= mps.product_mpo(S12.I(), 4*4)

    def get_h_czx(idxs=[i for i in range(16)]):
        #
        # ii) projectors on GHZ/Bell pairs of spins on half-plaquettes (P_2 of Fig.3(c)) of PHYSICAL REVIEW B 84, 235141 (2011)
        mpos_P2= []
        for sites in [(idxs[1],idxs[4]), (idxs[6],idxs[13]), (idxs[15],idxs[10]), (idxs[3],idxs[8])]:
            mpos_P2.append(
                mps.generate_mpo(mpo_I,
                    [mps.Hterm(positions=sites,operators=[P0,P0]), mps.Hterm(positions=sites,operators=[P1,P1])],
                    opts_svd={'tol':1e-14})
            )

        #
        # iii) tunneling between all up / all down plaquette states
        mpos_X4= mps.generate_mpo(mpo_I,[
                mps.Hterm(positions=(idxs[2],idxs[7],idxs[12],idxs[9]),operators=[Sp,]*4),
                mps.Hterm(positions=(idxs[2],idxs[7],idxs[12],idxs[9]),operators=[Sp.conj().transpose(),]*4)
            ], opts_svd={'tol':1e-14})

        #
        # iv) Plaquette hamiltonian term (Eq. 5) is product of above MPOs
        h_plaquette= (-1)*reduce(mps.multiply, mpos_P2+[mpos_X4,]) # performs product of MPOs as ((((mpos_P2[0] @ mpos_P2[1]), @ mpos_P2[2]), @ mpos_P2[3]), @ mpos_X4 )
                                                        # with mpos_X4 applied as last
        return h_plaquette

    h_plaquette= get_h_czx()
    # TODO: potentially bug with 'normalize' parameter. The compressed MPO is rescaled

    #
    # we can try to compress it further
    #h_p_compressed= h_plaquette.copy()
    #compression_data= mps.compression_(h_p_compressed, h_plaquette, method='2site', opts_svd={'tol': 1e-14, 'D_total': 8}, normalize=False)
    # In this case, the compression data show there was no gain - for given choice of tolerance on the overlap)
    #print(compression_data)
    #print(f"compressed MPO bond dims: {h_p_compressed.get_bond_dimensions()}")

    # v) Lets fuse individual spins on each site together to match on-site tensors of iPEPS and create new N=4 site MPO
    #    This will be the MPO will actually use to variationally optimize iPEPS for CZX Hamiltonian
    #
    h_p_czx_fused= mps.Mpo(4)

    # Fusion (or coarse-graining) of MPO sites has to respect YASTN's convention for resulting MPO tensors
    # see https://yastn.github.io/yastn/mps/properties.html#index-convention
    #
    #    -1(1)          -2(1)  -3(1)  -4(1)
    # 0--A1--(2)1 1(2)--A2--2--A3--3--A4--(2)-5
    #    -6(3)          -7(3)  -8(1)  -9(1)
    #
    for site in range(4):
        h_p_czx_fused[site]=yastn.ncon( list(h_plaquette[i] for i in range(4*site,4*(site+1))), \
        [[0,-1,1,-6],[1,-2,2,-7],[2,-3,3,-8],[3,-4,-5,-9]] ).fuse_legs(axes=(0,(1,2,3,4),5,(6,7,8,9)))
    return h_p_czx_fused

def get_H_zxz():
    # These operators act on
    def onsite_op(amplitude, op1, op2, op3, op4):
        #return amplitude * torch.einsum(op1, [0, 1], op2, [2, 3], op3, [4, 5], op4, [6, 7], [0, 2, 4, 6, 1, 3, 5, 7])
        return amplitude * torch.kron(op1, torch.kron(op2, torch.kron(op3, op4)))

    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float64)
    id = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)

    #    ( I X X I + X I I X ) + ( I I I I - Z Z Z Z)  <=>  I X + X I
    #                                                       I X   X I
    Xa = (onsite_op(1/2, id, x, x, id) + onsite_op(1/2, x, id, id, x)) \
        @ (onsite_op(1, id, id, id, id) - onsite_op(1, z, z, z, z))/2.

    #    ( X X I I + I I X X ) + ( I I I I - Z Z Z Z)  <=>  X X + I I
    #                                                       I I   X X
    Xb = (onsite_op(1/2, x, x, id, id) + onsite_op(1/2, id, id, x, x)) \
        @ (onsite_op(1, id, id, id, id) - onsite_op(1, z, z, z, z))/2.

    #  I I - Z Z
    #  Z Z   I I
    Za = onsite_op(1/2, id, id, z, z) - onsite_op(1/2, z, z, id, id)

    #  I Z - Z I
    #  I Z   Z I
    Zb = onsite_op(1/2, id, z, z, id) - onsite_op(1/2, z, id, id, z)
    return Za, Xa, Zb, Xb

def get_mirrors():
    #Define mirror symmetries
    
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)

    #Swapping of spin-1/2 onsite
    swap_01 = np.zeros((16, 16))
    swap_02 = np.zeros((16, 16))
    swap_03 = np.zeros((16, 16))
    swap_12 = np.zeros((16, 16))
    swap_13 = np.zeros((16, 16))
    swap_23 = np.zeros((16, 16))

    id01 = [1, 0, 2, 3]
    id02 = [2, 1, 0, 3]
    id03 = [3, 1, 2, 0]
    id12 = [0, 2, 1, 3]
    id13 = [0, 3, 2, 1]
    id23 = [0, 1, 3, 2]

    #construct swap operators
    for swap_op, swap_idx in zip([swap_01, swap_02, swap_03, swap_12, swap_13, swap_23],
                                    [id01, id02, id03, id12, id13, id23]):
        for j in range(2**4):
            j_swapped = list(format(j, f'04b'))
            swap_j = int("".join([j_swapped[i] for i in swap_idx]), 2)
            swap_op[j, swap_j] = 1

    Mx = torch.tensor(swap_01.dot(swap_23), dtype=torch.float64) @ torch.kron(x, torch.kron(x, torch.kron(x, x)))
    My = -torch.tensor(swap_03.dot(swap_12), dtype=torch.float64) @ torch.kron(y, torch.kron(y, torch.kron(y, y))).real
    return Mx, My

class CZX():
    def __init__(self, g_czx=1, g_zxz=0, V=0, yastn_cfg=None, global_args=cfg.global_args):
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2**4
        
        self.g_czx=g_czx
        self.g_zxz=g_zxz
        self.V= V

        yastn_cfg= yastn_cfg if not (yastn_cfg is None) else \
            yastn.make_config(backend=backend_torch)
        self.h_p_czx_fused= get_H_czx_mpo_fused(yastn_cfg)
        
        x = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype)
        z = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype)
        self.U_Z= torch.einsum('ai,bj,ck,dl->abcdijkl',z,z,z,z).reshape([2**4,]*2)
        self.U_X= torch.einsum('ai,bj,ck,dl->abcdijkl',x,x,x,x).reshape([2**4,]*2)
        self.Za, self.Xa, self.Zb, self.Xb= get_H_zxz()

    def eval_H_ops(self, state, env, verbosity=0):
        # Given an iPEPS and its CTMRG environment, evaluate the energy over all non-equivalent plaquettes
        # and 1x3 and 3x1 strips
        # Eq. 45
        
        id_site= torch.eye(2**4,dtype=torch.float64)
        id_mpo_site= id_site[None,:,None,:]
        mpo_czx= tuple(self.h_p_czx_fused[i][(),] for i in range(len(self.h_p_czx_fused)))
        mpo_ZaXaZa= ( self.Za[None,:,None,:], self.Xa[None,:,None,:], self.Za[None,:,None,:] )
        mpo_ZbXbZb= ( self.Zb[None,:,None,:], self.Xb[None,:,None,:], self.Zb[None,:,None,:] )

        eczx, ezxza, ezxzb, e_uz, e_ux= 0, 0, 0, 0, 0
        for coord in state.sites:
            r1x1_id= rdm1x1(coord, state, env, operator=id_site, sym_pos_def=False,)
            e_uz_i= rdm1x1(coord, state, env, operator=self.U_Z, sym_pos_def=False,)
            e_uz+=e_uz_i/r1x1_id
            e_ux_i= rdm1x1(coord, state, env, operator=self.U_X, sym_pos_def=False,)
            e_ux+=e_ux_i/r1x1_id
            if verbosity>1: print(f"{coord} norm {r1x1_id} <U_Z>/norm {e_uz_i/r1x1_id} <U_X>/norm {e_ux_i/r1x1_id}")

            e_h_p= eval_mpo_rdm2x2_oe(coord, state, env, mpo_czx)
            e_id_p= eval_mpo_rdm2x2_oe(coord, state, env, (id_mpo_site,)*4 )
            if verbosity>1: print(f"plaquette {coord} <h_CZX> {e_h_p} norm {e_id_p} <h_CZX>/norm {e_h_p/e_id_p}")
            eczx+= e_h_p/e_id_p

            ezxza_i= eval_mpo_rdm1x3_oe(coord, state, env, mpo_ZaXaZa)
            e_id_p= eval_mpo_rdm1x3_oe(coord, state, env, (id_mpo_site,)*3 )
            if verbosity>1: print(f"rdm1x3 {coord} <h_ZaXaZa> {ezxza_i} norm {e_id_p} <h_ZaXaZa>/norm {ezxza_i/e_id_p}")
            ezxza+= ezxza_i/e_id_p

            ezxzb_i= eval_mpo_rdm3x1_oe(coord, state, env, mpo_ZbXbZb)
            e_id_p= eval_mpo_rdm3x1_oe(coord, state, env, (id_mpo_site,)*3 )
            if verbosity>1: print(f"rdm3x1 {coord} <h_ZbXbZb> {ezxzb_i} norm {e_id_p} <h_ZbXbZb>/norm {ezxzb_i/e_id_p}")
            ezxzb+= ezxzb_i/e_id_p
        return (x/len(state.sites) for x in (eczx,ezxza,ezxzb,e_uz,e_ux))
    
    def energy_per_site(self,state,env,verbosity=0):
        eczx,ezxza,ezxzb,e_uz,e_ux= self.eval_H_ops(state, env, verbosity=verbosity)
        return self.g_zxz/2 * (ezxza + ezxzb) + self.V * (e_uz - e_ux + 2) + self.g_czx * eczx