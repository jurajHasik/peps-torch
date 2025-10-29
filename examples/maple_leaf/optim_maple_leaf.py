import context
import copy
import torch
import argparse
import config as cfg
import time
from itertools import combinations
from groups import su2
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.ad_optim_lbfgs_mod import optimize_state
from ipeps.integration_yastn import PepsAD
from ctm.generic.env_yastn import from_yastn_env_generic, from_env_generic_dense_to_yastn, \
    YASTN_ENV_INIT, YASTN_PROJ_METHOD
import yastn.yastn as yastn    
from yastn.yastn.tn.mps import Hterm, generate_mpo
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps.envs.fixed_pt import refill_env, fp_ctmrg
from yastn.yastn.tn.fpeps._peps import Peps2Layers
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--Jd", type=float, default=1., help="interactions around ?")
parser.add_argument("--Jh", type=float, default=1., help="interactions around hexagons")
parser.add_argument("--Jt", type=float, default=1., help="interactions around triangles")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--grad_type", type=str, default='default', help="gradient algo", choices=['default','fp'])
args, unknown_args = parser.parse_known_args()


### 2.3.1 Define Hamiltonian on maple leaf lattice
def H_mapleleaf_coarsegrained(Jd = 1.0, Jh = 1.0, Jt = 1.0):
    """
    Get H_eff as list of three two-site operators acting on three non-equivalent bonds
    of an effective triangular lattice. Each site of this triangular lattice represents 6 spin-1/2 triangle of underlying maple leaf lattice.

    $H= \sum_k h_0(k,k+(1,0)) + h_1(k,k+(0,1)) + h_2(k,k+(1,-1))$

    Get unitary U

    Hamiltonian terms are matrices H = \sum_{S0',S1',S0,S1} H_{S0',S1',S0,S1} |S0'>|S1'><S0|<S1| of shape (2**6,2**6,2**6,2**6) 
    """
    L = 2*6

    # Define spin operators
    Sz = np.array([[0.5, 0], [0, -0.5]])
    Sp = np.array([[0, 1], [0, 0]])
    Sm = np.array([[0, 0], [1, 0]])
    isigma_y = np.array([[0, 1], [-1, 0]])

    # Helper function for identity matrix
    def I(n):
        return np.eye(n)

    intra = [
            # Jd couplings
            (1, 2, Jd/6),
            (3, 4, Jd/6),
            (5, 6, Jd/6),
            # Jt couplings
            (2, 4, Jt/6),
            (4, 6, Jt/6),
            (6, 2, Jt/6),
            # Jh couplings
            (2, 3, Jh/6),
            (4, 5, Jh/6),
            (6, 1, Jh/6)
        ]

    # triangular direction (1,0); Unitary: R_y(2pi/3)
    inter10 = [
            (5, 7, Jt),
            (5, 8, Jh)
        ]

    # triangular direction (1,-1); Unitary: R_y(4pi/3)
    inter1_1 = [
            (5, 9, Jt),
            (6, 9, Jh)
        ]

    # triangular direction (0,-1); Unitary: R_y(2pi/3)
    inter0_1 = [
            (1, 9, Jt),
            (1, 10, Jh)
        ]

    # triangular direction (-1,0); Unitary: R_y(4pi/3)
    inter_10 = [
            (1, 11, Jt),
            (2, 11, Jh)
        ]

    # triangular direction (-1,1); Unitary: R_y(2pi/3)
    inter_11 = [
            (3, 11, Jt),
            (3, 12, Jh)
        ]

    # triangular direction (0,1); Unitary: R_y(4pi/3)
    inter01 = [
            (3, 7, Jt),
            (4, 7, Jh)
        ]

    def make_hamil_mat(J_i, s1, s2, L=2*6):
        """
        Generate S.S operator between sites s1 and s2 on Hilbert space of L spin-1/2's
        """
        [s1, s2] = sorted([s1, s2])

        # Compute Hamiltonian using Kronecker products
        hamil = J_i * np.kron(np.kron(np.kron(np.kron(I(2**(s1-1)), Sz), I(2**(s2-s1-1))), Sz), I(2**(L-s2)))
        hamil += 0.5 * J_i * np.kron(np.kron(np.kron(np.kron(I(2**(s1-1)), Sp), I(2**(s2-s1-1))), Sm), I(2**(L-s2)))
        hamil += 0.5 * J_i * np.kron(np.kron(np.kron(np.kron(I(2**(s1-1)), Sm), I(2**(s2-s1-1))), Sp), I(2**(L-s2)))

        return hamil

    def make_unitary_mat(theta):
        """
        Generate Unitary operator applied to site sites s1 on Hilbert space of L/2 spin-1/2's
        """
        uni = I(2) * np.cos(theta/2.0) - isigma_y * np.sin(theta/2.0)

        return uni

    h_eff = [
        np.zeros((2**L, 2**L)),
        np.zeros((2**L, 2**L)),
        np.zeros((2**L, 2**L))
        ]

    # Sum Hamiltonian terms for each pair in local tensor
    for s1, s2, J in intra:
        h_eff[0] += make_hamil_mat(J, s1, s2, L)
        h_eff[0] += make_hamil_mat(J, s1+6, s2+6, L)

        h_eff[1] += make_hamil_mat(J, s1, s2, L)
        h_eff[1] += make_hamil_mat(J, s1+6, s2+6, L)

        h_eff[2] += make_hamil_mat(J, s1, s2, L)
        h_eff[2] += make_hamil_mat(J, s1+6, s2+6, L)

    # triangular direction (1,0)
    for s1, s2, J in inter10:
        h_eff[0] += make_hamil_mat(J, s1, s2, L)

    # triangular direction (0,-1)
    for s1, s2, J in inter0_1:
        h_eff[1] += make_hamil_mat(J, s1, s2, L)

    # triangular direction (-1,1)
    for s1, s2, J in inter_11:
        h_eff[2] += make_hamil_mat(J, s1, s2, L)

    u = make_unitary_mat(2.0*np.pi/3)

    unitary = [
        np.kron(u,np.kron(u,np.kron(u,np.kron(u,np.kron(u,u))))), # triangular direction (1,0)
        np.kron(u,np.kron(u,np.kron(u,np.kron(u,np.kron(u,u))))), # triangular direction (0,-1)
        np.kron(u,np.kron(u,np.kron(u,np.kron(u,np.kron(u,u)))))  # triangular direction (-1,1)
        ]

    h_eff= [torch.as_tensor(t) for t in h_eff]
    unitary= [torch.as_tensor(t) for t in unitary]
    return h_eff, unitary

def spiral_u( k,r, sign=-1 ):
    """ 
    Generate spiral unitary rotation matrix for spin-1/2. Wavevector k is given in units of pi.
    
        u:= exp( sign (k.r) S^y )
    """
    theta= np.pi*( k[0]*r[0] + k[1]*r[1] )
    isigma_y = torch.as_tensor([[0, 1], [-1, 0]], dtype=torch.float64) # i[[0,-i],[i,0]]
    uni = torch.eye(2,dtype=torch.float64) * np.cos(theta/2.0) + sign * isigma_y * np.sin(theta/2.0)
    U= torch.einsum('ab,cd,ef,gh,ij,kl->acegikbdfhjl',uni,uni,uni,uni,uni,uni).reshape( (2**6,2**6) )
    return U

def H_mapleleaf_mpo_yastn(Jd = 1.0, Jh = 1.0, Jt = 1.0, global_args= cfg.global_args, convention='G'):
    """
    Build three MPOs representing Hamiltonian on maple leaf lattice coarse-grained into triangular lattice.
    Each site of triangular lattice represents 6 spin-1/2 triangle of underlying maple leaf lattice.
    Hamiltonian term H_S0S1,S0'S1' is represented in MPO form as::

                            U
        S0                  S1
        |                   |
        H_mps[0]--H_mps[1]--H_mps[2]--H_mps[3]    
                  |                   |
                  S0'                 S1'
                                      U^\dagger   

    where U is a unitary rotation associated with underlying magnetic texture.

    Returns:
        H_eff_mps: list[list[torch.Tensor]] 
            list of three 4-site MPOs
        U:  list[torch.Tensor]
            list of three unitaries associated with action of each MPO
    """
    cfg_dense= yastn.make_config(backend= 'torch', sym= 'dense', \
                                 default_dtype= global_args.dtype,  default_device= global_args.device)
    yastn_s12= yastn.operators.Spin12(**cfg_dense._asdict())

    def get_Hterms_SS(s1,s2,J):
        s1,s2=s1-1,s2-1
        return [Hterm(amplitude=J, positions=(s1,s2), operators=(yastn_s12.sz(),yastn_s12.sz())),
                Hterm(amplitude=0.5*J, positions=(s1,s2), operators=(yastn_s12.sp(),yastn_s12.sm())),
                Hterm(amplitude=0.5*J, positions=(s1,s2), operators=(yastn_s12.sm(),yastn_s12.sp()))]
    
    def get_intra_SS(Jd, Jh, Jt, offset=0):
        intra = [
            # Jd couplings
            (1, 2, Jd/6), (3, 4, Jd/6), (5, 6, Jd/6),
            # Jt couplings
            (2, 4, Jt/6), (4, 6, Jt/6), (6, 2, Jt/6),
            # Jh couplings
            (2, 3, Jh/6), (4, 5, Jh/6), (6, 1, Jh/6)
        ]
        H_terms= []
        for s1,s2,J in intra:
            H_terms+= get_Hterms_SS(s1+offset,s2+offset,J)
        return H_terms
    
    # convetion for lattice vectors 
    # 
    # We use 'G' convention for order/labeling of sites within the cluster
    #
    # 1\
    # | 6
    # 2< >5
    # | 4
    # 3/
    #
    # lattice vector conventions
    # 'S'               'G'
    #                   (-a1+a2)= a1_schmoll
    #
    #  --> a2=x          --> a1=x
    # |                 |
    # a1=y              a2=y
    # V                 V
    inter_terms,U=[],[]
    if convention in ['S']:
        inter_terms= {                                     # (a1,a2) S 
            (0,1): get_Hterms_SS(5,7,Jt)+get_Hterms_SS(5,8,Jh),   # (0,1)
            (1,0): get_Hterms_SS(3,11,Jt)+get_Hterms_SS(3,12,Jh), # (1,0)
            (1,1): get_Hterms_SS(3,7,Jt)+get_Hterms_SS(4,7,Jh),   # (1,1)
            (0,-1): get_Hterms_SS(1,11,Jt)+get_Hterms_SS(2,11,Jh),   # (0,-1)
            (-1,0): get_Hterms_SS(5,9,Jt)+get_Hterms_SS(6,9,Jh), # (-1,0)
            (-1,-1): get_Hterms_SS(1,9,Jt)+get_Hterms_SS(1,10,Jh)}   # (-1,-1)
        U= {d: spiral_u( (2./3., 2./3.),d, sign=1) for d in inter_terms.keys()}
    elif convention in ['G']:
        inter_terms= {                                     # (a1,a2) G
            (1,0): get_Hterms_SS(5,7,Jt)+get_Hterms_SS(5,8,Jh),    # (1,0)  H_eff[0]
            (0,-1): get_Hterms_SS(1,9,Jt)+get_Hterms_SS(1,10,Jh),  # (0,-1) H_eff[1]
            (-1,1): get_Hterms_SS(3,11,Jt)+get_Hterms_SS(3,12,Jh), # (-1,1) [=a1_S]
            (-1,0): get_Hterms_SS(1,5+6,Jt)+get_Hterms_SS(2,5+6,Jh),   # (-1,0)
            (0,1): get_Hterms_SS(3,1+6,Jt)+get_Hterms_SS(4,1+6,Jh),    # (0,1)
            (1,-1): get_Hterms_SS(5,3+6,Jt)+get_Hterms_SS(6,3+6,Jh)}   # (1,-1)
        U= {d: spiral_u( (2./3., 2./3.),d, sign=-1) for d in inter_terms.keys()}
    else:
        raise ValueError(f"Unknown convention {convention}")
    
    # Cast to format expected by energy evaluation functions. From dummy--O_s1,s1'--...--O_s12,s12'--dummy to 
    # 
    # dummy--M_{s1,...,s6}--M_{s1',...,s6'}--M_{s7...s12}--M_{s7'...s12'}--dummy
    #
    def _cast_mpo(mpo):
        O1= yastn.ncon( tuple(mpo[i] for i in range(0,6)), \
                   [[-0,-1,1,-7],[1,-2,2,-8],[2,-3,3,-9],[3,-4,4,-10],[4,-5,5,-11],[5,-6,-13,-12]] )
        O1= O1.to_dense()
        O1= O1.reshape([2**6,]*2 + [O1.shape[-1],])
        U0,S0,V1h= torch.linalg.svd(O1.reshape(2**6, -1), full_matrices=False, driver='gesvd' if O1.is_cuda else None)

        O2= yastn.ncon( tuple(mpo[i] for i in range(6,12)), \
                   [[-0,-1,1,-7],[1,-2,2,-8],[2,-3,3,-9],[3,-4,4,-10],[4,-5,5,-11],[5,-6,-13,-12]] )
        O2= O2.to_dense()
        O2= O2.reshape([O2.shape[0],]+[2**6,]*2)
        U2,S2,V2h= torch.linalg.svd(O2.reshape(O2.shape[0] * 2**6, -1), full_matrices=False, driver='gesvd' if O2.is_cuda else None)
        return [ U0.unsqueeze(0), (S0[:,None]*V1h).reshape(U0.shape[-1:] + O1.shape[1:]), \
                U2.reshape( O2.shape[:2] + U2.shape[-1:] ), (S2[:,None]*V2h).unsqueeze(-1) ]

    h_mpos= {d: generate_mpo(yastn_s12.I(), opts_svd=None, N=2*6,\
        terms= get_intra_SS(Jd, Jh, Jt, offset=0)+get_intra_SS(Jd, Jh, Jt, offset=6)+inter_terms[d]) for d in inter_terms.keys()}

    return {d: _cast_mpo(h_mpos[d]) for d in h_mpos.keys()}, U


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(cfg.global_args.torch_dtype)
    torch.set_default_device(cfg.global_args.device)

    
    dphys_1dof=2 # spin-1/2
    ndofs= 6     # number of spins coarse-grained into a single site
    phys_dim=dphys_1dof**ndofs

    """Choose/Load iPEPS state"""
    if args.instate!=None:
        # option 2: Read state from *.json file, i.e. like the ones saved throughout optimization
        state = read_ipeps(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
    elif args.opt_resume is not None:
        # option 3: Read state from *_checkpoint.p file, i.e. like the ones saved throughout optimization
        state= IPEPS(dict(), lX=1, lY=1)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        # 1.1 we create a random D=1 on-site tensor. This corresponds to product state wavefunction
        #     of "6 spin-1/2" triangles. Correlated states within "6 spin-1/2" triangles
        #     are represented exactly.                           
        on_site_tensor= torch.rand( (phys_dim,)+(args.bond_dim,)*4 )
        state= IPEPS(sites={(0,0): on_site_tensor})
    elif args.ipeps_init_type=='MF':
        # option 1: Use prebuilt test state and extend its bond dimension and add noise
        #     For example, a antiferromagnetic classical mean field states, i.e. a product state of underlying spin-1/2s
        #     is given by tensor product of individual spin states
        # Define spin-up and spin-down states
        spin_up = torch.as_tensor([1.0, 0.0])
        spin_down = torch.as_tensor([0.0, 1.0])

        # Create a list of alternating spin states: [spin_up, spin_down, spin_up, ...]
        spin_states = [(spin_up if i % 2 == 0 else spin_down, [i]) for i in range(ndofs)]

        # Compute the tensor product using einsum
        on_site_tensor = torch.einsum(
            *sum(tuple(spin_states), ()),
            ).reshape((phys_dim,) + (1,) * 4)
        state= IPEPS(sites={(0,0): on_site_tensor})
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)

    state.add_noise(args.instate_noise)
    print(state)

    """2.3.1.1 Construct bond Hamiltonians [if necessary] and unitaries and save them in file. Otherwise, read them from file
    #         Note: For HPC simulations, this is negligible overhead"""
    import os
    _test_Heff_path= f"H_eff_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt"
    if os.path.isfile(_test_Heff_path):
        data = torch.load(_test_Heff_path, weights_only=True)
        H_eff = data["H_eff"]
        U = data["U"]
        print(f"Loaded H_eff and U from {_test_Heff_path}")
    else:
        t0= time.perf_counter()
        H_eff, U= H_mapleleaf_coarsegrained(Jd=args.Jd, Jh=args.Jh, Jt=args.Jt)
        torch.save({"H_eff": H_eff, "U": U}, _test_Heff_path)
        t1= time.perf_counter()
        print(f"Constructed H_eff and U in {t1-t0} [s]. Saving to {_test_Heff_path}")

    """## 2.4 Evaluate energy using MPS representation of interaction operators
    This avoids working with open large (~ 2**6) physical indices.
    ### 2.4.1 Re-express operators as mps
    Note: This step takes long on slow machines due to repeated SVDs of large matrices
    """
    # 2.4.1.1 Construct mps representation of bond Hamiltonians [if necessary] and save them in file. Otherwise, read them from file
    #   Note: For HPC simulations, this is negligible overhead
    import os
    _test_HeffMPS_path= f"H_eff_mps_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt"
    if os.path.isfile(_test_HeffMPS_path):
        data = torch.load(_test_HeffMPS_path, weights_only=True)
        H_eff_mps = data["H_eff_mps"]
        U_mps= data["U_mps"]
        print(f"Loaded H_eff_mps from {_test_HeffMPS_path}")
    else:
        t0= time.perf_counter()
        H_eff_mps, U_mps= H_mapleleaf_mpo_yastn(Jd=args.Jd, Jh=args.Jh, Jt=args.Jt, global_args= cfg.global_args, convention='G')
        torch.save({"H_eff_mps": H_eff_mps, "U_mps": U_mps}, _test_HeffMPS_path)
        t1= time.perf_counter()
        print(f"Constructed H_eff_mps in {t1-t0} [s]. Saving to {_test_HeffMPS_path}")

    def get_energy(state, ctm_env):
        # 2.3.2.1 let's get 2-site RDM of (1,0) bond  (2x1 or horizontal bond or 2 columns x 1 row)
        #
        #       -->x
        #      |   site(0,0) site(1,0) => with unitary s0 (U s1 U^\dag)
        #      V
        #      y
        #
        #         and evaluate spin-spin interactions between all possible bonds of two triangles (containing 6 coarse grained sites).
        #         The indices are sorted as |ket>_site(0,0)|ket>_site(1,0)<bra|_site(0,0)<bra|_site(1,0)
        r2x1 = rdm.rdm2x1((0,0), state, ctm_env, mode='sl', sym_pos_def=False, force_cpu=False,
            unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)

        # 2.3.2.2 This is the place, where we can conveniently insert (any) unitary rotation, i.e. by conjugating RDM
        #         with a unitary U acting on site (1,0)
        #
        r2x1= torch.einsum(r2x1,[0,1,2,3],U[0],[11,1],U[0],[12,3],[0,11,2,12])
        ebond_2x1= torch.einsum('ijab,abij',r2x1,H_eff[0].reshape((dphys_1dof**ndofs,)*4))

        # 2.3.2.3 Repeat this evaluation for remaining non-equivalents bonds (using appropriate U if present)
        #         RDM of (0,1) bond (1x2 or vertical or 2 columns x 1 row)
        #
        #       -->x
        #      |   site(0,0) => with unitary  s0
        #      V   site(0,1)                 (U s1 U^\dag)
        #      y
        #
        r1x2 = rdm.rdm1x2((0,0), state, ctm_env, mode='sl', sym_pos_def=False, force_cpu=False,
            unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)
        r1x2= torch.einsum(r1x2,[0,1,2,3],U[1],[11,1],U[1],[12,3],[0,11,2,12])
        ebond_1x2= torch.einsum('ijab,abij',r1x2,H_eff[1].reshape((dphys_1dof**ndofs,)*4))

        # RDM of (1,-1) (diagonal of 2 columns x 2 rows)
        #
        #       -->x
        #      |   x         site(1,-1) => with unitary  x  (U s1 U^\dag)
        #      V   site(0,0) x                           s0  x
        #      y
        #
        r2x2_1n1 = rdm.rdm2x2_NNN_1n1((0,0), state, ctm_env, sym_pos_def=False, force_cpu=False,
        unroll=False, checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)
        r2x2_1n1= torch.einsum(r2x2_1n1,[0,1,2,3],U[2],[11,1],U[2],[12,3],[0,11,2,12])
        ebond_2x2_1n1= torch.einsum('ijab,abij',r2x2_1n1,H_eff[2].reshape((dphys_1dof**ndofs,)*4))

        return ebond_2x1, ebond_1x2, ebond_2x2_1n1

    """### 2.4.2 Define energy evaluation based on mps format of operators"""
    default_H_mpos= [ H_eff_mps[ (1,0) ], H_eff_mps[ (0,1) ], H_eff_mps[ (1,-1) ] ]
    default_U= [ U_mps[ (1,0) ], U_mps[ (0,1) ], U_mps[ (1,-1) ] ]
    # previous version
    # default_H_mpos= [ H_eff_mps[ (1,0) ], H_eff_mps[ (0,-1) ], H_eff_mps[ (-1,1) ] ]
    # default_U= [ U_mps[ (1,0) ], U_mps[ (1,0) ], U_mps[ (1,0) ] ]
    def get_energy_mpo(state, ctm_env, H_mpos= default_H_mpos, U= default_U):
        """
        H_mpos: list[list[torch.Tensor]] 
            list of three 4-site MPOs representing bond Hamiltonians to be evaluated on RDMs
            rdm2x1, rdm1x2, rdm2x2_1n1 in this order
        U:  list[torch.Tensor]
            list of three unitaries associated with action of each MPO. These are used to conjugate H_eff_mps,
            always acting on second site of each bond
        """
        # 2.4.2.1 This is the place, where we can conveniently insert (any) unitary rotation, i.e. by conjugating operator
        #         or in this case, its mps representation with a unitary U acting on site (1,0)
        #         The einsum expression for Tr(U[0],rho2x1,U[0]^\dag,H_eff[0]) is::
        #
        #           r2x1,[0,1,2,3],U[0],[11,1],U[0],[12,3],H_eff[0],[2,12,0,11]
        #
        #         From here, we can read how to conjugate mps representation of H_eff[0] with U[0]
        #
        h_eff_10= [
            H_mpos[0][0],
            H_mpos[0][1],
            torch.einsum('apb,ps->asb',H_mpos[0][2],U[0]),
            torch.einsum('apb,ps->asb',H_mpos[0][3],U[0])
        ]

        # Let's evaluate this operator and the norm on 2-site RDM of (1,0) bond (2x1 or horizontal bond or 2 columns x 1 row)
        #
        #       -->x
        #      |   site(0,0) site(1,0) => with unitary s0 s1
        #      V
        #      y
        #
        ebond_2x1,I2x1 = rdm.eval_mpo_rdm2x1((0,0), state, ctm_env, h_eff_10, sym_pos_def=False, force_cpu=False,
            unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)


        # 2.4.2.2 Repeat this evaluation for remaining non-equivalents bonds (using appropriate U if present)
        #         RDM of (0,1) bond (1x2 or vertical or 2 columns x 1 row)
        #
        #       -->x
        #      |   site(0,0) => with unitary  s0
        #      V   site(0,1)                  s1
        #      y
        #
        h_eff_01= [
            H_mpos[1][0],
            H_mpos[1][1],
            torch.einsum('apb,ps->asb',H_mpos[1][2],U[1]),
            torch.einsum('apb,ps->asb',H_mpos[1][3],U[1])
        ]
        ebond_1x2,I1x2 = rdm.eval_mpo_rdm1x2((0,0), state, ctm_env, h_eff_01, sym_pos_def=False, force_cpu=False,
            unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)

        # RDM of (1,-1) (diagonal of 2 columns x 2 rows)
        #
        #       -->x
        #      |   x         site(1,-1) => with unitary  x  s1
        #      V   site(0,0) x                           s0  x
        #      y
        #
        h_eff_1n1= [
            H_mpos[2][0],
            H_mpos[2][1],
            torch.einsum('apb,ps->asb',H_mpos[2][2],U[2]),
            torch.einsum('apb,ps->asb',H_mpos[2][3],U[2])
        ]
        ebond_2x2_1n1,I2x2_1n1 = rdm.eval_mpo_rdm2x2_NNN_1n1((0,0), state, ctm_env, h_eff_1n1, sym_pos_def=False, force_cpu=False,
            unroll=False, checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0)

        return (ebond_2x1, ebond_1x2, ebond_2x2_1n1), (I2x1, I1x2, I2x2_1n1)


    """# 3. Optimization
    ## 3.1 Define loss function
    """
    # 3.1.1 We will track convergence of CTM using spectra of CTM's corners
    #       Lets modify generic conv. check to print convergence info
    #
    #       It might be desirable to supress printing here
    #
    def f_conv_ctm_opt(*args,**kw_args):
        verbosity= kw_args.pop('verbosity',0)
        converged, history= ctmrg_conv_specC(*args,**kw_args)
        #
        # Use state, env, and history here for detailed reporting/debugging
        #state, env, _= args
        #print(f"{len(history['conv_crit'])}\n{env.get_spectra()}")
        # Optionally ?
        if converged and verbosity>0:
            print(f"CTM-CONV {len(history['conv_crit'])} {history['conv_crit'][-1]}")
        return converged, history

    # 3.1.2 Loss function, which, given an iPEPS, first performs CTMRG until convergence
    #       and then evaluates the energy per site
    def loss_fn_default(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        t0_ctm= time.perf_counter()
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=f_conv_ctm_opt, ctm_args=ctm_args)
        ctm_env_out= ctm_env_in
        t1_ctm= time.perf_counter()
        print(f"time CTM {t1_ctm-t0_ctm} [s]")

        t0_e_eval= time.perf_counter()
        # 2) evaluate loss with the converged environment
        #
        #    Here, we can experiment with two options: Evaluating full H_eff operators
        # e_bonds= get_energy(state, ctm_env_out)
        # loss= sum(e_bonds)
        #
        # OR
        #
        # evaluate compact mps representation of H_eff operators
        e_bonds, norm_bonds= get_energy_mpo(state, ctm_env_out)
        loss= sum(e_bonds)
        t1_e_eval= time.perf_counter()
        print(f"time loss {t1_e_eval-t0_e_eval} [s]")

        return (loss, ctm_env_out, *ctm_log)

    #
    # 3.1.3 Loss function using fixed-point AD
    def loss_fn_fp(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 2. convert to YASTN's iPEPS
        state_yastn= PepsAD.from_pt(state)

        # 3. proceed with YASTN's CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            env_leg = yastn.Leg(state_yastn.config, s=1, D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.1.1 post-init CTM steps 
        options_svd_pre_init= {
            "policy": YASTN_PROJ_METHOD["RSVD"],
                "D_total": cfg.main_args.chi, 'D_block': cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet, "niter": ctm_args.projector_rsvd_niter,
            }
        with torch.no_grad():
            sweep, max_dsv, max_D, converge= ctm_env_in.ctmrg_(
                method="2site",
                max_sweeps=math.ceil(args.chi/(args.bond_dim**2)),
                opts_svd=options_svd_pre_init,
                corner_tol=ctm_args.projector_svd_reltol
            )
        log.log(logging.INFO, f"WARM-UP: Number of ctm steps: {sweep:d}, t_warm_up: N/As")

        # 3.2 setup and run CTMRG
        options_svd={
            "policy": YASTN_PROJ_METHOD[ctm_args.projector_svd_method],
            "D_total": cfg.main_args.chi, "D_block" : cfg.main_args.chi,
            "tol": ctm_args.projector_svd_reltol,
            "eps_multiplet": ctm_args.projector_eps_multiplet,
            'verbosity': ctm_args.verbosity_projectors
        }

        ctm_env_out, env_ts_slices, env_ts = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': ctm_args.ctm_conv_tol, 'max_sweeps': ctm_args.ctm_max_iter, \
                'method': "2site", 'use_qr': False,
                'checkpoint_move': 'reentrant' if ctm_args.fwd_checkpoint_move==True else ctm_args.fwd_checkpoint_move,
                },
            ctm_opts_fp= {'opts_svd': {'policy': 'fullrank'}, 'verbosity': 3,})
        refill_env(ctm_env_out, env_ts, env_ts_slices)

        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_env_generic(ctm_env_out, vertexToSite=state.vertexToSite)

        # 3.4 evaluate loss
        t_loss0= time.perf_counter()
        e_bonds, norm_bonds= get_energy_mpo(state, env_pt)
        loss= sum(e_bonds)
        t_loss1= time.perf_counter()

        return (loss, ctm_env_out, [], None, t_loss1-t_loss0)

    """## 3.2 Run optimization
    ### 3.2.1 Configure optimization
    """
    """### 3.2.3 Define simplified variant of observables function, reporting properties of the iPEPS state through optimization"""

    # 2.2.1 Lets define spin-1/2 irrep, which can be used to get operators
    s2 = su2.SU2(dphys_1dof, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)


    @torch.no_grad()
    def f_obs_opt(state, _ctm_env, opt_context):
        if isinstance(_ctm_env, EnvCTM):
            ctm_env= from_yastn_env_generic(_ctm_env, vertexToSite=state.vertexToSite)
        else:
            ctm_env= _ctm_env

        # We don't report when in line search mode
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]

            # 2.2.2 let's get 1-site RDM [normalized & symmetrized] and use it to evaluate S^z on each of the 6 spins
            #       The indices of RDM(s) are ordered as |ket><bra|
            r1x1 = rdm.rdm1x1((0,0),state,ctm_env,force_cpu=True).reshape((dphys_1dof,)*(2*ndofs)) # from 2**6 x 2**6 matrix to 12-dim tensor
            _make_r1x1_inds= lambda i: list(range(0,i))+[20+i+ndofs]+list(range(i+1,ndofs))+list(range(0,i))+[20+i]+list(range(i+1,ndofs))

            # We can get directly vector S=(Sz,Sx,Sy), where the first index runs over spin operators
            Ss= [ torch.einsum(r1x1,_make_r1x1_inds(i),s2.S(),[51,20+i,20+i+ndofs]) for i in range(ndofs) ]
            print(f"{epoch}, {loss}, "+ ", ".join([f"i={i} (Sz,Sx,Sy)= ({Ss[i]}" for i in range(len(Ss))]) )

            # compute NN spin-spin correlations
            r1x1_12= torch.einsum('IJlmnoABlmno->IJAB',r1x1)
            r1x1_34= torch.einsum('lmIJnolmABno->IJAB',r1x1)
            r1x1_56= torch.einsum('lmnoIJlmnoAB->IJAB',r1x1)
            r1x1_24= torch.einsum('lImJnolAmBno->IJAB',r1x1)
            r1x1_46= torch.einsum('lmnIoJlmnAoB->IJAB',r1x1)
            r1x1_62= torch.einsum('lJmnoIlBmnoA->IJAB',r1x1)
            for label,R in zip( ["12", "34", "56", "24", "46", "62"], [r1x1_12, r1x1_34, r1x1_56, r1x1_24, r1x1_46, r1x1_62] ):
                SS_corr= torch.einsum('IJAB,ABIJ',R,s2.SS())
                print(f"<S.S>_{label}= {SS_corr}")

            # test ENV sensitivity
            if args.test_env_sensitivity:
                loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
                loc_ctm_args.ctm_max_iter= 1
                ctm_env_out1= ctm_env.extend(ctm_env.chi+10)
                ctm_env_out1, *ctm_log= ctmrg.run(state, ctm_env_out1, \
                    conv_check=f_conv_ctm_opt, ctm_args=loc_ctm_args)
                e_bonds, norm_bonds= get_energy_mpo(state, ctm_env_out1)
                loss1= sum(e_bonds)
                delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                    if len(opt_context['loss_history']['loss'])>1 else float('NaN')
                # if we are not linesearching, this can always happen
                # not "line_search" in opt_context.keys()
                _flag_antivar= (loss1-loss)>0 and \
                    (loss1-loss)*opt_context["opt_args"].env_sens_scale>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

                print(f"env_sensitivity: {loss1-loss} loss_diff: "\
                    +f"{delta_loss}" if args.test_env_sensitivity else ""\
                    +" Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))


    """### 3.2.4 Run the optimizer"""
    # create CTMRG environment with environment bond dimension \chi (which governs the precision) and initialize it
    ctm_env= ENV(args.chi, state)
    init_env(state, ctm_env)

    # converge initial environment
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=f_conv_ctm_opt)
    # evaluate observables on initial state
    E0_bonds= get_energy_mpo(state, ctm_env)
    print("epoch, loss, obs")
    f_obs_opt(state, ctm_env, {'loss_history': {'loss': [sum(E0_bonds[0]).item()]}, 'ctm_args': cfg.ctm_args})

    # We enter optimization
    def post_proc(state, ctm_env, opt_context):
        with torch.no_grad():
            for c in state.sites.keys():
                _tmp= state.sites[c]/state.sites[c].abs().max()
                state.sites[c].copy_(_tmp)
            # if "STATUS" in opt_context and opt_context["STATUS"]=="ENV_ANTIVAR":
            #     state_g= IPEPS_WEIGHTED(state=state).gauge().absorb_weights()
            #     for c in state.sites.keys():
            #         state.sites[c].copy_(state_g.sites[c])

    # optimize
    if args.gauge:
        state_g= IPEPS_WEIGHTED(state=state).gauge()
        state= state_g.absorb_weights()
        
    state.normalize_()
    loss_fn= loss_fn_fp if args.grad_type=='fp' else loss_fn_default
    if args.grad_type=='fp':
        ctm_env= from_env_generic_dense_to_yastn(ctm_env, state)
    optimize_state(state, ctm_env, loss_fn, obs_fn=f_obs_opt) #, post_proc=post_proc)


if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()