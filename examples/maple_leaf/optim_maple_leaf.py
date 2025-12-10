import context
import copy
import torch
import argparse
import config as cfg
import time
from groups import su2
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.ad_optim_lbfgs_mod import optimize_state
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
args, unknown_args = parser.parse_known_args()


### 2.3.1 Define Hamiltonian on maple leaf lattice
def H_mapleleaf_coarsegrained(Jd = 1.0, Jh = 1.0, Jt = 1.0):
    """
    Get H_eff as list of three two-site operators acting on three non-equivalent bonds
    of an effective triangular lattice. Each site of this triangular lattice represents 6 spin-1/2 triangle of underlying maple leaf lattice.

    $H= \sum_k h_0(k,k+(1,0)) + h_1(k,k+(0,1)) + h_2(k,k+(1,-1))$

    Get unitary U

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
    # import os
    # if os.path.isfile(f"H_eff_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt"):
    #     data = torch.load("H_eff_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt", weights_only=True)
    #     H_eff = data["H_eff"]
    #     U = data["U"]
    # else:
    H_eff, U= H_mapleleaf_coarsegrained(Jd=args.Jd, Jh=args.Jh, Jt=args.Jt)
    # torch.save({"H_eff": H_eff, "U": U}, "H_eff_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt")


    """## 2.4 Evaluate energy using MPS representation of interaction operators
    This avoids working with open large (~ 2**6) physical indices.
    ### 2.4.1 Re-express operators as mps
    Note: This step takes long on slow machines due to repeated SVDs of large matrices
    """
    # 2.4.1.1 Construct mps representation of bond Hamiltonians [if necessary] and save them in file. Otherwise, read them from file
    #   Note: For HPC simulations, this is negligible overhead
    # import os
    # if os.path.isfile("H_eff_mps_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt"):
    #     data = torch.load("H_eff_mps_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt", weights_only=True)
    #     H_eff_mps = data["H_eff_mps"]
    # else:
    # We need to permute indices of the effective Hamiltonian apporiately to get compact (small bond dimension) MPS
    H_eff_mps= [dict(zip(('U','S'),rdm.get_exact_mps(H_eff[i].reshape((dphys_1dof**ndofs,)*4).permute(0,2,1,3), min_S=1.0e-12))) for i in range(3)]
    # let's add the overall scale of the H_eff to the last mps tensor for each of the three terms
    for i in range(3):
        H_eff_mps[i]['U'][-1]= H_eff_mps[i]['U'][-1]*H_eff_mps[i]['S'][-1]
    # torch.save({"H_eff_mps": H_eff_mps,}, "H_eff_mps_Jd{args.Jd}-Jh{args.Jh}-Jt{args.Jt}.pt")


    """### 2.4.2 Define energy evaluation based on mps format of operators"""
    def get_energy_mps(state, ctm_env):

        # 2.4.2.1 This is the place, where we can conveniently insert (any) unitary rotation, i.e. by conjugating operator
        #         or in this case, its mps representation with a unitary U acting on site (1,0)
        #         The einsum expression for Tr(U[0],rho2x1,U[0]^\dag,H_eff[0]) is::
        #
        #           r2x1,[0,1,2,3],U[0],[11,1],U[0],[12,3],H_eff[0],[2,12,0,11]
        #
        #         From here, we can read how to conjugate mps representation of H_eff[0] with U[0]
        #
        h_eff_10= [
            H_eff_mps[0]['U'][0],
            H_eff_mps[0]['U'][1],
            torch.einsum('apb,ps->asb',H_eff_mps[0]['U'][2],U[0]),
            torch.einsum('apb,ps->asb',H_eff_mps[0]['U'][3],U[0])
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
            H_eff_mps[1]['U'][0],
            H_eff_mps[1]['U'][1],
            torch.einsum('apb,ps->asb',H_eff_mps[1]['U'][2],U[1]),
            torch.einsum('apb,ps->asb',H_eff_mps[1]['U'][3],U[1])
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
            H_eff_mps[2]['U'][0],
            H_eff_mps[2]['U'][1],
            torch.einsum('apb,ps->asb',H_eff_mps[2]['U'][2],U[1]),
            torch.einsum('apb,ps->asb',H_eff_mps[2]['U'][3],U[1])
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
        # Optionally ?
        if converged and verbosity>0:
            print(f"CTM-CONV {len(history['conv_crit'])} {history['conv_crit'][-1]}")
        return converged, history

    # 3.1.2 Loss function, which, given an iPEPS, first performs CTMRG until convergence
    #       and then evaluates the energy per site
    def loss_fn(state, ctm_env_in, opt_context):
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
        e_bonds, norm_bonds= get_energy_mps(state, ctm_env_out)
        loss= sum(e_bonds)
        t1_e_eval= time.perf_counter()
        print(f"time loss {t1_e_eval-t0_e_eval} [s]")

        return (loss, ctm_env_out, *ctm_log)

    """## 3.2 Run optimization
    ### 3.2.1 Configure optimization
    """
    """### 3.2.3 Define simplified variant of observables function, reporting properties of the iPEPS state through optimization"""

    # 2.2.1 Lets define spin-1/2 irrep, which can be used to get operators
    s2 = su2.SU2(dphys_1dof, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)


    @torch.no_grad()
    def f_obs_opt(state, ctm_env, opt_context):
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

            # test ENV sensitivity
            if args.test_env_sensitivity:
                loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
                loc_ctm_args.ctm_max_iter= 1
                ctm_env_out1= ctm_env.extend(ctm_env.chi+10)
                ctm_env_out1, *ctm_log= ctmrg.run(state, ctm_env_out1, \
                    conv_check=f_conv_ctm_opt, ctm_args=loc_ctm_args)
                e_bonds, norm_bonds= get_energy_mps(state, ctm_env_out1)
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
    print("epoch, loss, obs")
    f_obs_opt(state, ctm_env, {'loss_history': {'loss': [float('NaN')]}, 'ctm_args': cfg.ctm_args})

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
    optimize_state(state, ctm_env, loss_fn, obs_fn=f_obs_opt) #, post_proc=post_proc)


if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()