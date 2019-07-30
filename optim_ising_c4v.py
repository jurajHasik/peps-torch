import torch
import argparse
from args import *
from ipeps import *
from c4v import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from models import ising
from ad_optim import optimize_state

# additional model-dependent arguments
parser.add_argument("-hx", type=float, default=0., help="transverse field")
parser.add_argument("-q", type=float, default=0., help="next nearest-neighbour coupling")
args = parser.parse_args()
torch.set_num_threads(args.omp_cores)

if __name__=='__main__':
    
    model = ising.ISING_C4V(hx=args.hx, q=args.q)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site, \
            peps_args=PEPSARGS(), global_args=GLOBALARGS())
        assert len(state.sites)==1, "Not a 1-site ipeps"
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        add_random_noise(state, args.instate_noise)
        state.sites[(0,0)]= make_c4v_symm(state.sites[(0,0)])
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))

        sites = {(0,0): A}

        state = IPEPS(sites)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    def ctmrg_conv_energy(state, env, history, ctm_args = CTMARGS()):
        with torch.no_grad():
            e_curr = model.energy_1x1(state, env)
            history.append(e_curr.item())

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, history, t_ctm = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    loss= model.energy_1x1(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, ctm_args= CTMARGS(), opt_args= OPTARGS(), global_args= GLOBALARGS()):
        # symmetrize on-site tensor
        symm_sites= {(0,0): make_c4v_symm(state.sites[(0,0)])}
        symm_state= IPEPS(symm_sites)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(symm_state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_in, history, t_ctm= ctmrg_c4v.run(symm_state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_1x1(symm_state, ctm_env_in)
        
        return loss, ctm_env_in, history, t_ctm

    # optimize
    optimize_state(state, ctm_env, loss_fn, model, args)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, peps_args=PEPSARGS(), global_args=GLOBALARGS())
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, history, t_ctm = ctmrg_c4v.run(state, ctm_env)
    opt_energy = model.energy_1x1(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))
    