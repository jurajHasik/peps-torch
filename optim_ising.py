import torch
import argparse
from args import *
from ipeps import *
from env import *
import ctmrg
from models import ising
from ad_optim import optimize_state

# additional model-dependent arguments
parser.add_argument("-hx", type=float, default=0., help="transverse field")
parser.add_argument("-q", type=float, default=0., help="next nearest-neighbour coupling")
args = parser.parse_args()
torch.set_num_threads(args.omp_cores)

if __name__=='__main__':
    
    model = ising.ISING(hx=args.hx, q=args.q)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site, \
            peps_args=PEPSARGS(), global_args=GLOBALARGS())
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        add_random_noise(state, args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))

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

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    ctm_env_init = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    loss = model.energy_1x1(state, ctm_env_init)
    obs_values, obs_labels = model.eval_obs(state,ctm_env_init)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_init, ctm_args= CTMARGS(), opt_args= OPTARGS(), global_args= GLOBALARGS()):
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_init)

        # 1) compute environment by CTMRG
        ctm_env = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_1x1(state, ctm_env)
        
        return loss, ctm_env

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, model, args)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, peps_args=PEPSARGS(), global_args=GLOBALARGS())
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env = ctmrg.run(state, ctm_env)
    opt_energy = model.energy_1x1(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))
    