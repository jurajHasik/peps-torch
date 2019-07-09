import torch
from args import *
import env
from env import ENV
from ipeps import *
import ctmrg
import rdm
from env import *
from models import coupledLadders
from ad_optim import optimize_state
from IPython import embed

torch.set_num_threads(args.omp_cores)

if __name__=='__main__':

    torch.set_printoptions(precision=7)

    model = coupledLadders.COUPLEDLADDERS(alpha=0.0)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    

    if args.instate!=None:
        state = read_ipeps(args.instate, peps_args=PEPSARGS(), global_args=GLOBALARGS())
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim, args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
        D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)

        sites = {(0,0): A, (1,0): B, (0,1): C, (1,1): D}

        for k in sites.keys():
            sites[k] = sites[k]/torch.max(torch.abs(sites[k]))
        state = IPEPS(sites, lX=2, lY=2)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    def ctmrg_conv_energy(state, env, history, ctm_args = CTMARGS()):
        with torch.no_grad():
            e_curr = model.energy_2x1_1x2(state, env)
            history.append(e_curr.item())

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                return True
        return False

    def loss_fn(state, ctm_env_init, ctm_args= CTMARGS(), opt_args= OPTARGS(), global_args= GLOBALARGS()):
        # 0) pre-process state: normalize on-site tensors by largest elements
        for coord,site in state.sites.items():
            site = site/torch.max(torch.abs(site))
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_init)

        # 1) compute environment by CTMRG
        ctm_env = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_2x1_1x2(state, ctm_env)
        
        # compute,print observables
        obs= model.eval_obs(state,ctm_env)
        for label,val in obs.items():
            print(f"{label} {val:.3E}", end=" ")
        print("")

        return loss, ctm_env

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, verbosity=1)

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env = ctmrg.run(state, ctm_env)
    opt_energy = model.energy_2x1_1x2(state,ctm_env)
    print(f"(opt state) E(2x1+1x2): {opt_energy}")
    