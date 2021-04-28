import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import jq
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=0.0, help="nearest-neighbour coupling")
parser.add_argument("--q", type=float, default=1.0, help="plaquette interaction strength")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = jq.JQ(j1=args.j1, q=args.q)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.instate is not None:
        state = read_ipeps(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        state= IPEPS(dict(), lX=2, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

        sites= {(0,0): A, (1,0): B, (0,1): C, (1,1): D}
        for k in sites.keys():
            sites[k] = sites[k]/torch.max(torch.abs(sites[k]))
        state= IPEPS(sites, lX=2, lY=2)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x2_4site(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
    loss0 = model.energy_2x2_4site(state, ctm_env_init)
    obs_values, obs_labels = model.eval_obs(state,ctm_env_init)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out,  *ctm_log = ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        loss = model.energy_2x2_4site(state, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        epoch= len(opt_context["loss_history"]["loss"]) 
        loss= opt_context["loss_history"]["loss"][-1]
        obs_values, obs_labels = model.eval_obs(state,ctm_env)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = model.energy_2x2_4site(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{loss0}"]+[f"{v}" for v in obs_values]))
    
if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j1=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=2

    # basic tests
    def test_opt_GESDD(self):
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    def test_opt_fwd_checkpoint_move(self):
        args.fwd_checkpoint_move= True
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_gpu_fwd_checkpoint_move(self):
        args.GLOBALARGS_device="cuda:0"
        args.fwd_checkpoint_move= True
        main()