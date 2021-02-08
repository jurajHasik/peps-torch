import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from models import jq
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--c4v_type", default="TI", help="type of C4v ansatz. Supported types:"\
    +"TI, BIPARTITE, PLAQUETTE")
parser.add_argument("--j1", type=float, default=0.0, help="nearest-neighbour coupling")
parser.add_argument("--q", type=float, default=1.0, help="plaquette interaction strength")
parser.add_argument("--q_inter", type=float, default=None, help="relevant for c4v_type=PLAQUETTE")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    if args.c4v_type == "TI": 
        model= jq.JQ_C4V(j1=args.j1, q=args.q)
    elif args.c4v_type == "BIPARTITE": 
        model= jq.JQ_C4V_BIPARTITE(j1=args.j1, q=args.q)
    elif args.c4v_type == "PLAQUETTE":
        q_inter= args.q if args.q_inter is None else args.q_inter
        model= jq.JQ_C4V_PLAQUETTE(j1=args.j1, q=args.q, q_inter=q_inter)
    else:
        raise ValueError("Unsupported C4v ansatz: -c4v_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    
    if args.instate!=None:
        state= read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.opt_resume is not None:
        state= IPEPS_C4V()
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_1x1(state, env)
        history.append(e_curr.item())
        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)

    ctm_env, *ctm_log= ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss= model.energy_1x1(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # symmetrize on-site tensor
        state_sym= to_ipeps_c4v(state, normalize=True)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg_c4v.run(state_sym, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss = model.energy_1x1(state_sym, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state,ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = model.energy_1x1(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

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
        args.opt_max_iter=3

    # basic tests
    def test_opt_SYMEIG_BIPARTITE(self):
        args.c4v_type="BIPARTITE"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_BIPARTITE_gpu(self):
        args.c4v_type="BIPARTITE"
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()