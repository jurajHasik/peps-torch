import context
import argparse
import torch
import config as cfg
import yastn.yastn as yastn
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.sym import sym_U1
from ipeps.ipeps_abelian import *
from ctm.generic.env import *
import ctm.generic.ctmrg as ctmrg
from models import coupledLadders
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
#from ctm.generic import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--alpha", type=float, default=0., help="inter-ladder coupling")
parser.add_argument("--bz_stag", type=float, default=0., help="staggered magnetic field")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    model= coupledLadders.COUPLEDLADDERS(alpha=args.alpha, bz_val=args.bz_stag)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.instate!=None:
        state= read_ipeps(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    # TODO checkpointing
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN(settings, dict(), lX=2, lY=2)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= coupledLadders.COUPLEDLADDERS(alpha=args.alpha, bz_val=args.bz_stag)

    print(state)
    # convert to dense
    state_d= state.to_dense()
    print(state_d)

    @torch.no_grad()
    def ctmrg_conv_energy(state_d, env_d, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x1_1x2(state_d, env_d).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_d = ENV(args.chi, state_d)
    init_env(state_d, ctm_env_d)
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env_d, *ctm_log = ctmrg.run(state_d, ctm_env_d, conv_check=ctmrg_conv_f)
    loss= model.energy_2x1_1x2(state_d, ctm_env_d)
    obs_values, obs_labels= model.eval_obs(state_d,ctm_env_d)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in_d, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        state_d= state.to_dense()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_d, ctm_env_in_d)

        # 1) compute environment by CTMRG
        ctm_env_out_d, *ctm_log = ctmrg.run(state_d, ctm_env_in_d, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss= model.energy_2x1_1x2(state_d, ctm_env_out_d)

        return (loss, ctm_env_out_d, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env_d, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        state_d= state.to_dense()
        obs_values, obs_labels = model.eval_obs(state_d,ctm_env_d)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

        # with torch.no_grad():
        #     if (not opt_context["line_search"]) and args.top_freq>0 \
        #         and epoch%args.top_freq==0:
        #         coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
        #         for c,d in coord_dir_pairs:
        #             # transfer operator spectrum
        #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
        #             l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
        #             print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env_d, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, settings)
    state_d= state.to_dense()
    ctm_env_d = ENV(args.chi, state_d)
    init_env(state_d, ctm_env_d)
    ctm_env_d, *ctm_log = ctmrg.run(state_d, ctm_env_d, conv_check=ctmrg_conv_f)
    opt_energy = model.energy_2x1_1x2(state_d,ctm_env_d)
    obs_values, obs_labels = model.eval_obs(state_d,ctm_env_d)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()