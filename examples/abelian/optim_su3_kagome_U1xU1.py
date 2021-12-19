import context
import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1xU1_torch as settings_U1xU1
from ipeps.ipess_kagome_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import su3_kagome
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--phi", type=float, default=0.5, help="arctan(K/J): J -> 2-site coupling; K -> 3-site coupling")
parser.add_argument("--theta", type=float, default=0., help="arctan(H/K): K -> 3-site coupling; K -> chiral coupling")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
parser.add_argument("--symmetry", default=None, help="symmetry structure", choices=["NONE","U1xU1"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    # Parametrization
    param_j = np.round(np.cos(np.pi*args.phi), decimals=15)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=15)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=15)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    # TODO(?) choose symmetry group
    if not args.symmetry or args.symmetry=="NONE":
        settings= settings_full
    elif args.symmetry=="U1xU1":
        settings= settings_U1xU1
    # override default device specified in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    # override default dtype
    settings_full.default_dtype= settings.default_dtype= cfg.global_args.dtype
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)
    
    # the model (in particular operators forming Hamiltonian) is defined in a dense form
    # with no symmetry structure
    model= su3_kagome.KAGOME_SU3_U1xU1(settings,j=param_j,k=param_k,h=param_h,global_args=cfg.global_args)

    # initialize an ipeps
    if args.instate!=None:
        state= read_ipess_kagome_generic(args.instate, settings)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        T_u= torch.zeros(config=settings, s=(-1,-1,-1))
        T_d= torch.zeros(config=settings, s=(-1,-1,-1))
        B_c= torch.zeros(config=settings, s=(-1,1,1))
        B_a= torch.zeros(config=settings, s=(-1,1,1))
        B_b= torch.zeros(config=settings, s=(-1,1,1))
        state= IPESS_KAGOME_GENERIC_ABELIAN(settings, {'T_u': T_u, 'B_a': B_a,\
            'T_d': T_d, 'B_b': B_b, 'B_c': B_c})
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) select the "energy" function
    energy_f_down_t_1x1subsystem = model.energy_down_t_1x1subsystem
    energy_f = model.energy_per_site_2x2subsystem
    
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        # simply use single down triangle energy to evaluate the CTMRG convergence
        e_curr= energy_f_down_t_1x1subsystem(state, env).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 0) force recomputation of on-site tensors in the active autograd context
        #    Otherwise, this operation is not recorded and hence not differentiated
        state.sites= state.build_onsite_tensors()

        # 1) build double-layer open on-site tensors
        #    Some objects, in this case open-double layer tensors, are pre-computed
        state.build_sites_dl_open()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        
        # 2) evaluate loss with the converged environment
        loss= energy_f(state, ctm_env_out)

        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state,ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipess_kagome_generic(outputstatefile, settings)
    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()