import copy
import torch
import argparse
import config as cfg
from examples.abelian.settings_full_torch import settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
import yamps.tensor as TA
from ipeps.ipeps_abelian_c4v_lc import *
from models.abelian import j1j2
from ctm.one_site_c4v_abelian.env_c4v_abelian import *
from ctm.one_site_c4v_abelian import ctmrg_c4v
from ctm.one_site_c4v_abelian.rdm_c4v import rdm2x1
from optim.ad_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
parser.add_argument("--symmetry", default=None, help="symmetry structure", choices=["NONE","U1"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    # TODO(?) choose symmetry group
    if not args.symmetry or args.symmetry=="NONE":
        settings= settings_full
    elif args.symmetry=="U1":
        settings= settings_U1
    # override default device specified in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    settings.back.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg
    settings.back.set_num_threads(args.omp_cores)
    settings.back.random_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v_lc(args.instate, settings)
        state= state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].max_abs()
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN_C4V_LC(settings, None, None, None)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    # 2) convergence criterion based on 2-site reduced density matrix 
    #    of nearest-neighbours
    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        rho2x1= rdm2x1(state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= (rho2x1-history["rdm"]).norm().item()

        # log dist and observables
        if args.obs_freq>0 and \
            (len(history["log"])%args.obs_freq==0 or 
            (len(history["log"])-1)%args.obs_freq==0):
            e_curr = energy_f(state, env, force_cpu=args.force_cpu)
            obs_values, obs_labels = model.eval_obs(state, env, force_cpu=args.force_cpu)
            print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]\
                +[f"{v}" for v in obs_values]))
        # else:
        #    print(f"{len(history['log'])}, {dist}")

        # update history and check convergence
        history["rdm"]=rho2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": env.compute_multiplets()[1]})
        converged= dist<ctm_args.ctm_conv_tol and \
            not len(history['log']) >= ctm_args.ctm_max_iter
        return converged, history

    # 3) initialize environment
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)
    print(ctm_env)
    
    # 4) (optional) compute observables as given by initial environment 
    e_curr0= energy_f(state,ctm_env,force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build on-site tensors from su2sym components
        # state.coeffs[(0,0)]= state.coeffs[(0,0)]/state.coeffs[(0,0)].abs().max()
        state.sites[(0,0)]= state.build_onsite_tensors()
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].max_abs()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        loss0 = energy_f(state, ctm_env_out, force_cpu=args.force_cpu)
        
        loc_ctm_args= copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=args.force_cpu)

        #loss=(loss0+loss1)/2
        loss= torch.max(loss0,loss1)

        return loss, ctm_env_out, history, t_ctm, t_obs

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
            +[f"{state.coeffs[(0,0)].abs().max()}"]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)
    
    # 6) compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v_lc(outputstatefile, settings)
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env,force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()