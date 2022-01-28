import context
import copy
import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
from ipeps.ipeps_abelian_c4v import *
from models.abelian import j1j2
from ctm.one_site_c4v_abelian.env_c4v_abelian import *
from ctm.one_site_c4v_abelian import ctmrg_c4v
from ctm.one_site_c4v_abelian.rdm_c4v import rdm2x1
# from optim.ad_optim import optimize_state
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
    # override defaults in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    settings.backend.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem
    energy_f2= model.energy_1x1

    # initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v(args.instate, settings)
        state= state.add_noise(args.instate_noise)
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p="inf")
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN_C4V(settings, None)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        rho2x1= rdm2x1(state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= (rho2x1-history["rdm"]).norm().item()
        history["rdm"]=rho2x1
        history["log"].append(dist)

        if dist<ctm_args.ctm_conv_tol or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": env.compute_multiplets()[1]})
        converged= dist<ctm_args.ctm_conv_tol and \
            not len(history['log']) >= ctm_args.ctm_max_iter
        return converged, history

    state= state.symmetrize()
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)
    print(ctm_env)
    
    # 4) (optional) compute observables as given by initial environment 
    e_curr0= energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 0) preprocess
        # create a copy of state, symmetrize and normalize making all operations
        # tracked. This does not "overwrite" the parameters tensors, living outside
        # the scope of loss_fn
        state_sym= state.symmetrize()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg_c4v.run(state_sym, ctm_env_in, 
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        
        # 2) evaluate loss with converged environment
        loss0 = energy_f(state_sym, ctm_env_out, force_cpu=args.force_cpu)

        # 2b) in case of two-branch convergence of CTM - take higher of two energies
        #     in two consecutive iterations
        loc_ctm_args= copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=args.force_cpu)

        loss= torch.max(loss0,loss1)

        return (loss, ctm_env_out, *ctm_log)

    # def _to_json(l):
    #     re=[l[i,0].item() for i in range(l.size()[0])]
    #     im=[l[i,1].item() for i in range(l.size()[0])]
    #     return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            state_sym= state.symmetrize()
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state_sym, ctm_env, force_cpu=args.force_cpu)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]+\
                [f"{state.site().norm(p='inf')}"]))

    #         if args.top_freq>0 and epoch%args.top_freq==0:
    #             coord_dir_pairs=[((0,0), (1,0))]
    #             for c,d in coord_dir_pairs:
    #                 # transfer operator spectrum
    #                 print(f"TOP spectrum(T)[{c},{d}] ",end="")
    #                 l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
    #                 print("TOP "+json.dumps(_to_json(l)))

    # def post_proc(state, ctm_env, opt_context):
    #     symm, max_err= verify_c4v_symm_A1(state.site())
    #     # print(f"post_proc {symm} {max_err}")
    #     if not symm:
    #         # force symmetrization outside of autograd
    #         with torch.no_grad():
    #             symm_site= make_c4v_symm(state.site())
    #             # we **cannot** simply normalize the on-site tensors, as the LBFGS
    #             # takes into account the scale
    #             # symm_site= symm_site/torch.max(torch.abs(symm_site))
    #             state.sites[(0,0)].copy_(symm_site)

    # optimize
    # optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn, post_proc=post_proc)
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v(outputstatefile, settings)
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