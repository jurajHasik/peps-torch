import context
import copy
import torch
import argparse
import config as cfg
import yastn.yastn as yastn

from ipeps.ipeps_abelian_c4v_lc import *
from ipeps.integration_yastn import PepsAD
from models.abelian import j1j2

from ctm.one_site_c4v_abelian.env_c4v_abelian import *

import time
from ctm.generic.env_yastn import ctmrg, YASTN_ENV_INIT
from ctm.generic_abelian.env_yastn import from_yastn_env_generic
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint, env_raw_data, refill_env
from yastn.yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg

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
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
parser.add_argument("--grad_type", type=str, default='default', help="gradient algo", choices=['default','fp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings_full= yastn.make_config(backend=backend, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)
    settings.backend.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg

    # model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    # energy_f= model.energy_1x1_lowmem
    model= j1j2.J1J2_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    energy_f= model.energy_2x1_or_2Lx2site_2x2rdms

    # initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v_lc(args.instate, settings)
        state= state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')()
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN_C4V_LC(settings, None, dict(), None)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    # 2) convergence criterion based spectra of corner tensors
    @torch.no_grad()
    def yastn_ctm_conv_check(env,history,corner_tol):
        converged,max_dsv,history= ctm_conv_corner_spec(env,history,corner_tol)
        return converged, history

    def loss_fn_default(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 1. build on-site tensors from su2sym components
        #    Here, for C4v-symmetric single-site iPEPS
        # state.coeffs[(0,0)]= state.coeffs[(0,0)]/state.coeffs[(0,0)].abs().max()
        state.sites[(0,0)]= state.build_onsite_tensors()
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')

        # 2. convert to 2-site bipartite YASTN's iPEPS
        state_bp= state.get_bipartite_state()
        state_yastn= PepsAD.from_pt(state_bp)

        # 3. proceed with YASTN's CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)

        # 3.2 setup and run CTMRG
        options_svd={
            "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }
        _ctm_conv_f= lambda _x,_y: yastn_ctm_conv_check(_x,_y,ctm_args.ctm_conv_tol)
        ctm_env_out, converged, conv_history, t_ctm, t_check= ctmrg(ctm_env_in, _ctm_conv_f,  options_svd,
                    max_sweeps=ctm_args.ctm_max_iter, 
                    method="2site", 
                    use_qr=False,
                    checkpoint_move=ctm_args.fwd_checkpoint_move
                    )


        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_env_generic(ctm_env_out, vertexToSite=state_bp.vertexToSite)

        # 3.4 evaluate loss
        t_loss0= time.perf_counter()
        loss= energy_f(state_bp, env_pt).to_number()
        t_loss1= time.perf_counter()

        return (loss, ctm_env_out, conv_history, t_ctm, t_check)


    def loss_fn_fp(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 1. build on-site tensors from su2sym components
        #    Here, for C4v-symmetric single-site iPEPS
        # state.coeffs[(0,0)]= state.coeffs[(0,0)]/state.coeffs[(0,0)].abs().max()
        state.sites[(0,0)]= state.build_onsite_tensors()
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')

        # 2. convert to 2-site bipartite YASTN's iPEPS
        state_bp= state.get_bipartite_state()
        state_yastn= PepsAD.from_pt(state_bp)

        # 3. proceed with YASTN's CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)

        # 3.2 setup and run CTMRG
        options_svd={
            "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }

        # env_serialized= env_raw_data(ctm_env_in)
        # ctm_env_out, slices_out, env_out_data= FixedPoint.apply(*env_serialized, ctm_env_in.config, ctm_env_in, options_svd, options_svd['D_total'], 
        #                                ctm_args.ctm_conv_tol, ctm_args, *state_yastn.get_parameters() )
        # # ctm_env_out= EnvCTM(state_yastn, init=None)
        # refill_env(ctm_env_out, env_out_data, slices_out)

        ctm_env_out, env_ts_slices, env_ts = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': ctm_args.ctm_conv_tol, 'max_sweeps': ctm_args.ctm_max_iter, \
                'method': "2site", 'use_qr': False, 'svd_policy': 'fullrank', 'D_block': None}, \
            ctm_opts_fp= {'svd_policy': 'fullrank'})
        refill_env(ctm_env_out, env_ts, env_ts_slices)

        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_env_generic(ctm_env_out, vertexToSite=state_bp.vertexToSite)

        # 3.4 evaluate loss
        t_loss0= time.perf_counter()
        loss= energy_f(state_bp, env_pt).to_number()
        t_loss1= time.perf_counter()

        return (loss, ctm_env_in, [], None, None)


    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state_bp= state.get_bipartite_state()
        env_pt= from_yastn_env_generic(ctm_env, vertexToSite=state_bp.vertexToSite)

        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state_bp,env_pt,force_cpu=args.force_cpu)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
            +[f"{state.coeffs[(0,0)].abs().max()}"]))

    # optimize
    if args.grad_type=='default':
        loss_fn= loss_fn_default
    elif args.grad_type=='fp':
        loss_fn= loss_fn_fp
    optimize_state(state, None, loss_fn, obs_fn=obs_fn)
    
if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()