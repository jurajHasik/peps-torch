import os
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
from ctm.generic.env_yastn import ctmrg, YASTN_ENV_INIT, YASTN_PROJ_METHOD
from ctm.generic_abelian.env_yastn import *
from yastn.yastn.tn.fpeps import EnvCTM, EnvCTM_c4v
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps.envs.rdm import rdm1x2
from yastn.yastn.tn.fpeps.envs.fixed_pt import refill_env, fp_ctmrg
from yastn.yastn.tn.fpeps.envs.fixed_pt_c4v import refill_env_c4v, fp_ctmrg_c4v
from yastn.yastn.tn.fpeps._peps import Peps2Layers

from optim.ad_optim_lbfgs_mod import optimize_state

import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument(
    "--u1_charges",
    dest="u1_charges",
    default=None,
    type=int,
    help="U(1) charges assigned to the states of the local physical Hilbert space followed by charges of states in the virtual space",
    nargs="+",
)
parser.add_argument(
    "--u1_total_charge",
    dest="u1_total_charge",
    default=0,
    type=int,
    help="total U(1) charge",
)
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "RDM"])
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
parser.add_argument("--yast_backend", type=str, default='torch',
    help="YAST backend", choices=['torch','torch_cpp'])
parser.add_argument("--grad_type", type=str, default='default', help="gradient algo", choices=['default','fp','c4v', 'c4v_fp'])
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
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    settings.backend.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg

    if args.grad_type in ['default','fp']:
        model= j1j2.J1J2_NOSYM(settings_full, j1=args.j1, j2=args.j2)
        energy_f= model.energy_2x1_or_2Lx2site_2x2rdms
        obs_f= model.eval_obs
    elif args.grad_type in ['c4v', 'c4v_fp']:
        if cfg.ctm_args.projector_svd_method=='DEFAULT':
            cfg.ctm_args.projector_svd_method= 'GESDD' if args.grad_type=='c4v' else 'QR'
        model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
        energy_f= model.energy_1x1_lowmem


    # initialize the ipeps
    if args.instate!=None and not args.u1_charges:
        state= read_ipeps_c4v_lc(args.instate, settings)
        state= state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')()
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN_C4V_LC(settings, None, dict(), None)
        state.load_checkpoint(args.opt_resume)
    elif args.u1_charges and len(args.u1_charges)==args.bond_dim+2:
        u1basis= generate_a_basis(2, args.bond_dim, args.u1_charges, args.u1_total_charge)
        coeffs= torch.zeros(len(u1basis), dtype=cfg.global_args.torch_dtype, device='cpu').to(cfg.global_args.device)
        # if instate is passed together with u1_charges, we assume instate serves as a reference to be extended
        if args.instate!=None:
            ref_state= read_ipeps_c4v_lc(args.instate, settings)
            coeffs= torch.as_tensor(rebase_params(ref_state.coeffs[(0,0)], torch.stack([m_t[1] for m_t in ref_state.elem_tensors ]),\
                                  u1basis, args.instate_noise, D=None), dtype=cfg.global_args.torch_dtype).to(cfg.global_args.device)
        else:
            coeffs= torch.rand_like(coeffs)
        u1basis= [ ( {"meta": {"pg": "A_1",}} ,t) for i,t \
                  in enumerate(u1basis.to(dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device).unbind(dim=0)) ]
        state= IPEPS_ABELIAN_C4V_LC(
            settings, u1basis,
            {(0,0): coeffs},
            {"abelian_charges": args.u1_charges, "total_abelian_charge": args.u1_total_charge},
        )
        
    else:
        raise ValueError("Missing trial state: (--instate=None or empty --u1_charges) and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) convergence criterion based spectra of corner tensors
    @torch.no_grad()
    def yastn_ctm_conv_check_cspec(env,history,corner_tol):
        converged,max_dsv,history= ctm_conv_corner_spec(env,history,corner_tol)
        # logging ?
        # C= history[0][((0,0),'tl')]
        # print(f"{max_dsv}" + str([f"{b} {C[b].shape[0]}" for b in C.get_blocks_charge() ]))
        return converged, history
    
    @torch.no_grad()
    def ctmrg_conv_rdm2x1(env,history,corner_tol): 
        if not history:
            history=dict({"log": []})
        env_bp = env.get_env_bipartite()
        R, R_norm= rdm1x2( (0,0), env_bp.psi.ket, env_bp, )
        # logging ?
        #D,_= R.fuse_legs(axes=((0,2),(1,3))).eigh(axes=(0,1))
        #print(f"{[x.item() for x in D.to_dense().diag().sort()[0]]}")
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= (R - history["rdm"]).norm().item()
        # update history
        history["rdm"]=R
        history["log"].append(dist)
        print(dist)

        converged= dist<corner_tol
        if converged:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return converged, history
        return False, history

    ctm_conf_f= ctmrg_conv_rdm2x1 if (args.grad_type in ['c4v','c4v_fp'] and args.ctm_conv_crit=="RDM") else yastn_ctm_conv_check_cspec


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
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.2 setup and run CTMRG
        options_svd={
            "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }
        _ctm_conv_f= lambda _x,_y: ctm_conf_f(_x,_y,ctm_args.ctm_conv_tol)
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
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.2 setup and run CTMRG
        options_svd={
            "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }

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

        return (loss, ctm_env_out, [], None, None)


    def loss_c4v(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 1. build on-site tensors from su2sym components
        #    Here, for C4v-symmetric single-site iPEPS
        # state.coeffs[(0,0)]= state.coeffs[(0,0)]/state.coeffs[(0,0)].abs().max()
        state.sites[(0,0)]= state.build_onsite_tensors()
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')

        # 2. convert to 1-site YASTN's iPEPS
        state_yastn= PepsAD.from_pt(state)

        # 3. proceed with YASTN's C4v-CTMRG implementation

        # 3.2 setup and run CTMRG
        options_svd={
                "policy": YASTN_PROJ_METHOD[ctm_args.projector_svd_method],
                "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }
        _ctm_conv_f= lambda _x,_y: ctm_conf_f(_x,_y,ctm_args.ctm_conv_tol)

        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            with torch.no_grad():
                env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
                ctm_env_in = EnvCTM_c4v(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
                # 3.2.1 post-init CTM steps (allow expansion of the environment in case of qr policy)        
                options_svd_pre_init= {**options_svd} 
                options_svd_pre_init.update({"policy": "arnoldi"})
                ctm_env_in, converged, conv_history, t_ctm, t_check= ctmrg(ctm_env_in, _ctm_conv_f,  options_svd_pre_init,
                    max_sweeps= ctm_args.fpcm_init_iter,
                    method="default",
                    checkpoint_move=False
                    )
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.2.2 run CTMRG
        ctm_env_out, converged, conv_history, t_ctm, t_check= ctmrg(ctm_env_in, _ctm_conv_f,  options_svd,
                    max_sweeps=ctm_args.ctm_max_iter,
                    method="default",
                    checkpoint_move=ctm_args.fwd_checkpoint_move
                    )
        print(f"t_ctm: {t_ctm:.1f}s")
        log.log(logging.INFO, f"# of ctm steps: {len(conv_history):d}, t_ctm: {t_ctm:.1f}s")

        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_c4v_env_c4v(ctm_env_out)

        # 3.4 evaluate loss
        t_loss0= time.perf_counter()
        loss= energy_f(state, env_pt)
        t_loss1= time.perf_counter()

        return (loss, ctm_env_out, conv_history, t_ctm, t_check, t_loss1-t_loss0)


    def loss_c4v_fp(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 1. build on-site tensors from su2sym components
        #    Here, for C4v-symmetric single-site iPEPS
        # state.coeffs[(0,0)]= state.coeffs[(0,0)]/state.coeffs[(0,0)].abs().max()
        state.sites[(0,0)]= state.build_onsite_tensors()
        state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')

        # 2. convert to 1-site YASTN's iPEPS
        state_yastn= PepsAD.from_pt(state)

        # 3. proceed with YASTN's C4v-CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            with torch.no_grad():
                env_leg = yastn.Leg(state_yastn.config, s=1, t=(0,), D=(1,))
                ctm_env_in = EnvCTM_c4v(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
                # 3.1.1 post-init CTM steps (allow expansion of the environment in case of qr policy)
                if ctm_args.projector_svd_method=='QR':
                    options_svd_pre_init= {
                        "policy": "arnoldi",
                        "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                        "eps_multiplet": ctm_args.projector_eps_multiplet,
                    }
                    ctm_env_in, converged, conv_history, t_ctm, t_check= ctmrg(ctm_env_in, lambda _0,_1: (False, None),
                        options_svd_pre_init,
                        max_sweeps= ctm_args.fpcm_init_iter,
                        method="default",
                        checkpoint_move=False
                    )
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.2 setup and run CTMRG
        options_svd={
            "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet,
            }
        ctm_env_out, env_ts_slices, env_ts, t_ctm = fp_ctmrg_c4v(ctm_env_in, \
            ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': ctm_args.ctm_conv_tol, 'max_sweeps': ctm_args.ctm_max_iter, \
                'method': "default", 'use_qr': False, 'svd_policy': YASTN_PROJ_METHOD[ctm_args.projector_svd_method], 'D_block': args.chi, \
                    "svds_thresh":ctm_args.fwd_svds_thresh, "svds_solver":ctm_args.fwd_svds_solver}, \
            ctm_opts_fp= {'svd_policy': 'fullrank'})
        refill_env_c4v(ctm_env_out, env_ts, env_ts_slices)
        print(f"t_ctm: {t_ctm:.1f}s")

        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_c4v_env_c4v(ctm_env_out)

        # 3.4 evaluate loss
        t_loss0= time.perf_counter()
        loss= energy_f(state, env_pt)
        t_loss1= time.perf_counter()
        return (loss, ctm_env_out, [], t_ctm, None, t_loss1-t_loss0)


    @torch.no_grad()
    def obs_fn_default(state, ctm_env, opt_context):
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

    @torch.no_grad()
    def obs_fn_c4v(state, ctm_env, opt_context):
        env_pt= from_yastn_c4v_env_c4v(ctm_env)

        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print(f"id {opt_context['id']}, LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
        obs_values, obs_labels = model.eval_obs(state,env_pt,force_cpu=args.force_cpu)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
            +[f"{state.coeffs[(0,0)].abs().max()}"]))

    # optimize
    if args.grad_type=='default':
        loss_fn= loss_fn_default
        obs_fn= obs_fn_default
    elif args.grad_type=='fp':
        loss_fn= loss_fn_fp
        obs_fn= obs_fn_default
    elif args.grad_type=='c4v':
        loss_fn= loss_c4v
        obs_fn= obs_fn_c4v
    elif args.grad_type=='c4v_fp':
        loss_fn= loss_c4v_fp
        obs_fn= obs_fn_c4v

    print("\n\n"+", ".join(["epoch","loss","avg_m"]))
    optimize_state(state, None, loss_fn, obs_fn=obs_fn)

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class Test_U1c4v_2site(unittest.TestCase):
    r"""
    Test case for optimization of U(1) x C4v symmetric ansatz via both explicit 2-site unit cell
    and implicit 2-site unit cell.
    """
    tol= 1.0e-6
    tol_2site= 1.0e-3 # explicit 2-site unit cell uses generic directional CTM.
                      # This mainly tests fatal error rather then precise observables.
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_opt_U1c4v_2site"

    def setUp(self):
        abs_dir = os.path.dirname(os.path.abspath(__file__))

        args.bond_dim=3
        args.seed=1
        args.opt_max_iter= 4
        args.instate_noise=0
        if args.bond_dim==3:
            args.instate = os.path.join(abs_dir, "../../../test-input/abelian/c4v/BFGS100LS_U1B_D3-chi72-j20.0-run0-iRNDseed321_state.json")
            self.expected_energy= -0.6646014335383251 # at chi=36
            args.chi=36
        elif args.bond_dim==4:
            args.instate = os.path.join(abs_dir, "../../../test-input/abelian/c4v/BFGS100LS_U1B_D4-chi97-j20.0-run0-iU1BD4j20chi97n0_state.json")
            self.expected_energy= -0.6689670982443985 # at chi=32
            args.chi=32

        # args.CTMARGS_ctm_env_init_type= "eye"
        # args.GLOBALARGS_dtype= "complex128"

    def _opt_U1c4v_2site(self):
        import builtins
        from unittest.mock import patch
        from io import StringIO
        local_tol= self.tol if 'c4v' in args.grad_type else self.tol_2site

        # i) run optimization
        tmp_out= StringIO()
        original_print = builtins.print
        def passthrough_print(*args, **kwargs):
            original_print(*args, **kwargs)
            kwargs.update(file=tmp_out)
            original_print(*args, **kwargs)

        with patch('builtins.print', new=passthrough_print) as tmp_print:
            main()

        # parse FINAL observables
        obs_opt_lines=[]
        final_obs=None
        OPT_OBS= OPT_OBS_DONE= False
        tmp_out.seek(0)
        l= tmp_out.readline()
        while l:
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="":
                OPT_OBS_DONE= True
                OPT_OBS=False
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>1:
                if not any(ch.isalpha() for ch in l[:1]): # skip lines with starting non-numeric values (i.e. not epoch)
                    obs_opt_lines.append(l)
            if "epoch, loss," in l and not OPT_OBS_DONE:
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert len(obs_opt_lines)>0

        # compare the line of observables with lowest energy from optimization (i)
        # TODO and final observables evaluated from best state stored in *_state.json output file
        best_e_line_index= np.argmin([ float(l.split(',')[1]) for l in obs_opt_lines ])
        opt_line_last= [float(x) for x in obs_opt_lines[best_e_line_index].split(",")]
        for val0,val1 in zip(opt_line_last[1:2], [self.expected_energy] ):
            assert np.isclose(val0,val1, rtol=local_tol, atol=local_tol)

    def test_opt(self):
        for grad_type in ['default','fp','c4v','c4v_fp']:
            for fwd_checkpoint_move in [True,False]:
                with self.subTest(grad_type=grad_type, fwd_checkpoint_move=fwd_checkpoint_move):
                    args.grad_type= grad_type
                    args.CTMARGS_fwd_checkpoint_move= fwd_checkpoint_move
                    args.out_prefix= self.OUT_PRFX
                    self._opt_U1c4v_2site()

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        out_prefix=self.OUT_PRFX
        for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",out_prefix+".log"]:
            if os.path.isfile(f): os.remove(f)