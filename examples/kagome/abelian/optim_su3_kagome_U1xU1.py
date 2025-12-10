import os
import context
import torch
import numpy as np
import argparse
import config as cfg
import yastn.yastn as yastn
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
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    # Parametrization
    param_j = np.round(np.cos(np.pi*args.phi), decimals=15)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=15)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=15)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    from yastn.yastn.sym import sym_U1xU1
    if args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings= yastn.make_config(backend=backend, sym=sym_U1xU1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    # the model (in particular operators forming Hamiltonian) is defined in a dense form
    # with no symmetry structure
    model= su3_kagome.KAGOME_SU3_U1xU1(settings,j=param_j,k=param_k,h=param_h,global_args=cfg.global_args)

    # initialize an ipeps
    if args.instate!=None:
        state= read_ipess_kagome_generic(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        T_u= yastn.Tensor(config=settings, s=(-1,-1,-1))
        T_d= yastn.Tensor(config=settings, s=(-1,-1,-1))
        B_c= yastn.Tensor(config=settings, s=(-1,1,1))
        B_a= yastn.Tensor(config=settings, s=(-1,1,1))
        B_b= yastn.Tensor(config=settings, s=(-1,1,1))
        state= IPESS_KAGOME_GENERIC_ABELIAN(settings, {'T_u': T_u, 'B_a': B_a,\
            'T_d': T_d, 'B_b': B_b, 'B_c': B_c}, build_sites=False)
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

    @torch.no_grad()
    def ctmrg_conv_specC(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history={'spec': [], 'diffs': []}
        # use corner spectra
        diff=float('inf')
        diffs=None
        spec= env.get_spectra()
        spec_nosym_sorted= { s_key : s_t._data.sort(descending=True)[0] \
            for s_key, s_t in spec.items() }
        if len(history['spec'])>0:
            s_old= history['spec'][-1]
            diffs= []
            for k in spec.keys():
                x_0,x_1 = spec_nosym_sorted[k], s_old[k]
                if x_0.size(0)>x_1.size(0):
                    diffs.append( (sum((x_1-x_0[:x_1.size(0)])**2) \
                        + sum(x_0[x_1.size(0):]**2)).item() )
                else:
                    diffs.append( (sum((x_0-x_1[:x_0.size(0)])**2) \
                        + sum(x_1[x_0.size(0):]**2)).item() )
            diff= sum(diffs)
        history['spec'].append(spec_nosym_sorted)
        history['diffs'].append(diffs)

        if (len(history['diffs']) > 1 and abs(diff) < ctm_args.ctm_conv_tol)\
            or len(history['diffs']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['diffs']), "history": history['diffs']})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
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

        # 1) re-build precomputed double-layer on-site tensors
        #    Some objects, in this case open-double layer tensors, are pre-computed
        state.sync_precomputed()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        
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
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOptim_TrimerState(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt_u1xu1_trimerized"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/IPESS_TRIMER_1-3_1x1_abelian-U1xU1_T3T8_state.json"
        args.symmetry="U1xU1"
        args.theta=0
        args.phi=0
        args.bond_dim=4
        args.chi=16
        args.instate_noise=0.1
        args.seed=212
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype= "complex128"

    def test_basic_opt_noisy_trimer(self):
        from io import StringIO
        from unittest.mock import patch 
        from cmath import isclose

        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        final_obs=None
        final_opt_line=None
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                final_opt_line= l
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs
        assert final_opt_line

        # compare with the reference
        ref_data="""
        -0.6666666666666664, 0j, 0j, 0j, 0.0, 0.0, 0.3333333333333333, 0.3333333333333333, 
        0.3333333333333333, -0.9999999999999999, -0.9999999999999999, -0.9999999999999999
        """
        # compare final observables from optimization and the observables from the 
        # final state
        final_opt_line_t= [complex(x) for x in final_opt_line.split(",")[1:]]
        fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
        for val0,val1 in zip(final_opt_line_t, fobs_tokens):
            assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

        # compare final observables from final state against expected reference 
        # drop first token, corresponding to iteration step
        ref_tokens= [complex(x) for x in ref_data.split(",")]
        for val,ref_val in zip(fobs_tokens, ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)

class TestCheckpoint_TrimerState(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-chck_u1xu1_trimerized"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/IPESS_TRIMER_1-3_1x1_abelian-U1xU1_T3T8_state.json"
        args.symmetry="U1xU1"
        args.theta=0
        args.phi=0
        args.bond_dim=4
        args.chi=16
        args.instate_noise=0.1
        args.seed=300
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype= "complex128"

    def test_checkpoint_noisy_trimer(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        # i) run optimization and store the optimization data
        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        obs_opt_lines=[]
        final_obs=None
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines.append(l)
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs
        assert len(obs_opt_lines)>0

        # ii) run optimization for 3 steps
        args.opt_max_iter= 3 
        main()
        
        # iii) run optimization from checkpoint
        args.instate=None
        args.opt_resume= self.OUT_PRFX+"_checkpoint.p"
        args.opt_max_iter= 100
        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        obs_opt_lines_chk=[]
        final_obs_chk=None
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines_chk.append(l)
            if "checkpoint.loss" in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:    
                final_obs_chk= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs_chk
        assert len(obs_opt_lines_chk)>0

        # compare initial observables from checkpointed optimization (iii) and the observables 
        # from original optimization (i) at one step after total number of steps done in (ii)
        opt_line_iii= [complex(x) for x in obs_opt_lines_chk[0].split(",")[1:]]
        opt_line_i= [complex(x) for x in obs_opt_lines[4].split(",")[1:]]
        fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
        for val3,val1 in zip(opt_line_iii, opt_line_i):
            assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

        # compare final observables from optimization (i) and the final observables 
        # from the checkpointed optimization (iii)
        fobs_tokens_1= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
        fobs_tokens_3= [complex(x) for x in final_obs_chk[len("FINAL"):].split(",")]
        for val3,val1 in zip(fobs_tokens_3, fobs_tokens_1):
            assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

        # compare final observables from the checkpointed optimization (iii) with the reference 
        ref_data="""
        -0.6666666666666664, 0j, 0j, 0j, 0.0, 0.0, 0.3333333333333333, 0.3333333333333333, 
        0.3333333333333333, -0.9999999999999999, -0.9999999999999999, -0.9999999999999999
        """
        # compare final observables from final state of checkpointed optimization 
        # against expected reference. Drop first token, corresponding to iteration step
        ref_tokens= [complex(x) for x in ref_data.split(",")]
        fobs_tokens_3= [complex(x) for x in final_obs_chk[len("FINAL"):].split(",")]
        for val,ref_val in zip(fobs_tokens_3, ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)