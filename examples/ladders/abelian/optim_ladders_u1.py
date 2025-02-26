import os
import copy
import context
import argparse
import torch
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import coupledLadders
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
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
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

    model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha,\
        Bz_val=args.bz_stag)

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
        model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha,\
            Bz_val=args.bz_stag)

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x1_1x2(state, env).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history
    @torch.no_grad()
    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        return ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)

    # alternatively use ctmrg_conv_specC from ctm.generinc.env
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss= model.energy_2x1_1x2(state, ctm_env).to_number()
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build double-layer open on-site tensors
        state.sync_precomputed()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss= model.energy_2x1_1x2(state, ctm_env_out)
        loss= loss.to_number()

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS "+", ".join([f"{epoch}",f"{loss}"]))
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
            obs_values, obs_labels = model.eval_obs(state,ctm_env)

            # test ENV sensitivity
            if args.test_env_sensitivity:
                loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
                loc_ctm_args.ctm_max_iter= 1
                ctm_env_out1= ctm_env.clone()
                ctm_env_out1.chi= ctm_env.chi+10
                ctm_env_out1, *ctm_log= ctmrg.run(state, ctm_env_out1, \
                    conv_check=ctmrg_conv_f, ctm_args=loc_ctm_args)
                loss1= model.energy_2x1_1x2(state, ctm_env_out1).to_number()
                delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                    if len(opt_context['loss_history']['loss'])>1 else float('NaN')
                # if we are not linesearching, this can always happen
                # not "line_search" in opt_context.keys()
                _flag_antivar= (loss1-loss)>0 and (loss1-loss)>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
                + ([f"{loss1-loss}"] if args.test_env_sensitivity else []) ))
            log.info(f"env_sensitivity: {loss1-loss} loss_diff: "\
                +f"{delta_loss}" if args.test_env_sensitivity else ""\
                +" Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

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
    if args.test_env_sensitivity:
        state_g= IPEPS_ABELIAN_WEIGHTED(state=state).gauge()
        state= state_g.absorb_weights()
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, settings)
    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = model.energy_2x1_1x2(state,ctm_env).to_number()
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCheckpoint_VBSstate(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-chck_u1_vbs"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/ABU1_BFGS100LS_D2-chi24-a0.1-run0-svd8_i2SUVBSn0_state.json"
        args.alpha=0.1
        args.bond_dim=2
        args.chi=16
        args.instate_noise=0.1
        args.seed=300
        args.out_prefix=self.OUT_PRFX

    def test_checkpoint_noisy_vbs(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        # i) run optimization and store the optimization data
        args.opt_max_iter= 10
        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        obs_opt_lines=[]
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines.append(l)
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            l= tmp_out.readline()
        assert len(obs_opt_lines)>0

        # ii) run optimization for 3 steps
        args.opt_max_iter= 3
        main()
        
        # iii) run optimization from checkpoint
        args.instate=None
        args.opt_resume= self.OUT_PRFX+"_checkpoint.p"
        args.opt_max_iter= 7
        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        obs_opt_lines_chk=[]
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines_chk.append(l)
            if "checkpoint.loss" in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            l= tmp_out.readline()
        assert len(obs_opt_lines_chk)>0

        # compare initial observables from checkpointed optimization (iii) and the observables 
        # from original optimization (i) at one step after total number of steps done in (ii)
        opt_line_iii= [float(x) for x in obs_opt_lines_chk[0].split(",")[1:]]
        opt_line_i= [float(x) for x in obs_opt_lines[4].split(",")[1:]]
        for val3,val1 in zip(opt_line_iii, opt_line_i):
            assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

        # compare final observables from optimization (i) and the final observables 
        # from the checkpointed optimization (iii)
        fobs_tokens_1= [float(x) for x in obs_opt_lines[-1].split(",")[1:]]
        fobs_tokens_3= [float(x) for x in obs_opt_lines_chk[-1].split(",")[1:]]
        for val3,val1 in zip(fobs_tokens_3, fobs_tokens_1):
            assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)