import os
import context
import torch
import argparse
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import j1j2
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
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
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
    
    # the model (in particular operators forming Hamiltonian) is defined in a dense form
    # with no symmetry structure
    model= j1j2.J1J2_NOSYM(settings_full,j1=args.j1,j2=args.j2)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    elif args.tiling == "8SITE":
        def lattice_to_site(coord):
            shift_x = coord[0] + 2*(coord[1] // 2)
            vx = shift_x % 4
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    if args.instate!=None:
        state= read_ipeps(args.instate, settings, vertexToSite=lattice_to_site)
        state= state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=2, lY=1)
        elif args.tiling == "4SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=2, lY=2)
        elif args.tiling == "8SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= j1j2.J1J2_NOSYM(settings_full,j1=args.j1,j2=args.j2)

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling=="BIPARTITE" or args.tiling=="2SITE" or args.tiling=="4SITE" \
        or args.tiling=="8SITE":
        energy_f=model.energy_2x1_or_2Lx2site_2x2rdms
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(state, ctm_env).to_number()
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # recompute double-layer closed/open on-site tensors
        state.sync_precomputed()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        
        # 2) evaluate loss with the converged environment
        loss= energy_f(state, ctm_env_out)
        loss= loss.to_number()
        
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
    state= read_ipeps(outputstatefile, settings, vertexToSite=state.vertexToSite)
    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env).to_number()
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCheckpoint_NeelBipartiteState(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-chck_u1_AB"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/c4v/BFGS100LS_U1B_D3-chi72-j20.0-run0-iRNDseed321_blocks_2site_state.json"
        args.j2=0.5
        args.chi=24
        args.tiling= "BIPARTITE"
        args.out_prefix=self.OUT_PRFX

    def test_checkpoint_neel_bipartite(self):
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
