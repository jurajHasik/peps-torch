import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import j1j2
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim_sgd_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--j3", type=float, default=0, help="next-to-next nearest-neighbour coupling")
parser.add_argument("--lmbd", type=float, default=0, help="chiral plaquette interaction")
parser.add_argument("--hz_stag", type=float, default=0, help="staggered mag. field")
parser.add_argument("--h_uni", nargs=3, type=float, default=[0,0,0], help="uniform mag. field with components in directions h^z, h^x, h^y")
parser.add_argument("--delta_zz", type=float, default=1, help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["BIPARTITE", "1SITE", "2SITE", "4SITE", "8SITE"])
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model= j1j2.J1J2(j1=args.j1, j2=args.j2, j3=args.j3, lmbd=args.lmbd,
        hz_stag=args.hz_stag, h_uni=args.h_uni, delta_zz=args.delta_zz)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "1SITE":
        def lattice_to_site(coord):
            return (0, 0)
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
            +"BIPARTITE, 1SITE, 2SITE, 4SITE, 8SITE")

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state= IPEPS(dict(), lX=2, lY=1)
        elif args.tiling == "1SITE":
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "4SITE":
            state= IPEPS(dict(), lX=2, lY=2)
        elif args.tiling == "8SITE":
            state= IPEPS(dict(), lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A = torch.zeros([model.phys_dim]+ [bond_dim]*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

        sites = {(0,0): A}
        if args.tiling in ["BIPARTITE", "2SITE", "4SITE", "8SITE"]:
            sites[(1,0)] = torch.zeros_like(A)
        if args.tiling in ["4SITE", "8SITE"]:
            sites[(0,1)]= torch.zeros_like(A)
            sites[(1,1)]= torch.zeros_like(A)
        if args.tiling == "8SITE":
            sites[(2,0)]= torch.zeros_like(A)
            sites[(3,0)]= torch.zeros_like(A)
            sites[(2,1)]= torch.zeros_like(A)
            sites[(3,1)]= torch.zeros_like(A)
        state = IPEPS(sites, vertexToSite=lattice_to_site)
        state.add_noise(noise=1.0)
        state.normalize_()
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= j1j2.J1J2(j1=args.j1, j2=args.j2, j3=args.j3, lmbd=args.lmbd,
            hz_stag=args.hz_stag, h_uni=args.h_uni, delta_zz=args.delta_zz)

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f=model.energy_2x2_2site
        eval_obs_f= model.eval_obs
    elif args.tiling == "1SITE":
        energy_f= model.energy_2x2_1site_BP
        # TODO include eval_obs with rotation on B-sublattice
        eval_obs_f= model.eval_obs_1site_BP
    elif args.tiling == "4SITE":
        energy_f=model.energy_2x2_4site
        eval_obs_f= model.eval_obs
    elif args.tiling == "8SITE":
        energy_f=model.energy_2x2_8site
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state, ctm_env)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
             conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        ctm_env_out= ctm_env_in

        # 2) evaluate loss with the converged environment
        loss = energy_f(state, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = eval_obs_f(state,ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

            with torch.no_grad():
                if args.top_freq>0 and epoch%args.top_freq==0:
                    coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
                    for c,d in coord_dir_pairs:
                        # transfer operator spectrum
                        print(f"TOP spectrum(T)[{c},{d}] ",end="")
                        l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
                        print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state,ctm_env)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{loss0}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

import os
try:
    import pytest
except:
    pytest=False
    warnings.warn("pytest not available.")

class TestOptBasic(unittest.TestCase):
    def setUp(self):
        args.j2=1.
        args.j3=1.
        args.hz_stag=1.
        args.delta_zz=1.
        args.bond_dim=2
        args.chi=8
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. Arnoldi methods not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_BIPARTITE_LS_strong_wolfe(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="strong_wolfe"
        main()

    def test_opt_GESDD_BIPARTITE_LS_backtracking(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    def test_opt_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_LS_backtracking_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

class TestOpt_1SITE_uniformh(unittest.TestCase):
    tol= 1.0e-6

    def setUp(self):
        args.bond_dim=3
        args.chi=18
        args.seed=123
        args.CTMARGS_ctm_conv_tol= 1.0e-6
        args.GLOBALARGS_dtype= "complex128"

    @pytest.mark.slow
    def test_opt_1site_uniformh(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        args.j3=0.125
        args.h_uni=[0,0,3.9]
        args.tiling= "1SITE"
        args.out_prefix= "TEST_J1J2J3H_1SITE"
        args.opt_max_iter= 40

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
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                OPT_OBS_DONE= True
                OPT_OBS=False
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines.append(l)
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert len(obs_opt_lines)>0

        # compare the line of observables with lowest energy from optimization (i) 
        # and final observables evaluated from best state stored in *_state.json output file
        # drop the last column, not separated by comma
        ref_obs="""-1.389482685835177, 0.49274486715126437, 0.49274486715126437, (-0.007350694172083588+0j), 
        (0.1663566417095179+0.46375525782274074j), (0.1663566417095179-0.46375525782274074j), 0.178628242537441, 
        0.17854780045652874"""

        opt_line_last= [complex(x) for x in obs_opt_lines[-1].split(",")[1:-1]]
        ref_tokens= [complex(x) for x in ref_obs.split(",")]
        # compare energy per site, avg magnetization
        for val0,val1 in zip(opt_line_last[:2], ref_tokens[:2]):
            assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        out_prefix=args.out_prefix
        for suffix in ["_checkpoint.p","_state.json",".log"]:
            f= out_prefix+suffix
            if os.path.isfile(f): os.remove(f)

class TestOpt4SITE(unittest.TestCase):
    tol= 1.0e-6

    def setUp(self):
        args.bond_dim=2
        args.chi=8
        args.seed=123
        args.CTMARGS_ctm_conv_tol= 1.0e-6

    @pytest.mark.slow
    def test_opt_4site(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        args.j2=0.3
        args.tiling= "4SITE"
        args.out_prefix= "TEST_J1J2_4SITE"
        args.opt_max_iter= 40

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
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                OPT_OBS_DONE= True
                OPT_OBS=False
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                obs_opt_lines.append(l)
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert len(obs_opt_lines)>0

        # compare the line of observables with lowest energy from optimization (i) 
        # and final observables evaluated from best state stored in *_state.json output file
        # drop the last column, not separated by comma
        ref_obs="""-0.5430086212529559, 0.3439627621221997, 0.3440126387668104, 0.3437736797969528, 
        0.3441560859026258, 0.3439086440224097, -0.3404606733491301, -0.04930745921218108, 
        -0.04930745921218108, 0.3402337397327219, 0.04920716684207956, 0.04920716684207956, 
        0.3406737771551228, 0.04883430170154231, 0.04883430170154231, -0.3404093367912687, 
        -0.04893504734503799, -0.04893504734503799, -0.3275362627220919, -0.32795775261024607, 
        -0.3273537925317087, -0.3280033934375568, -0.327373572784679, -0.3277528502254806, 
        -0.3277228937834971, -0.32759323792859424"""

        opt_line_last= [float(x) for x in obs_opt_lines[-1].split(",")[1:-1]]
        ref_tokens= [float(x) for x in ref_obs.split(",")]
        # compare energy per site, avg magnetization
        for val0,val1 in zip(opt_line_last[:2], ref_tokens[:2]):
            assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        out_prefix=args.out_prefix
        for suffix in ["_checkpoint.p","_state.json",".log"]:
            f= out_prefix+suffix
            if os.path.isfile(f): os.remove(f)