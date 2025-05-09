import os
import context
import argparse
import yastn.yastn as yastn
import config as cfg
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from ctm.generic_abelian import transferops
from models.abelian import j1j2
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
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument('--top_t', nargs="+", type=int, default=[-2,0,2], help="TM charge sectors")
parser.add_argument("--yast_backend", type=str, default='np', 
    help="YAST backend", choices=['np','torch','torch_cpp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='np':
        from yastn.yastn.backend import backend_np as backend
    elif args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings_full= yastn.make_config(backend=backend, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    if settings.backend.BACKEND_ID == 'torch':
        import torch
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
            +"BIPARTITE, 1SITE, 2SITE, 4SITE, 8SITE")

    if args.instate!=None:
        state= read_ipeps(args.instate, settings, vertexToSite=lattice_to_site)
        state= state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=2, lY=2)
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

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env).to_number()
        obs_values, obs_labels = model.eval_obs(state, env)
        history.append([e_curr]+obs_values)
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    
    # 3) evaluate observables for initial environment
    loss0 = energy_f(state, ctm_env).to_number()
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    # 5) compute final observables and timings
    e_curr0 = energy_f(state, ctm_env).to_number()
    obs_values0, obs_labels = model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    # for c_loc,c_ten in ctm_env.C.items(): 
    #     u,s,v= c_ten.svd(([0],[1]))
    #     print(f"\n\nspectrum C[{c_loc}]")
    #     for charges in s.get_blocks_charge():
    #         print(charges)
    #         sector= s[charges]
    #         for i in range(len(sector)):
    #             print(f"{i} {sector[i]}")

    # ----- S(0).S(r) -----
    site_dir_list=[((0,0), (1,0)), ((0,0), (0,1)), ((1,0), (1,0)), ((1,0), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env, args.corrf_r, rl_0=None)
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # transfer operator spectrum 1-site-width channel
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}] {tuple(args.top_t)}")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env, edge_t=tuple(args.top_t))
        for i in range(l.shape[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run_u1_d3_2x1_neel"
    BACKENDS = ['np', 'torch']

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/c4v/BFGS100LS_U1B_D3-chi72-j20.0-run0-iRNDseed321_blocks_2site_state.json"
        args.bond_dim=3
        args.chi=32
        args.out_prefix=self.OUT_PRFX

    def test_ctmrg_j1j2_bipartite(self):
        from io import StringIO 
        from unittest.mock import patch 
        from math import isclose

        for b_id in self.BACKENDS:
            with self.subTest(b_id=b_id):
                args.yast_backend=b_id
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                final_obs=None
                l= tmp_out.readline()
                while l:
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs

                # compare with the reference
                ref_data="""
                -0.6645979511667757, 0.3713621967866411, 0.3713621967866413, 0.37136219678664095, 
                0.3713621967866413, 0.0, 0.0, -0.37136219678664095, 0.0, 0.0, -0.33229727696449596, 
                -0.33229727696449607, -0.3322972769393827, -0.33229727693938854
                """
                fobs_tokens= [float(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [float(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    assert isclose(val,ref_val, rel_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json"]:
            if os.path.isfile(f): os.remove(f)