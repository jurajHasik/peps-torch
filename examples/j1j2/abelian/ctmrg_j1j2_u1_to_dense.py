import os
import context
import torch
import argparse
import config as cfg
import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1
from yastn.yastn.backend import backend_torch as backend
from ipeps.ipeps_abelian import read_ipeps
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import j1j2
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["BIPARTITE", "1SITE", "2SITE", "4SITE", "8SITE"])
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = j1j2.J1J2(j1=args.j1, j2=args.j2)

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
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    # initialize an ipeps
    if args.instate!=None:
        state_ab= read_ipeps(args.instate, settings, vertexToSite=lattice_to_site)
        state_ab= state_ab.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state_ab)
    state= state_ab.to_dense()
    print(state)

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= j1j2.J1J2(j1=args.j1, j2=args.j2)

    # 2) select the "energy" function 
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f= model.energy_2x2_2site
        eval_obs_f= model.eval_obs
    elif args.tiling == "1SITE":
        energy_f= model.energy_2x2_1site_BP
        eval_obs_f= model.eval_obs_1site_BP
    elif args.tiling == "4SITE":
        energy_f= model.energy_2x2_4site
        eval_obs_f= model.eval_obs
    elif args.tiling == "8SITE":
        energy_f= model.energy_2x2_8site
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            e_curr = energy_f(state, env)
            obs_values, obs_labels = eval_obs_f(state, env)
            history.append([e_curr.item()]+obs_values)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # 7) ----- additional observables ---------------------------------------------
    # TODO correct corrf_SS for 1site ansatz - include rotation on B-sublattice
    corrSS= model.eval_corrf_SS((0,0), (1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(1,0)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS= model.eval_corrf_SS((0,0), (0,1), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(0,1)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)), ((0,0), (0,1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
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

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/c4v/BFGS100LS_U1B_D3-chi72-j20.0-run0-iRNDseed321_blocks_2site_state.json"
        args.bond_dim=3
        args.chi=32
        args.out_prefix=self.OUT_PRFX

    def test_ctmrg_j1j2_bipartite(self):
        from io import StringIO 
        from unittest.mock import patch 
        from math import isclose

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
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json"]:
            if os.path.isfile(f): os.remove(f)