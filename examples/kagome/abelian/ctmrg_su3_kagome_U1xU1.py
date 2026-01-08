import os
import context
import numpy as np
import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1xU1
import argparse
import config as cfg
from ipeps.ipess_kagome_abelian import read_ipess_kagome_generic
from linalg.custom_svd import truncated_svd_gesdd
from models.abelian import su3_kagome
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
# from ctm.generic import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--phi", type=float, default=0.5, help="arctan(K/J): J -> 2-site coupling; K -> 3-site coupling")
parser.add_argument("--theta", type=float, default=0., help="arctan(H/K): K -> 3-site coupling; K -> chiral coupling")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
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
    settings= yastn.make_config(backend=backend, sym=sym_U1xU1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    if settings.backend.BACKEND_ID == 'torch':
        import torch
        torch.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    param_j = np.round(np.cos(np.pi*args.phi), decimals=15)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=15)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=15)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
   
    model= su3_kagome.KAGOME_SU3_U1xU1(settings,j=param_j,k=param_k,h=param_h,global_args=cfg.global_args)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.instate!=None:
        state= read_ipess_kagome_generic(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) define convergence criterion for ctmrg
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        # simply use single down triangle energy to evaluate the CTMRG convergence
        e_curr = model.energy_down_t_1x1subsystem(state, env)
        history.append(e_curr.item())
        obs_values, obs_labels = model.eval_obs(state, env)
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            return True, history
        return False, history

    def ctmrg_conv_specC(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history={'spec': [], 'diffs': []}
        # use corner spectra
        diff=float('inf')
        diffs=None
        spec= env.get_spectra()
        if args.yast_backend=='np':
            spec_nosym_sorted= { s_key : np.sort(s_t._data)[::-1] \
                for s_key, s_t in spec.items() }            
        else:
            spec_nosym_sorted= { s_key : s_t._data.sort(descending=True)[0] \
                for s_key, s_t in spec.items() }
        if len(history['spec'])>0:
            s_old= history['spec'][-1]
            diffs= []
            for k in spec.keys():
                x_0,x_1 = spec_nosym_sorted[k], s_old[k]
                n_x0= x_0.shape[0] if args.yast_backend=='np' else x_0.size(0)
                n_x1= x_1.shape[0] if args.yast_backend=='np' else x_1.size(0)
                if n_x0>n_x1:
                    diffs.append( (sum((x_1-x_0[:n_x1])**2) \
                        + sum(x_0[n_x1:]**2)).item() )
                else:
                    diffs.append( (sum((x_0-x_1[:n_x0])**2) \
                        + sum(x_1[n_x0:]**2)).item() )
            diff= sum(diffs)
        history['spec'].append(spec_nosym_sorted)
        history['diffs'].append(diffs)
        obs_values, obs_labels = model.eval_obs(state, env)
        print(", ".join([f"{len(history['diffs'])}",f"{diff}"]+[f"{v}" for v in obs_values]))

        if (len(history['diffs']) > 1 and abs(diff) < ctm_args.ctm_conv_tol)\
            or len(history['diffs']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['diffs']), "history": history['diffs']})
            return True, history
        return False, history

    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    print(ctm_env)

    # 3) evaluate observables for initial environment
    loss= model.energy_per_site_2x2subsystem(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","conv-crit"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_specC)

    # 5) compute final observables and timings
    loss= model.energy_per_site_2x2subsystem(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # # environment diagnostics
    # for c_loc,c_ten in ctm_env.C.items():
    #     u,s,v= truncated_svd_gesdd(c_ten, c_ten.size(0))
    #     print(f"\n\nspectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")

    # # transfer operator spectrum
    # site_dir_list=[((0,0), (1,0)),((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
    # for sdp in site_dir_list:
    #     print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
    #     l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env)
    #     for i in range(l.size()[0]):
    #         print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrg_TrimerState(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run_u1xu1_trimerized"
    BACKENDS = ['np', 'torch']

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/IPESS_TRIMER_1-3_1x1_abelian-U1xU1_T3T8_state.json"
        args.theta=0
        args.phi=0
        args.bond_dim=4
        args.chi=16
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_trimer(self):
        from io import StringIO
        from unittest.mock import patch 
        from cmath import isclose

        for b_id in self.BACKENDS:
            with self.subTest(b_id=b_id):
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                final_obs=None
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs

                # compare with the reference
                ref_data="""
                -0.6666666666666664, 0j, 0j, 0j, 0.0, 0.0, 0.3333333333333333, 0.3333333333333333, 
                0.3333333333333333, -0.9999999999999999, -0.9999999999999999, -0.9999999999999999
                """
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [complex(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)

class TestCtmrg_AKLTState(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run_u1xu1_aklt"
    BACKENDS = ['np', 'torch']

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/IPESS_AKLT_3b_D3_1x1_abelian-U1xU1_T3T8_state.json"
        args.theta=0
        args.phi=0.5
        args.bond_dim=3
        args.chi=18
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_aklt(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        for b_id in self.BACKENDS:
            with self.subTest(b_id=b_id):
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                final_obs=None
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs

                # compare with the reference
                ref_data="""
                -0.6666666666666664, 0j, 0j, 0j, 0.0, 0.0, 0., 0., 0., 0., 0., 0.
                """
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [complex(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)