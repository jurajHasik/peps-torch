import context
import torch
from linalg.custom_svd import truncated_svd_gesdd
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import coupledLadders
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--alpha", type=float, default=0., help="inter-ladder coupling")
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= coupledLadders.COUPLEDLADDERS(alpha=args.alpha)

    if args.instate!=None:
        state = read_ipeps(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

        sites = {(0,0): A, (1,0): B, (0,1): C, (1,1): D}

        for k in sites.keys():
            sites[k] = sites[k]/torch.max(torch.abs(sites[k]))
        state = IPEPS(sites, lX=2, lY=2)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")
    
    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= coupledLadders.COUPLEDLADDERS(alpha=args.alpha)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    

    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            e_curr = model.energy_2x1_1x2(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()]+obs_values)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # ----- S(0).S(r) -----
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r)
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ----- (S(0).S(x))(S(rx).S(rx+x)) -----
    for sdp in site_dir_list:
        corrDD= model.eval_corrf_DD_H(*sdp, state, ctm_env_init, args.corrf_r)
        print(f"\n\nDD[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrDD.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    # ----- (S(0).S(y))(S(rx).S(rx+y)) -----
    for sdp in site_dir_list:
        corrDD_V= model.eval_corrf_DD_V(*sdp,state, ctm_env_init, args.corrf_r)
        print(f"\n\nDD_V[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrDD_V.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrDD_V[label][i]}" for label in corrDD_V.keys()]))

    # environment diagnostics
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= truncated_svd_gesdd(c_ten, c_ten.size(0))
        print(f"\n\nspectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    # transfer operator spectrum
    # for sdp in site_dir_list:
    #     print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
    #     l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
    #     for i in range(l.size()[0]):
    #         print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrg(unittest.TestCase):
    def setUp(self):
        args.instate=None
        args.alpha=1.0
        args.bond_dim=2
        args.chi=16

    # basic tests
    def test_ctmrg_GESDD(self):
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_GESDD_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        main()

class TestLadders_VBS2x2(unittest.TestCase):
    def setUp(self):
        import os
        args.instate=os.path.dirname(os.path.realpath(__file__))+"/../../test-input/VBS_2x2_ABCD.in"
        args.chi=16
        args.opt_max_iter=50

    def test_ctmrg_Ladders_VBS2x2(self):
        cfg.configure(args)
        cfg.print_config()
        torch.set_num_threads(args.omp_cores)
        
        model = coupledLadders.COUPLEDLADDERS(alpha=args.alpha)
        
        state = read_ipeps(args.instate)

        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=[]
                e_curr = model.energy_2x1_1x2(state, env)
                history.append([e_curr.item()])

                if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                    return True, history
            return False, history

        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

        e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps=1.0e-12
        self.assertTrue(abs(e_curr0-(-0.375)) < eps)
        for coord,site in state.sites.items():
            self.assertTrue(obs_dict[f"m{coord}"] < eps, msg=f"m{coord}")
            self.assertTrue(obs_dict[f"SS2x1{coord}"] < eps, msg=f"SS2x1{coord}")
            for l in ["sz","sp","sm"]:
                self.assertTrue(abs(obs_dict[f"{l}{coord}"]) < eps, msg=f"{l}{coord}")
        for coord in [(0,0),(1,0)]:
            self.assertTrue(abs(obs_dict[f"SS1x2{coord}"]-(-0.75)) < eps, msg=f"SS1x2{coord}")

class TestLadders_VBS1x2(unittest.TestCase):
    def setUp(self):
        import os
        args.instate=os.path.dirname(os.path.realpath(__file__))+"/../../test-input/VBS_1x2_AB_D2.in"
        args.chi=16
        args.opt_max_iter=50

    def test_ctmrg_Ladders_VBS1x2(self):
        cfg.configure(args)
        cfg.print_config()
        torch.set_num_threads(args.omp_cores)
        
        model = coupledLadders.COUPLEDLADDERS_D2_BIPARTITE(alpha=args.alpha)
        
        state = read_ipeps(args.instate)

        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=[]
                e_curr = model.energy_2x1_1x2(state, env)
                history.append([e_curr.item()])

                if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                    return True, history
            return False, history

        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

        e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps=1.0e-12
        self.assertTrue(abs(e_curr0-(-0.375)) < eps)
        for coord,site in state.sites.items():
            self.assertTrue(obs_dict[f"m{coord}"] < eps, msg=f"m{coord}")
            self.assertTrue(obs_dict[f"SS2x1{coord}"] < eps, msg=f"SS2x1{coord}")
            for l in ["sz","sp","sm"]:
                self.assertTrue(abs(obs_dict[f"{l}{coord}"]) < eps, msg=f"{l}{coord}")
        for coord in [(0,0)]:
            self.assertTrue(abs(obs_dict[f"SS1x2{coord}"]-(-0.75)) < eps, msg=f"SS1x2{coord}")