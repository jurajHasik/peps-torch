import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from models import akltS2
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = akltS2.AKLTS2()
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
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

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        B = B/torch.max(torch.abs(B))

        sites = {(0,0): A, (1,0): B}
        
        if args.tiling == "4SITE":
            C= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            D= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            sites[(0,1)]= C/torch.max(torch.abs(C))
            sites[(1,1)]= D/torch.max(torch.abs(D))

        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=dict({"log": []})    
            dist= float('inf')
            list_rdm=[]
            for coord,site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                list_rdm.extend([rdm2x1,rdm1x2])

            # compute observables
            e_curr = model.energy_2x1_1x2(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            print(", ".join([f"{len(history['log'])}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history["log"]) > 1:
                dist=0.
                for i in range(len(list_rdm)):
                    dist+= torch.dist(list_rdm[i], history["rdm"][i], p=2).item()
            history["rdm"]=list_rdm
            history["log"].append(dist)
            if dist<ctm_args.ctm_conv_tol:
                log.info({"history_length": len(history['log']), "history": history['log']})
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    # environment diagnostics
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"\n\nspectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrg(unittest.TestCase):
    def setUp(self):
        args.bond_dim=2
        args.chi=16
        args.instate=None

    # basic tests
    def test_ctmrg_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_ctmrg_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

class TestAKLT_BIPARTITE(unittest.TestCase):
    def setUp(self):
        import os
        args.instate=os.path.dirname(os.path.realpath(__file__))+"/../../test-input/AKLT-S2_2x1_biLat.in"
        args.chi=32
        args.opt_max_iter=50

    def test_ctmrg_AKLT_BIPARTITE(self):
        cfg.configure(args)
        torch.set_num_threads(args.omp_cores)
        
        model = akltS2.AKLTS2()
        
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)

        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=dict({"log": []})    
                dist= float('inf')
                list_rdm=[]
                for coord,site in state.sites.items():
                    rdm2x1 = rdm.rdm2x1(coord,state,env)
                    rdm1x2 = rdm.rdm1x2(coord,state,env)
                    list_rdm.extend([rdm2x1,rdm1x2])

                if len(history["log"]) > 1:
                    dist=0.
                    for i in range(len(list_rdm)):
                        dist+= torch.dist(list_rdm[i], history["rdm"][i], p=2).item()
                history["rdm"]=list_rdm
                history["log"].append(dist)
                if dist<ctm_args.ctm_conv_tol:
                    log.info({"history_length": len(history['log']), "history": history['log']})
                    return True, history
            return False, history

        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

        e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps=1.0e-12
        self.assertTrue(e_curr0 < eps)
        for coord,site in state.sites.items():
            self.assertTrue(obs_dict[f"m{coord}"] < eps, msg=f"m{coord}")
            for l in ["sz","sp","sm"]:
                self.assertTrue(abs(obs_dict[f"{l}{coord}"]) < eps, msg=f"{l}{coord}")

class TestAKLT_4SITE(unittest.TestCase):
    def setUp(self):
        import os
        args.instate=os.path.dirname(os.path.realpath(__file__))+"/../../test-input/AKLT-S2_2x2_ABCD.in"
        args.chi=32
        args.opt_max_iter=50

    def test_ctmrg_AKLT_4SITE(self):
        cfg.configure(args)
        torch.set_num_threads(args.omp_cores)
        
        model = akltS2.AKLTS2()
        
        def lattice_to_site(coord):
            vx= (coord[0] + abs(coord[0]) * 2) % 2
            vy= (coord[1] + abs(coord[1]) * 2) % 2
            return (vx,vy)

        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=dict({"log": []})
                dist= float('inf')
                list_rdm=[]
                for coord,site in state.sites.items():
                    rdm2x1 = rdm.rdm2x1(coord,state,env)
                    rdm1x2 = rdm.rdm1x2(coord,state,env)
                    list_rdm.extend([rdm2x1,rdm1x2])

                if len(history["log"]) > 1:
                    dist=0.
                    for i in range(len(list_rdm)):
                        dist+= torch.dist(list_rdm[i], history["rdm"][i], p=2).item()
                history["rdm"]=list_rdm
                history["log"].append(dist)
                if dist<ctm_args.ctm_conv_tol:
                    log.info({"history_length": len(history['log']), "history": history['log']})
                    return True, history
            return False, history

        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

        e_curr0 = model.energy_2x1_1x2(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps=1.0e-12
        self.assertTrue(e_curr0 < eps)
        for coord,site in state.sites.items():
            self.assertTrue(obs_dict[f"m{coord}"] < eps, msg=f"m{coord}")
            for l in ["sz","sp","sm"]:
                self.assertTrue(abs(obs_dict[f"{l}{coord}"]) < eps, msg=f"{l}{coord}")