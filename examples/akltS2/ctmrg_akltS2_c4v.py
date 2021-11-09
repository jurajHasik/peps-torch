import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import akltS2
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = akltS2.AKLTS2_C4V_BIPARTITE()
    
    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # convergence by 2x1 subsystem reduced density matrix
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=dict({"log": []})
            rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
            dist= float('inf')
            
            # compute observables
            e_curr = model.energy_1x1(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            print(", ".join([f"{len(history['log'])}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history["log"]) > 1:
                dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
            history["rdm"]=rdm2x1
            history["log"].append(dist)
            if dist<ctm_args.ctm_conv_tol:
                log.info({"history_length": len(history['log']), "history": history['log']})
                return True, history
        return False, history

    # convergence by spectrum of the corner matrix
    # def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
    #     with torch.no_grad():
    #         if not history:
    #             history=[]
    #         e_curr = model.energy_1x1(state, env)
    #         obs_values, obs_labels = model.eval_obs(state, env)
    #         print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
            
    #         u,s,v= torch.svd(env.C[env.keyC], compute_uv=False)
    #         history.append([s]+[e_curr.item()]+obs_values)

    #         if len(history) > 1 and torch.dist(history[-1][0],history[-2][0]) < ctm_args.ctm_conv_tol:
    #             return True
    #     return False

    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_1x1(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    corrSS= model.eval_corrf_SS(state, ctm_env_init, args.corrf_r)
    print("\nr "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrDD= model.eval_corrf_DD_H(state, ctm_env_init, args.corrf_r)
    print("\nr "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    # environment diagnostics
    print("\nspectrum(C)")
    u,s,v= torch.svd(ctm_env_init.C[ctm_env_init.keyC], compute_uv=False)
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
        args.opt_max_iter=2
        args.instate=None

    # basic tests
    def test_ctmrg_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

class TestAKLT(unittest.TestCase):
    def setUp(self):
        import os
        args.instate=os.path.dirname(os.path.realpath(__file__))+"/../../test-input/AKLT-S2_1x1.in"
        args.chi=32
        args.opt_max_iter=50

    def test_ctmrg_AKLT(self):
        cfg.configure(args)
        torch.set_num_threads(args.omp_cores)
    
        model = akltS2.AKLTS2_C4V_BIPARTITE()
        state = read_ipeps_c4v(args.instate)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=dict({"log": []})
                rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
                dist= float('inf')
                if len(history["log"]) > 1:
                    dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
                history["rdm"]=rdm2x1
                history["log"].append(dist)
                if dist<ctm_args.ctm_conv_tol:
                    log.info({"history_length": len(history['log']), "history": history['log']})
                    return True, history
            return False, history

        ctm_env_init = ENV_C4V(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

        e_curr0 = model.energy_1x1(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps=1.0e-14
        self.assertTrue(e_curr0 < eps)
        self.assertTrue(obs_dict["m"] < eps)
        for l in ["sz","sp","sm"]:
            self.assertTrue(abs(obs_dict[l]) < eps)