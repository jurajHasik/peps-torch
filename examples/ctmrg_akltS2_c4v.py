import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from models import akltS2
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
# additional observables-related arguments
parser.add_argument("-corrf_r", type=int, default=1, help="maximal correlation function distance")
args, unknown = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    
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
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
    #     with torch.no_grad():
    #         e_curr = model.energy_1x1(state, env)
    #         obs_values, obs_labels = model.eval_obs(state, env)
    #         history.append([e_curr.item()]+obs_values)
    #         print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

    #         if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
    #             return True
    #     return False

    def ctmrg_conv_specdistC(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            e_curr = model.energy_1x1(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
            
            u,s,v= torch.svd(env.C[env.keyC], compute_uv=False)
            history.append([s]+[e_curr.item()]+obs_values)

            if len(history) > 1 and torch.dist(history[-1][0],history[-2][0]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_1x1(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_specdistC)

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
    main()

class TestCtmrg(unittest.TestCase):
    def setUp(self):
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=2

    # basic tests
    def test_ctmrg_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()  

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()