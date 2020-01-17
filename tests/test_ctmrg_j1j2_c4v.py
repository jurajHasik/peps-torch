import context
import torch
import argparse
import config as cfg
from ipeps import *
from groups.c4v import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0.5, help="next nearest-neighbour coupling")
args, unknown= parser.parse_known_args()

class TestCtmrg(unittest.TestCase):
    
    def setUp(self):
        args.j2=0.5
        args.bond_dim=2
        args.chi=16
        args.ipeps_init_type=='RANDOM'
        args.GLOBALARGS_device="cpu"

    def test_energy_GESDD(self):
        args.CTMARGS_projector_svd_method="GESDD"
        self._energy()

    def test_energy_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        self._energy()

    def _energy(self):
        cfg.configure(args)
        cfg.print_config()
        torch.set_num_threads(args.omp_cores)
    
        model = j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
        energy_f= model.energy_1x1_lowmem

        # initialize random ipeps
        bond_dim = args.bond_dim
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))
        sites = {(0,0): A}
        state = IPEPS(sites)

        print(state)

        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                e_curr = energy_f(state, env)
                obs_values, obs_labels = model.eval_obs(state, env)
                history.append([e_curr.item()]+obs_values)
                print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

                if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                    return True
            return False

        ctm_env_init = ENV_C4V(args.chi, state)
        init_env(state, ctm_env_init)
        print(ctm_env_init)

        # initial environment
        e_ref0= model.energy_1x1(state, ctm_env_init)
        e_curr0= energy_f(state, ctm_env_init)
        self.assertTrue(torch.allclose(e_ref0, e_curr0, rtol=1e-08, atol=1e-08))
        obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

        print(", ".join(["epoch","energy"]+obs_labels))
        print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

        ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

        # converged environment
        e_ref0= model.energy_1x1(state, ctm_env_init)
        e_curr0= energy_f(state, ctm_env_init)
        self.assertTrue(torch.allclose(e_ref0, e_curr0, rtol=1e-08, atol=1e-08))

if __name__=='__main__':
    unittest.main()