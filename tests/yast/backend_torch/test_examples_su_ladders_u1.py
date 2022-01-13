import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),\
    '../../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),\
    '../../../examples/abelian')))
from math import isclose
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN
from ipeps.ipeps_abelian import read_ipeps
from examples.abelian.su_ladders_u1 import args
from examples.abelian.su_ladders_u1 import settings_U1
from examples.abelian.su_ladders_u1 import ENV_ABELIAN
from examples.abelian.su_ladders_u1 import ctmrg
from examples.abelian.su_ladders_u1 import coupledLadders
from examples.abelian.su_ladders_u1 import main as main_exec

class Test_IO_ipeps_abelian(unittest.TestCase):

    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run_u1_d2_2x2_neel"

    def test_run_u1_d2_2x2_neel(self):
        args.instate=self.DIR_PATH+"/../input_files/NEEL_D1_2x2_abelian-U1_state.json"
        args.symmetry="U1"
        args.bond_dim=2
        args.opt_max_iter=2
        args.chi=8
        args.alpha=1.0
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype="float64"

        main_exec()

        # read output state and compute observables
        from examples.abelian.su_ladders_u1 import cfg
        model= coupledLadders.COUPLEDLADDERS_U1(settings_U1, alpha=args.alpha)
        state= read_ipeps(self.OUT_PRFX+"_state.json", settings_U1)
        ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
        
        # 1) compute environment by CTMRG
        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history=[]
            e_curr = model.energy_2x1_1x2(state, env).item()
            history.append(e_curr)

            if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
                or len(history) >= ctm_args.ctm_max_iter:
                return True, history
            return False, history

        ctm_env, *ctm_log = ctmrg.run(state, ctm_env, \
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)

        # 2) evaluate loss with converged environment
        loss= model.energy_2x1_1x2(state, ctm_env)
        obs_values, obs_labels = model.eval_obs(state,ctm_env)

        # compare with the reference
        ref_data="""
        4, 0.2, 0.1, -0.6445416898415841, 0.4506543371777977, 0.4506556828285157, 
        0.45065299152707666, 0.4506556828285194, 0.45065299152707905, -0.4506556828285157, 
        0.0, 0.0, 0.45065299152707666, 0.0, 0.0, 0.4506556828285194, 0.0, 0.0, -0.45065299152707905, 
        0.0, 0.0, -0.32583413516431853, -0.32251217411029437, -0.325834135164321, -0.3225121741102977, 
        -0.3207078532083157, -0.32071640993522227, -0.3200248931622293, -0.32002498451133754
        """
        ref_tokens= [float(x) for x in ref_data.split(",")]
        for val,ref_val in zip([loss.item()]+obs_values, ref_tokens[3:]):
            assert isclose(val,ref_val, rel_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json"]:
            if os.path.isfile(f): os.remove(f)

if __name__ == '__main__':
    unittest.main()


