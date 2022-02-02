import numpy as np
import unittest
import settings_full
import settings_U1
# import settings_U1_U1
import yast.yast as yast
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN
from ctm.generic_abelian.env_abelian import ENV_ABELIAN
import ctm.generic_abelian.ctmrg as ctmrg_abelian
import ctm.generic_abelian.rdm as rdm

class Test_ctmrg_rdm_abelian_full(unittest.TestCase):
    
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_full(cls):
        a = yast.zeros(config=settings_full, s=cls._ref_s_dir, D=(2, 2, 2, 2, 2))
        b = yast.zeros(config=settings_full, s=cls._ref_s_dir, D=(2, 2, 2, 2, 2))
        T= np.zeros((2,2,2,2,2))
        T[0,0,0,0,0]= -1.000635518923222
        T[1,1,0,0,0]= -0.421284989637812
        T[1,0,1,0,0]= -0.421284989637812
        T[1,0,0,1,0]= -0.421284989637812
        T[1,0,0,0,1]= -0.421284989637812
        Tb= np.tensordot(np.asarray([[0,1],[-1,0]]), T, axes=([1],[0]))
        a.set_block(val=T)
        b.set_block(val=Tb)
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_full, sites, vertexToSite)
        env= ENV_ABELIAN(chi=8, state=state, init=True)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,
            ctm_args= cfg.ctm_args)

        return state, env_out

    @classmethod
    def setUpClass(cls):
        cls.state_full, cls.env_full= cls._get_2x1_BIPARTITE_full()

    def test_rdm1x1_abelian_2x1_BIPARTITE_full(self):
        state, env= self.state_full, self.env_full
        rho1x1= rdm.rdm1x1((0,0), state, env)

    def test_rdm2x1_abelian_2x1_BIPARTITE_full(self):
        state, env= self.state_full, self.env_full
        rho2x1= rdm.rdm2x1((0,0), state, env)

    def test_rdm1x2_abelian_2x1_BIPARTITE_full(self):
        state, env= self.state_full, self.env_full
        rho1x2= rdm.rdm1x2((0,0), state, env)

    def test_rdm2x2_abelian_2x1_BIPARTITE_full(self):
        state, env= self.state_full, self.env_full
        rho2x2= rdm.rdm2x2((0,0), state, env)

class Test_ctmrg_rdm_abelian_U1(unittest.TestCase):
    
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls):
        # AFM D=2
        a = yast.Tensor(config=settings_U1, s=cls._ref_s_dir, n=0)
                        # t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        tmp1= -1.000635518923222*np.ones((1,1,1,1,1))
        tmp2= -0.421284989637812*np.ones((1,1,1,1,1))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val=tmp1)
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp2)

        b = yast.Tensor(config=settings_U1, s=cls._ref_s_dir, n=0)
                        # t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        b.set_block((0,0,0,0,0), (1,1,1,1,1), val=-tmp1)
        b.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp2)

        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_U1, sites, vertexToSite)
        env= ENV_ABELIAN(chi=8, state=state, init=True)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= np.sort(s.to_numpy().diagonal())[::-1]
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,
            ctm_args= cfg.ctm_args)

        return state, env

    @classmethod
    def setUpClass(cls):
        cls.state_u1, cls.env_u1= cls._get_2x1_BIPARTITE_U1()

    def test_rdm1x1_abelian_2x1_BIPARTITE_U1(self):
        state, env= self.state_u1, self.env_u1
        rho1x1= rdm.rdm1x1((0,0), state, env)

    def test_rdm2x1_abelian_2x1_BIPARTITE_U1(self):
        state, env= self.state_u1, self.env_u1
        rho2x1= rdm.rdm2x1((0,0), state, env)

    def test_rdm1x2_abelian_2x1_BIPARTITE_U1(self):
        state, env= self.state_u1, self.env_u1
        rho1x2= rdm.rdm1x2((0,0), state, env)

    def test_rdm2x2_abelian_2x1_BIPARTITE_U1(self):
        state, env= self.state_u1, self.env_u1
        rho2x2= rdm.rdm2x2((0,0), state, env)

# TODO test correctness by comparing eigenvalues of RDMs from 
#      full and U1 cases

if __name__ == '__main__':
    unittest.main()