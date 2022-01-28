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

class Test_ctmrg_abelian(unittest.TestCase):
    
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_full(cls):
        a = yast.rand(config=settings_full, s=cls._ref_s_dir, D=(2, 3, 2, 3, 2))
        b = yast.rand(config=settings_full, s=cls._ref_s_dir, D=(2, 3, 2, 3, 2))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_full, sites, vertexToSite)

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls):
        # AFM D=2
        a = yast.Tensor(config=settings_U1, s=cls._ref_s_dir, n=0)
                        # t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val='ones')
        tmp_B= 0.3*np.ones((1,1,1,1,1))
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp_B)

        b = yast.Tensor(config=settings_U1, s=cls._ref_s_dir, n=0)
                        # t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        b.set_block((0,0,0,0,0), (1,1,1,1,1), val=-1*np.ones((1,1,1,1,1)))
        b.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp_B)
        b.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp_B)

        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    @classmethod
    def _get_1x1_U1(cls):
        # AFM D=2
        a = yast.zeros(config=settings_U1, s=cls._ref_s_dir, n=0,
                        t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val='ones')
        tmp_B= 0.3*np.ones((1,1,1,1,1))
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp_B)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp_B)

        sites=dict({(0,0): a})

        def vertexToSite(r): return (0, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    @classmethod
    def _get_2x1_BIPARTITE_U1_U1(cls):
        a = yast.rand(config=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
                        t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
                        D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

        b = yast.rand(config=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
                        t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
                        D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1_U1, sites, vertexToSite)

    def setUp(self):
        pass


    def test_ctmrg_abelian_full_chi1(self):
        state= self._get_2x1_BIPARTITE_full()
        env= ENV_ABELIAN(state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,
            ctm_args= cfg.ctm_args)

    def test_ctmrg_abelian_full(self):
        state= self._get_2x1_BIPARTITE_full()
        env= ENV_ABELIAN(chi=8, state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

    def test_ctmrg_abelian_2x1_BIPARTITE_U1(self):
        chi=9
        np.random.seed(2)
        state= self._get_2x1_BIPARTITE_U1()
        env= ENV_ABELIAN(chi=chi, state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        print("----- CTMRG_ABELIAN FINISHED -----")

        state_dense= state.to_dense()
        env_dense= ENV_ABELIAN(chi=chi, state=state_dense, init=True)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        env_out, *ctm_log= ctmrg_abelian.run(state_dense, env_dense, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

    def IGNORE_test_ctmrg_abelian_1x1_U1(self):
        chi=9
        np.random.seed(2)
        state= self._get_1x1_U1()
        env= ENV_ABELIAN(chi=chi, state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        print("----- CTMRG_ABELIAN FINISHED -----")

        state_dense= state.to_dense()
        env_dense= ENV_ABELIAN(chi=chi, state=state_dense, init=True)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        env_out, *ctm_log= ctmrg_abelian.run(state_dense, env_dense, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

if __name__ == '__main__':
    unittest.main()
