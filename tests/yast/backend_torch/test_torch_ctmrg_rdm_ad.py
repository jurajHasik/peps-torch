import numpy as np
import torch
import unittest
import settings_full_torch
import settings_U1_torch
import yast.yast as yast
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN
from ctm.generic_abelian.env_abelian import ENV_ABELIAN
import ctm.generic_abelian.ctmrg as ctmrg_abelian
import ctm.generic_abelian.rdm as rdm

class Test_ctmrg_abelian_full_torch(unittest.TestCase):
    
    chi=8
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_full(cls, checkpoint=False):
        a = yast.zeros(config=settings_full_torch, s=cls._ref_s_dir, D=(2, 2, 2, 2, 2))
        b = yast.zeros(config=settings_full_torch, s=cls._ref_s_dir, D=(2, 2, 2, 2, 2))
        T= np.zeros((2,2,2,2,2))
        T[0,0,0,0,0]= -1.000635518923222
        T[1,1,0,0,0]= -0.421284989637812
        T[1,0,1,0,0]= -0.421284989637812
        T[1,0,0,1,0]= -0.421284989637812
        T[1,0,0,0,1]= -0.421284989637812
        Tb= np.tensordot(np.asarray([[0,1],[-1,0]]), T, axes=([1],[0]))
        a.set_block(val=T)
        b.set_block(val=Tb)
        for t in a.A.values(): t.requires_grad_(True)
        for t in b.A.values(): t.requires_grad_(True)
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_full_torch, sites, vertexToSite)
        env= ENV_ABELIAN(chi=cls.chi, state=state, init=True)

        @torch.no_grad()
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        cfg.ctm_args.fwd_checkpoint_move= checkpoint
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,
            ctm_args= cfg.ctm_args)

        return state, env_out

    @classmethod
    def setUpClass(cls):
        cfg.global_args.dtype= "float64"

    def test_rdm1x1_abelian_2x1_BIPARTITE_full(self):
        state, env= self._get_2x1_BIPARTITE_full()
        rho1x1= rdm.rdm1x1((0,0), state, env)
        rho1x1.A[()][0,0].backward()

    def test_rdm1x1_abelian_2x1_BIPARTITE_full_checkpoint(self):
        state, env= self._get_2x1_BIPARTITE_full(checkpoint=True)
        rho1x1= rdm.rdm1x1((0,0), state, env)
        rho1x1.A[()][0,0].backward()

class Test_ctmrg_abelian_U1_torch(unittest.TestCase):
    
    chi=8
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls, checkpoint=False):
        # AFM D=2
        a = yast.Tensor(config=settings_U1_torch, s=cls._ref_s_dir, n=0)
                        # t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        tmp1= -1.000635518923222*np.ones((1,1,1,1,1))
        tmp2= -0.421284989637812*np.ones((1,1,1,1,1))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val=tmp1)
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp2)

        b = yast.Tensor(config=settings_U1_torch, s=cls._ref_s_dir, n=0)
                        # t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        # D=((1, 1), (1,1), (1,1), (1,1), (1,1)))
        b.set_block((0,0,0,0,0), (1,1,1,1,1), val=-tmp1)
        b.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp2)

        for t in a.A.values(): t.requires_grad_(True)
        for t in b.A.values(): t.requires_grad_(True)
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_U1_torch, sites, vertexToSite)
        env= ENV_ABELIAN(chi=cls.chi, state=state, init=True)

        @torch.no_grad()
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= np.sort(s.to_numpy().diagonal())[::-1]
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter= 2
        cfg.ctm_args.fwd_checkpoint_move= checkpoint
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,
            ctm_args= cfg.ctm_args)

        return state, env

    @classmethod
    def setUpClass(cls):
        cfg.global_args.dtype= "float64"

    def test_rdm1x1_abelian_2x1_BIPARTITE_U1(self):
        state, env= self._get_2x1_BIPARTITE_U1()
        rho1x1= rdm.rdm1x1((0,0), state, env)
        next(iter(rho1x1.A.values()))[0,0].backward()

    def test_rdm1x1_abelian_2x1_BIPARTITE_U1_checkpoint(self):
        state, env= self._get_2x1_BIPARTITE_U1(checkpoint=True)
        rho1x1= rdm.rdm1x1((0,0), state, env)
        next(iter(rho1x1.A.values())).backward()

# TODO test correctness by comparing eigenvalues of RDMs from 
#      full and U1 cases

if __name__ == '__main__':
    unittest.main()