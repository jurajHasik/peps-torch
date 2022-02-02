import pathlib
import numpy as np
import unittest
import settings_full
import settings_U1
import yast.yast as yast
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN, read_ipeps
from ctm.generic_abelian.env_abelian import ENV_ABELIAN
import ctm.generic_abelian.ctmrg as ctmrg_abelian

class Test_ctmrg_abelian_U1_D5_basic(unittest.TestCase):
    
    chi= 25
    instate= "../input_files/D5_1x1_U1_state.json"
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_1x1_full(cls):
        instate= pathlib.Path(__file__).parent.absolute() / cls.instate
        state_U1= read_ipeps(instate, settings_U1)
        # set correct signature
        T0= state_U1.site((0,0))
        a= yast.Tensor(config=T0.config, s=cls._ref_s_dir, n=T0.get_tensor_charge())
        for c,block in T0.A.items(): 
            a.set_block(tuple(cls._ref_s_dir*np.asarray(c)), block.shape, val=block) 
        state_U1.sites= {(0,0): a}
        state_dense= state_U1.to_dense()
        return state_dense

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls):
        instate= pathlib.Path(__file__).parent.absolute() / cls.instate
        state_U1= read_ipeps(instate, settings_U1)

        # create 2x1 bipartite
        T0= state_U1.site((0,0))
        a= yast.Tensor(config=T0.config, s=cls._ref_s_dir, n=T0.get_tensor_charge())
        for c,block in T0.A.items():
            a.set_block(tuple(cls._ref_s_dir*np.asarray(c)), block.shape, val=block) 
        b= yast.Tensor(config=a.config, s=cls._ref_s_dir, n=-np.asarray(a.get_tensor_charge()))
        for c,block in a.A.items():
            b.set_block(tuple(-np.asarray(c)), block.shape, val=block)

        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    def setUp(self):
        pass

    def test_ctmrg_abelian_1x1_full(self):
        state= self._get_1x1_full()
        env= ENV_ABELIAN(chi=25, state=state, init=True)
        print(state)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            if not history: history=0
            history+=1
            print(history)
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

    def test_ctmrg_abelian_2x1_BIPARTITE_U1(self):
        np.random.seed(2)
        state= self._get_2x1_BIPARTITE_U1()
        env= ENV_ABELIAN(chi=self.chi, state=state, init=True)
        print(env)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= np.sort(s.to_numpy().diagonal())[::-1]
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state, env, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

    def test_ctmrg_abelian_2x1_BIPARTITE_full(self):
        np.random.seed(2)
        state= self._get_2x1_BIPARTITE_U1()
        state_dense= state.to_dense()
        env_dense= ENV_ABELIAN(chi=self.chi, state=state_dense, init=True)
        print(env_dense)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            for cid,c in env.C.items():
                u,s,v= yast.linalg.svd(c,(0,1))
                s= s.to_numpy().diagonal()
                print(f"{cid}: {s}")
            return False, history

        cfg.ctm_args.ctm_max_iter=2
        env_out, *ctm_log= ctmrg_abelian.run(state_dense, env_dense, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

class Test_ctmrg_abelian_U1_D5_spectra(unittest.TestCase):
    
    chi= 25
    instate= "../input_files/D5_1x1_U1_state.json"
    _ref_s_dir= IPEPS_ABELIAN._REF_S_DIRS

    @classmethod
    def _get_1x1_full(cls):
        instate= pathlib.Path(__file__).parent.absolute() / cls.instate
        state_U1= read_ipeps(instate, settings_U1)
        state_dense= state_U1.to_dense()
        return state_dense

    @classmethod
    def _get_2x1_BIPARTITE_U1(cls):
        instate= pathlib.Path(__file__).parent.absolute() / cls.instate
        state_U1= read_ipeps(instate, settings_U1)

        # create 2x1 bipartite
        T0= state_U1.site((0,0))
        a= yast.Tensor(config=T0.config, s=cls._ref_s_dir, n=T0.get_tensor_charge())
        for c,block in T0.A.items():
            a.set_block(tuple(cls._ref_s_dir*np.asarray(c)), block.shape, val=block) 
        b= yast.Tensor(config=a.config, s=cls._ref_s_dir, \
            n=-np.asarray(a.get_tensor_charge()))
        for c,block in a.A.items():
            b.set_block(tuple(-np.asarray(c)), block.shape, val=block)

        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    def setUp(self):
        pass

    @unittest.skip("Execution time is expected to be long")
    def test_ctmrg_abelian_2x1_BIPARTITE_U1_vs_2x1_BIPARTITE_full_vs_1x1_full(self):
        cfg.ctm_args.ctm_max_iter=50

        # ----- 1x1_full -----
        state1x1= self._get_1x1_full()
        env1x1= ENV_ABELIAN(chi=self.chi, state=state1x1, init=True)
        print(state1x1)
        print(env1x1)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            if not history: history=0
            history+=1
            print(history,end=" ")
            return False, history

        env1x1, *ctm_log= ctmrg_abelian.run(state1x1, env1x1, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        # store corner spectra
        s_1x1={}
        for cid,c in env1x1.C.items():
            u,s,v= yast.linalg.svd(c,(0,1))
            s_1x1[cid]= s.to_numpy().diagonal()
        
        # ----- 2x1_BIPARTITE_full -----
        stateU1= self._get_2x1_BIPARTITE_U1()
        state_dense= stateU1.to_dense()
        env_dense= ENV_ABELIAN(chi=self.chi, state=state_dense, init=True)
        print(env_dense)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            if not history: history=0
            history+=1
            print(history,end=" ")
            return False, history

        env_dense, *ctm_log= ctmrg_abelian.run(state_dense, env_dense, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

        s_2x1_full={}
        for cid,c in env_dense.C.items():
            u,s,v= yast.linalg.svd(c,(0,1))
            s_2x1_full[cid]= s.to_numpy().diagonal()

        # ----- 2x1_BIPARTITE_U1 -----
        envU1= ENV_ABELIAN(chi=self.chi, state=stateU1, init=True)
        print(envU1)

        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            # compute SVD of corners
            if not history: history=0
            history+=1
            print(history,end=" ")
            return False, history

        envU1, *ctm_log= ctmrg_abelian.run(stateU1, envU1, conv_check=ctmrg_conv_f,\
            ctm_args=cfg.ctm_args)

        s_2x1_U1={}
        for cid,c in envU1.C.items():
            u,s,v= yast.linalg.svd(c,(0,1))
            s_2x1_U1[cid]= np.sort(s.to_numpy().diagonal())[::-1]  
        
        # ----- compare -----
        for cid in s_1x1:
            assert np.allclose(s_1x1[cid],s_2x1_full[cid])
            assert np.allclose(s_1x1[cid],s_2x1_U1[cid])

if __name__ == '__main__':
    unittest.main()