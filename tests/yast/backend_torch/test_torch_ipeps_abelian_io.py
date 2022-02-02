import yast.yast as yast
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
# import yast.yast.backend_torch.config_U1_U1_R as settings_U1_U1
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN
from ipeps.ipeps_abelian import write_ipeps, read_ipeps
import numpy as np
import unittest
import os

class Test_IO_ipeps_abelian(unittest.TestCase):

    outf_full= "test-write-full_state.json"
    outf_U1= "test-write-U1_state.json"
    outf_U1_U1= "test-write-U1-U1_state.json"

    def setUp(self):
        cfg.global_args.dtype= "float64"

    def _ipeps_abelian_test_equal(self, state0, state1):
        self.assertTrue(state0.nsym==state1.nsym)
        self.assertTrue(state0.sym==state1.sym)
        self.assertTrue(state0.lX==state1.lX)
        self.assertTrue(state0.lY==state1.lY)
        self.assertTrue(state0.sites.keys()==state1.sites.keys())
        for k in state1.sites.keys():
            self.assertTrue(state1.sites[k].config.sym.NSYM==state0.nsym)
            self.assertTrue(state1.sites[k].config.sym.SYM_ID==state0.sym)
            self.assertTrue(state1.sites[k].get_rank()==state0.sites[k].get_rank())
            self.assertTrue(np.array_equal(
                state1.sites[k].get_signature(),state0.sites[k].get_signature()))
            self.assertTrue(np.array_equal(\
                state1.sites[k].get_tensor_charge(),state0.sites[k].get_tensor_charge()))
            if state0.nsym>0:
                self.assertTrue(np.array_equal(\
                    state1.sites[k].get_leg_charges_and_dims(),state0.sites[k].get_leg_charges_and_dims()))
            self.assertTrue(yast.linalg.norm(state0.sites[k]-state1.sites[k])<1.0e-8)

    def test_write_full(self):
        a = yast.rand(config=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        b = yast.rand(config=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_full, sites, vertexToSite)
        write_ipeps(state, self.outf_full)

    def test_write_U1(self):
        a = yast.rand(config=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))

        b = yast.rand(config=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_U1, sites, vertexToSite)
        write_ipeps(state, self.outf_U1)

    # def test_write_U1_U1(self):
    #     a = yast.rand(config=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(1,-1),(1,-1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

    #     b = yast.rand(config=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(-1,1),(-1,1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])
    #     sites=dict({(0,0): a, (1,0): b})

    #     def vertexToSite(r):
    #         x = (r[0] + abs(r[0]) * 2) % 2
    #         y = abs(r[1])
    #         return ((x + y) % 2, 0)

    #     state= IPEPS_ABELIAN(settings_U1_U1, sites, vertexToSite)
    #     write_ipeps(state, self.outf_U1_U1)

    def test_read_full(self):
        a = yast.rand(config=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        b = yast.rand(config=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state0= IPEPS_ABELIAN(settings_full, sites, vertexToSite)
        write_ipeps(state0, self.outf_full)
        state1= read_ipeps(self.outf_full, settings_full, vertexToSite)

        self._ipeps_abelian_test_equal(state0, state1)

    def test_read_U1(self):
        a = yast.rand(config=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))

        b = yast.rand(config=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state0= IPEPS_ABELIAN(settings_U1, sites, vertexToSite)
        write_ipeps(state0, self.outf_U1)
        state1= read_ipeps(self.outf_U1, settings_U1, vertexToSite)

        self._ipeps_abelian_test_equal(state0, state1)

    # def test_read_U1_U1(self):
    #     a = yast.rand(config=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(1,-1),(1,-1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

    #     b = yast.rand(config=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(-1,1),(-1,1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])
    #     sites=dict({(0,0): a, (1,0): b})

    #     def vertexToSite(r):
    #         x = (r[0] + abs(r[0]) * 2) % 2
    #         y = abs(r[1])
    #         return ((x + y) % 2, 0)

    #     state0= IPEPS_ABELIAN(settings_U1_U1, sites, vertexToSite)
    #     write_ipeps(state0, self.outf_U1_U1)
    #     state1= read_ipeps(self.outf_U1_U1, settings_U1_U1, vertexToSite)

    #     self._ipeps_abelian_test_equal(state0, state1)

    def tearDown(self):
        for f in [self.outf_full, self.outf_U1, self.outf_U1_U1]:
            if os.path.isfile(f): os.remove(f)

if __name__ == '__main__':
    unittest.main()


