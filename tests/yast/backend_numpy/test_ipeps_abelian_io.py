# import yamps.tensor as TA
import yamps.yast as TA
import settings_full
import settings_U1
# import settings_U1_U1
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

    def _ipeps_abelian_test_equal(self, state0, state1):
        self.assertTrue(state0.nsym==state1.nsym)
        self.assertTrue(state0.sym==state1.sym)
        self.assertTrue(state0.lX==state1.lX)
        self.assertTrue(state0.lY==state1.lY)
        self.assertTrue(state0.sites.keys()==state1.sites.keys())
        for k in state1.sites.keys():
            self.assertTrue(state1.sites[k].config.sym.nsym==state0.nsym)
            self.assertTrue(state1.sites[k].config.sym.name==state0.sym)
            self.assertTrue(state1.sites[k].get_ndim(native=True)==state0.sites[k].get_ndim(native=True))
            self.assertTrue(state1.sites[k].get_ndim()==state0.sites[k].get_ndim())
            self.assertTrue(np.array_equal(state1.sites[k].s,state0.sites[k].s))
            self.assertTrue(np.array_equal(state1.sites[k].n,state0.sites[k].n))
            if state0.nsym>0:
                self.assertTrue(np.array_equal(state1.sites[k].tset,state0.sites[k].tset))
            self.assertTrue(state0.sites[k].norm_diff(state1.sites[k])<1.0e-8)

    def test_write_full(self):
        a = TA.rand(settings=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        b = TA.rand(settings=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        state= IPEPS_ABELIAN(settings_full, sites, vertexToSite)
        write_ipeps(state, self.outf_full)

    def test_write_U1(self):
        a = TA.rand(settings=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))

        b = TA.rand(settings=settings_U1, s=(1, 1, 1, 1, 1), n=1,
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
    #     a = TA.rand(settings=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(1,-1),(1,-1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

    #     b = TA.rand(settings=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
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
        a = TA.rand(settings=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
        b = TA.rand(settings=settings_full, s=(-1, 1, 1, -1, -1), D=(2, 3, 2, 3, 2))
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
        a = TA.rand(settings=settings_U1, s=(1, 1, 1, 1, 1), n=1,
                        t=((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))

        b = TA.rand(settings=settings_U1, s=(1, 1, 1, 1, 1), n=1,
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
    #     a = TA.rand(settings=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
    #                     t=[(1,-1),(1,-1), (0, 2),(0,2), (0,2),(0,2), (0, -2),(0,-2), (0,-2),(0,-2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

    #     b = TA.rand(settings=settings_U1_U1, s=(-1, 1, 1, -1, -1), n=(1,1),
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
