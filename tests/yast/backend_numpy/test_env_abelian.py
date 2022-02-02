import numpy as np
import unittest
import settings_full
import settings_U1
# import settings_U1_U1
import yast.yast as yast
import config as cfg
from ipeps.ipeps_abelian import IPEPS_ABELIAN
from ctm.generic_abelian.env_abelian import ENV_ABELIAN

class Test_env_abelian(unittest.TestCase):
    
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
        a = yast.rand(config=settings_U1, s=cls._ref_s_dir, n=1,
                        t=((-1, 1), (0, -2), (0, -2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))

        b = yast.rand(config=settings_U1, s=cls._ref_s_dir, n=1,
                        t=((-1, 1), (0, -2), (0, -2), (0, 2), (0, 2)),
                        D=((1, 1), (2,1), (2,1), (2,1), (2,1)))
        sites=dict({(0,0): a, (1,0): b})

        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        return IPEPS_ABELIAN(settings_U1, sites, vertexToSite)

    # @classmethod
    # def _get_2x1_BIPARTITE_U1_U1(cls):
    #     a = yast.rand(config=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
    #                     t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])

    #     b = yast.rand(config=settings_U1_U1, s=cls._ref_s_dir, n=(1,1),
    #                     t=[(-1,1),(-1,1), (0,-2),(0,-2), (0,-2),(0,-2), (0,2),(0,2), (0,2),(0,2)],
    #                     D=[(1,1),(1,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1), (2,1),(2,1)])
    #     sites=dict({(0,0): a, (1,0): b})

    #     def vertexToSite(r):
    #         x = (r[0] + abs(r[0]) * 2) % 2
    #         y = abs(r[1])
    #         return ((x + y) % 2, 0)

    #     return IPEPS_ABELIAN(settings_U1_U1, sites, vertexToSite)

    def setUp(self):
        pass

    # basic tests
    def test_env_abelian_init_full(self):
        state= self._get_2x1_BIPARTITE_full()
        env= ENV_ABELIAN(state=state, init=True)

    def test_env_abelian_init_U1(self):
        state= self._get_2x1_BIPARTITE_U1()
        env= ENV_ABELIAN(state=state, init=True)

if __name__ == '__main__':
    unittest.main()
