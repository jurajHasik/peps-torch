import context
import os
import torch
import unittest
from ipeps.ipeps import IPEPS, read_ipeps
torch.set_default_dtype(torch.float64)

class TestIPEPSBasic(unittest.TestCase):
    TEST_STATE_OUT="TEST_STATE_OUT.json"

    def setUp(self):
        # Simple tensors for testing
        self.a = torch.ones(2, 1,1,1,1)
        self.b = torch.zeros(2, 1,1,1,1) + 2
        self.c = torch.zeros(2, 1,1,1,1) + 3
        self.d = torch.zeros(2, 1,1,1,1) + 4
        self.e = torch.zeros(2, 1,1,1,1) + 5
        self.f = torch.zeros(2, 1,1,1,1) + 6

    def test_1site_vertexToSite(self):
        # Example 1: 1-site translational iPEPS
        def asserts(w):
            self.assertTrue(torch.equal(w.site((0,0)), self.a))
            self.assertTrue(torch.equal(w.site((5,5)), self.a))
        
        sites = {(0,0): self.a}
        def vertexToSite(coord):
            return (0,0)
        wfc = IPEPS(sites, vertexToSite)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=vertexToSite)
        asserts(wfc2)

    def test_2site_vertexToSite(self):
        # Example 2: 2-site bipartite iPEPS
        def asserts(w):
            self.assertTrue(torch.equal(w.site((0,0)), self.a))
            self.assertTrue(torch.equal(w.site((1,0)), self.b))
            self.assertTrue(torch.equal(w.site((2,0)), self.a))
            self.assertTrue(torch.equal(w.site((0,1)), self.b))
        
        sites = {(0,0): self.a, (1,0): self.b}
        def vertexToSite(coord):
            x = (coord[0] + abs(coord[0]) * 2) % 2
            y = abs(coord[1])
            return ((x + y) % 2, 0)
        wfc = IPEPS(sites, vertexToSite)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=vertexToSite)
        asserts(wfc2)

    def test_3x2_vertexToSite(self):
        # Example 3: iPEPS with 3x2 unit cell with PBC
        def asserts(w):
            self.assertTrue(torch.equal(w.site((0,0)), self.a))
            self.assertTrue(torch.equal(w.site((1,0)), self.b))
            self.assertTrue(torch.equal(w.site((2,0)), self.c))
            self.assertTrue(torch.equal(w.site((0,1)), self.d))
            self.assertTrue(torch.equal(w.site((1,1)), self.e))
            self.assertTrue(torch.equal(w.site((2,1)), self.f))
            # Test periodicity
            self.assertTrue(torch.equal(w.site((3,0)), self.a))
            self.assertTrue(torch.equal(w.site((4,1)), self.e))
            
        sites = {
            (0,0): self.a, (1,0): self.b, (2,0): self.c,
            (0,1): self.d, (1,1): self.e, (2,1): self.f
        }
        wfc = IPEPS(sites, lX=3, lY=2)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=None)
        asserts(wfc2)

    def test_pattern_from_pattern_bipartite(self):
        def asserts(w):
            self.assertTrue(torch.equal(wfc.site((0,0)), self.a))
            self.assertTrue(torch.equal(wfc.site((1,0)), self.b))
            self.assertTrue(torch.equal(wfc.site((0,1)), self.b))
            self.assertTrue(torch.equal(wfc.site((1,1)), self.a))
            # Test periodicity
            self.assertTrue(torch.equal(wfc.site((2,0)), self.a))
            self.assertTrue(torch.equal(wfc.site((3,1)), self.a))

        # Example: pattern as in from_pattern docstring
        pattern = [[0, 1], [1, 0]]
        sites = {(0,0): self.a, (1,0): self.b}
        wfc = IPEPS(sites, pattern=pattern)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=None)
        asserts(wfc2)
        

    def test_pattern_from_pattern_120deg(self):
        def asserts(w):
            self.assertTrue(torch.equal(w.site((0,0)), self.a))
            self.assertTrue(torch.equal(w.site((1,0)), self.b))
            self.assertTrue(torch.equal(w.site((0,1)), self.b))
            self.assertTrue(torch.equal(w.site((1,1)), self.c))
            # Test periodicity
            self.assertTrue(torch.equal(w.site((3,0)), self.a))
            self.assertTrue(torch.equal(w.site((3,1)), self.b))
            self.assertTrue(torch.equal(w.site((-1,1)), self.a))

        # Example: pattern as in from_pattern docstring
        pattern = [
            ["a", "b", "c"], 
            ["b", "c", "a"], # "b"
            ["c", "a", "b"] 
            ]
        sites = {(0,0): self.a, (1,0): self.b, (2,0): self.c,}
        wfc = IPEPS(sites, pattern=pattern)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=None)
        asserts(wfc2)
        

    def test_pattern_from_pattern_zigzag(self):
        def asserts(w):
            self.assertTrue(torch.equal(w.site((0,0)), self.a))
            self.assertTrue(torch.equal(w.site((1,0)), self.b))
            self.assertTrue(torch.equal(w.site((2,0)), self.c))
            self.assertTrue(torch.equal(w.site((3,0)), self.d))
            # Test periodicity
            self.assertTrue(torch.equal(w.site((4,0)), self.a))
            self.assertTrue(torch.equal(w.site((4,1)), self.c))
            self.assertTrue(torch.equal(w.site((3,2)), self.d))
            
        # Example: pattern as in from_pattern docstring
        pattern = [
            [(0,0), (1,0), (0,1), (1,1)], # (0,0)
            [(0,1), (1,1), (0,0), (1,0)], # (0,1)
            ]
        sites = {(0,0): self.a, (1,0): self.b, (0,1): self.c, (1,1): self.d}
        wfc = IPEPS(sites, pattern=pattern)
        asserts(wfc)

        wfc.write_to_file(self.TEST_STATE_OUT)
        wfc2= read_ipeps(self.TEST_STATE_OUT, vertexToSite=None)
        asserts(wfc2)

    def tearDown(self):
        if os.path.isfile(self.TEST_STATE_OUT): os.remove(self.TEST_STATE_OUT)

if __name__ == "__main__":
    unittest.main()