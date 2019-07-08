import torch
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-omp_cores", type=int, default=1,help="number of OpenMP cores")
parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
parser.add_argument("-instate", default=None, help="Input state JSON")
parser.add_argument("-ipeps_init_type", default="RANDOM", help="initialization of trial iPEPS state")
parser.add_argument("-out_prefix", default="output", help="Output files prefix")
parser.add_argument("-bond_dim", type=int, default=1, help="iPEPS auxiliary bond dimension")
parser.add_argument("-chi", type=int, default=20, help="environment bond dimension")
parser.add_argument("-ctm_max_iter", type=int, default=1, help="maximal number of CTM iterations")
parser.add_argument("-opt_max_iter", type=int, default=100, help="maximal number of CTM iterations")

args = parser.parse_args()

class GLOBALARGS():
    def __init__(self):
        self.dtype = torch.float64
        self.device = 'cpu'

class PEPSARGS():
    def __init__(self):
        pass

class CTMARGS():
    def __init__(self):
        self.ctm_max_iter = 50
        self.ctm_env_init_type = 'CTMRG'
        self.ctm_conv_tol = 1.0e-10
        self.projector_svd_reltol = 1.0e-8
        self.ctm_move_sequence = [(0,-1), (-1,0), (0,1), (1,0)]
        self.verbosity_initialization = 1
        self.verbosity_ctm_convergence = 1
        self.verbosity_projectors = 0
        self.verbosity_ctm_move = 0

class OPTARGS():
    def __init__(self):
        self.opt_ctm_reinit = True
        self.lr = 1.0
        self.max_iter_per_epoch = 20
        pass
