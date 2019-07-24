import torch
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-omp_cores", type=int, default=1,help="number of OpenMP cores")
parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
parser.add_argument("-instate", default=None, help="Input state JSON")
parser.add_argument("-instate_noise", type=float, default=0., help="magnitude of noise added to the trial \"instate\"")
parser.add_argument("-ipeps_init_type", default="RANDOM", help="initialization of the trial iPEPS state")
parser.add_argument("-out_prefix", default="output", help="Output files prefix")
parser.add_argument("-bond_dim", type=int, default=1, help="iPEPS auxiliary bond dimension")
parser.add_argument("-chi", type=int, default=20, help="environment bond dimension")
parser.add_argument("-ctm_max_iter", type=int, default=1, help="maximal number of CTM iterations")
parser.add_argument("-opt_max_iter", type=int, default=100, help="maximal number of CTM iterations")
parser.add_argument("-resume", type=str, default=None, help="file with checkpoint to resume")
#args = parser.parse_args()

class GLOBALARGS():
    r"""
    Holds global configuration options

    :ivar dtype: data type of all torch.tensor. Default: ``torch.float64``
    :vartype dtype: torch.dtype
    :ivar device: device on which all the torch.tensors are stored. Default: ``'cpu'``
    :vartype device: str
    """
    def __init__(self): 
        self.dtype = torch.float64
        self.device = 'cpu'

class PEPSARGS():
    def __init__(self):
        pass

class CTMARGS():
    r"""
    Holds configuration of CTM algorithm

    :ivar ctm_max_iter: maximum iterations of directional CTM algorithm. Default: ``50``
    :vartype ctm_max_iter: int
    :ivar ctm_env_init_type: default initialization method for ENV objects. Default: ``'CTMRG'``
    :vartype ctm_env_init_type: str
    :ivar ctm_conv_tol: threshold for convergence of CTM algorithm. Default: ``'1.0e-10'``
    :vartype ctm_conv_tol: float
    :ivar projector_svd_reltol: relative threshold on the magnitude of the smallest elements of 
                                singular value spectrum used in the construction of projectors. 
                                Default: ``1.0e-8``
    :vartype projector_svd_reltol: float
    :ivar ctm_move_sequence: sequence of directional moves within single CTM iteration. The possible 
                             directions are encoded as tuples(int,int) 
                                
                                * up = (0,-1)
                                * left = (-1,0)
                                * down = (0,1)
                                * right = (1,0)

                             Default: ``[(0,-1), (-1,0), (0,1), (1,0)]``
    :vartype ctm_move_sequence: list[tuple(int,int)]
    :ivar verbosity_initialization: verbosity of initialization method for ENV objects. Default: ``0``
    :vartype verbosity_initialization: int
    :ivar verbosity_ctm_convergence: verbosity of evaluation of CTM convergence criterion. Default: ``0``
    :vartype verbosity_ctm_convergence: int
    :ivar verbosity_projectors: verbosity of projector construction. Default: ``0``
    :vartype verbosity_projectors: int
    :ivar verbosity_ctm_move: verbosity of directional CTM moves. Default: ``0``
    :vartype verbosity_ctm_move: int
    """
    def __init__(self):
        self.ctm_max_iter = 50
        self.ctm_env_init_type = 'CTMRG'
        self.ctm_conv_tol = 1.0e-10
        self.projector_svd_reltol = 1.0e-8
        self.ctm_move_sequence = [(0,-1), (-1,0), (0,1), (1,0)]
        self.verbosity_initialization = 0
        self.verbosity_ctm_convergence = 0
        self.verbosity_projectors = 0
        self.verbosity_ctm_move = 0

class OPTARGS():
    r"""
    Holds configuration of optimization process

    :ivar opt_ctm_reinit: reinitialize environment from scratch within every loss 
                          function evaluation. Default: ``True``
    :vartype opt_ctm_reinit: bool
    :ivar lr: initial learning rate. Default: ``1.0``
    :vartype lr: float
    :ivar max_iter_per_epoch: maximum number of optimizer iterations per epoch. Default: ``1``
    :vartype max_iter_per_epoch: int
    :ivar verbosity_opt_epoch: verbosity within optimization epoch. Default: ``1``
    :vartype verbosity_opt_epoch: int
    :ivar resume: path to the checkpoint file used to resume a computation from optimization. If resume is None, a new computation is initialized. Default: ``None``
    :vartype resume: str
    """
    def __init__(self):
        self.opt_ctm_reinit = True
        self.lr = 1.0
        self.max_iter_per_epoch = 20
        self.verbosity_opt_epoch = 1
        self.resume = None
