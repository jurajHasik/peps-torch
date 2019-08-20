import torch
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-omp_cores", type=int, default=1,help="number of OpenMP cores")
    parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
    parser.add_argument("-instate", default=None, help="Input state JSON")
    parser.add_argument("-instate_noise", type=float, default=0., help="magnitude of noise added to the trial \"instate\"")
    parser.add_argument("-ipeps_init_type", default="RANDOM", help="initialization of the trial iPEPS state")
    parser.add_argument("-out_prefix", default="output", help="Output files prefix")
    parser.add_argument("-bond_dim", type=int, default=1, help="iPEPS auxiliary bond dimension")
    parser.add_argument("-chi", type=int, default=20, help="environment bond dimension")
    parser.add_argument("-opt_max_iter", type=int, default=100, help="maximal number of epochs")
    parser.add_argument("-opt_resume", type=str, default=None, help="file with checkpoint to resume")
    parser.add_argument("-opt_resume_override_params", type=bool, default=False, help="override optimizer parameters stored in checkpoint")

    configs=[global_args, peps_args, ctm_args, opt_args]
    for c in configs:
        group_name=type(c).__name__
        group_prefix=group_name+"_" 
        parser.add_argument_group(group_prefix)
        c_list= list(filter(lambda x: "__" not in x, dir(c)))
        for x in c_list:
            if isinstance(getattr(c,x), bool):
                # default is False
                if not getattr(c,x): 
                    parser.add_argument("-"+group_prefix+x, action='store_true')
                # default is True
                else:
                    parser.add_argument("-"+group_prefix+"no_"+x, action='store_false', dest=group_prefix+x)
            else:
                parser.add_argument("-"+group_prefix+x, type=type(getattr(c,x)), default=getattr(c,x))

    return parser

def configure(parsed_args):
    configs=[global_args, peps_args, ctm_args, opt_args]
    keys=[type(c).__name__ for c in configs]
    conf_dict=dict(zip(keys, configs))

    raw_args= list(filter(lambda x: "__" not in x,dir(parsed_args)))
    grouped_args=dict(zip(keys,[[] for c in range(len(configs))]))
    for x in raw_args:
        for k in keys:
            if k in x:
                grouped_args[k].append(x)

    for k,g_args in grouped_args.items():
        for a in g_args:
            # strip prefix key+"_"
            a_noprefix=a[1+len(k):]
            setattr(conf_dict[k],a_noprefix,getattr(parsed_args,a))

def print_config():
    print(global_args)
    print(peps_args)
    print(ctm_args)
    print(opt_args)

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

    def __repr__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class PEPSARGS():
    def __init__(self):
        pass

class CTMARGS():
    r"""
    Holds configuration of the CTM algorithm. The default settings can be modified through 
    command line arguments as follows ``-CTMARGS_<variable-name> desired-value``

    :ivar ctm_max_iter: maximum iterations of directional CTM algorithm. Default: ``50``
    :vartype ctm_max_iter: int
    :ivar ctm_env_init_type: default initialization method for ENV objects. Default: ``'CTMRG'``
    :vartype ctm_env_init_type: str
    :ivar ctm_conv_tol: threshold for convergence of CTM algorithm. Default: ``'1.0e-10'``
    :vartype ctm_conv_tol: float
    :ivar projector_method: method used to construct projectors which facilitate truncation
                            of environment bond dimension :math:`\chi` within CTM algorithm

                                * 4X4: Projectors are built from two halfs of 4x4 tensor
                                  network
                                * 4X2: Projectors are built from two enlarged corners (2x2)
                                  making up a 4x2 (or 2x4) tensor network

                            Default: ``'4X4'``  
    :vartype projector_method: str
    :ivar projector_svd_method: singular value decomposition algorithm used in the construction
                                of the projectors:

                                    * GESDD: using pytorch wrapper for LAPACK's gesdd
                                    * RSVD: randomized SVD

                                Default: ``'GESDD'``
    :vartype projector_svd_method: str
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
    :ivar fwd_checkpoint_c2x2: recompute forward pass of enlarged corner functions (c2x2_*) during 
                               backward pass within optimization to save memory. Default: ``False``
    :vartype fwd_checkpoint_c2x2: bool
    :ivar fwd_checkpoint_halves: recompute forward pass of halves functions (halves_of_4x4_*) during 
                                 backward pass within optimization to save memory. Default: ``False``
    :vartype fwd_checkpoint_halves: bool
    :ivar fwd_checkpoint_projectors: recompute forward pass of projector construction (except SVD) during 
                                     backward pass within optimization to save memory. Default: ``False``
    :vartype fwd_checkpoint_projectors: bool
    :ivar fwd_checkpoint_absorb: recompute forward pass of absorp and truncate functions (absorb_truncate_*) 
                                 during backward pass within optimization to save memory. Default: ``False``
    :vartype fwd_checkpoint_absorb: bool
    """
    def __init__(self):
        self.ctm_max_iter = 50
        self.ctm_env_init_type = 'CTMRG'
        self.ctm_conv_tol = 1.0e-8
        self.projector_method = '4X4'
        self.projector_svd_method = 'GESDD'
        self.projector_svd_reltol = 1.0e-8
        self.ctm_move_sequence = [(0,-1), (-1,0), (0,1), (1,0)]
        self.verbosity_initialization = 0
        self.verbosity_ctm_convergence = 0
        self.verbosity_projectors = 0
        self.verbosity_ctm_move = 0
        self.fwd_checkpoint_c2x2 = False
        self.fwd_checkpoint_halves = False
        self.fwd_checkpoint_projectors = False
        self.fwd_checkpoint_absorb = False

    def __repr__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class OPTARGS():
    r"""
    Holds configuration of the optimization. The default settings can be modified through 
    command line arguments as follows ``-OPTARGS_<variable-name> desired-value``

    :ivar opt_ctm_reinit: reinitialize environment from scratch within every loss 
                          function evaluation. Default: ``True``
    :vartype opt_ctm_reinit: bool
    :ivar lr: initial learning rate. Default: ``1.0``
    :vartype lr: float
    :ivar tolerance_grad: stopping criterion wrt. norm of the gradient (which norm ? See 
                          ``torch.optim.LPBFG``). Default: ``1.0e-5``
    :vartype tolerance_grad: float
    :ivar tolerance_change: stopping criterion wrt. change of the loss function. 
                            Default: ``1.0e-9``
    :vartype tolerance_change: float
    :ivar max_iter_per_epoch: maximum number of optimizer iterations per epoch. Default: ``1``
    :vartype max_iter_per_epoch: int
    :ivar verbosity_opt_epoch: verbosity within optimization epoch. Default: ``1``
    :vartype verbosity_opt_epoch: int
    :ivar opt_logging: turns on recording of additional data from optimization, such as
                       CTM convergence, timings, gradients, etc. The information 
                       is logged in file ``{out_prefix}_log.json``. Default: ``True``
    :vartype opt_logging: bool
    """
    def __init__(self):
        self.opt_ctm_reinit = True
        self.lr = 1.0
        self.tolerance_grad = 1e-5
        self.tolerance_change = 1e-9
        self.history_size = 100
        self.max_iter_per_epoch = 1
        self.verbosity_opt_epoch = 1
        self.opt_logging = True

    def __repr__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

global_args= GLOBALARGS()
peps_args= PEPSARGS()
ctm_args= CTMARGS()
opt_args= OPTARGS()