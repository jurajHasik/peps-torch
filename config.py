import torch
import argparse
import logging

def _torch_version_check(version):
    # for version="X.Y.Z" checks if current version is higher or equal to X.Y
    assert version.count('.')==2 and version.replace('.','').isdigit(),"Invalid version string"
    try:
        import pkg_resources
        return pkg_resources.parse_version(torch.__version__) >= pkg_resources.parse_version(version)
    except ModuleNotFoundError:
        try:
            from packaging import version
            return version.parse(torch.__version__) >= version.parse(version)
        except ModuleNotFoundError:
            tokens= torch.__version__.split('.')
            tokens_v= version.split('.')
            return int(tokens[0]) > int(tokens_v[0]) or \
                (int(tokens[0])==int(tokens_v[0]) and int(tokens[1]) >= int(tokens_v[1])) 
    return True

def get_args_parser():
    parser = argparse.ArgumentParser(description='',allow_abbrev=False)
    parser.add_argument("--omp_cores", type=int, default=1,help="number of OpenMP cores")
    parser.add_argument("--instate", default=None, help="Input state JSON")
    parser.add_argument("--instate_noise", type=float, default=0., help="magnitude of noise added to the trial \"instate\"")
    parser.add_argument("--ipeps_init_type", default="RANDOM", help="initialization of the trial iPEPS state")
    parser.add_argument("--out_prefix", default="output", help="Output files prefix")
    parser.add_argument("--bond_dim", type=int, default=1, help="iPEPS auxiliary bond dimension")
    parser.add_argument("--chi", type=int, default=20, help="environment bond dimension")
    parser.add_argument("--opt_max_iter", type=int, default=100, help="maximal number of epochs")
    parser.add_argument("--opt_resume", type=str, default=None, help="file with checkpoint to resume")
    parser.add_argument("--opt_resume_override_params", action='store_true', help="override optimizer parameters stored in checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")

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
                    parser.add_argument("--"+group_prefix+x, action='store_true')
                # default is True
                else:
                    parser.add_argument("--"+group_prefix+"no_"+x, action='store_false', dest=group_prefix+x)
            else:
                parser.add_argument("--"+group_prefix+x, type=type(getattr(c,x)), default=getattr(c,x))

    return parser

def configure(parsed_args):
    configs=[global_args, peps_args, ctm_args, opt_args]
    keys=[type(c).__name__ for c in configs]
    conf_dict=dict(zip(keys, configs))

    raw_args= list(filter(lambda x: "__" not in x and not callable(getattr(parsed_args,x)),\
        dir(parsed_args)))
    grouped_args=dict(zip(keys,[[] for c in range(len(configs))]))
    nogroup_args=dict()
    def _search_keys(x):
        for k in keys:
            if k in x: return k
        return None
    for x in raw_args:
        ind= _search_keys(x)
        if ind is not None:
            grouped_args[ind].append(x)
        else:
            nogroup_args[x]=getattr(parsed_args,x)

    # set prefix args
    for k,g_args in grouped_args.items():
        for a in g_args:
            # strip prefix key+"_"
            a_noprefix=a[1+len(k):]
            setattr(conf_dict[k],a_noprefix,getattr(parsed_args,a))
    # set main args
    for name,val in nogroup_args.items():
        setattr(main_args,name,val)

    # custom handling
    if global_args.dtype=="float64":
        global_args.torch_dtype= torch.float64
    elif global_args.dtype=="complex128":
        global_args.torch_dtype= torch.complex128
    else:
        raise NotImplementedError(f"Unsupported dtype {global_args.dtype}")

    # validate
    # if ctm_args.step_core_gpu:
    #     assert global_args.gpu and torch.cuda.device(global_args.gpu), "CTMARGS_step_core_gpu"\
    #         +" resquested without providing valid GLOBALARGS_gpu"

    # set up logger
    logging.basicConfig(filename=main_args.out_prefix+".log", filemode='w', level=logging.INFO)

def print_config():
    import numpy as np
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except ImportError:
        print(f"SciPy: N/A")
    try:
        import subprocess
        import pathlib
        root_dir= pathlib.Path(__file__).parent.resolve()
        ret= subprocess.run(f"cd {root_dir} ; git rev-parse --short HEAD",\
            stdout=subprocess.PIPE, shell=True, check=True, text=True)
        print(f"peps-torch git ref: {ret.stdout.rstrip()}")
    except subprocess.CalledProcessError as e:
        print(f"peps-torch git ref: N/A")
    try:
        import subprocess
        import pathlib
        root_dir= pathlib.Path(__file__).parent.resolve()
        ret= subprocess.run(f"cd {root_dir}/yast ; git rev-parse --short HEAD",\
            stdout=subprocess.PIPE, shell=True, check=True,  text=True)
        print(f"yast git ref: {ret.stdout.rstrip()}")
    except subprocess.CalledProcessError as e:
        print(f"yast git ref: N/A")
    print(main_args)
    print(global_args)
    print(peps_args)
    print(ctm_args)
    print(opt_args)

class MAINARGS():
    r"""
    Main simulation options. The default settings can be modified through 
    command line arguments as follows ``--<option-name> desired-value``

    :ivar omp_cores: number of OpenMP cores. Default: ``1``
    :vartype omp_cores: int:
    :ivar instate: input state file. Default: ``None``
    :vartype instate: str or Path
    :ivar instate_noise: magnitude of noise applied to the input state, if any. Default: ``0.0``
    :vartype instate_noise: float
    :ivar ipeps_init_type: initialization of the trial iPEPS state, if no ``instate`` is provided. Default: ``RANDOM``
    :vartype ipeps_init_type: str
    :ivar out_prefix: output file prefix. Default: ``output``
    :vartype out_prefix: str
    :ivar bond_dim: iPEPS auxiliary bond dimension. Default: ``1``
    :vartype bond_dim: int
    :ivar chi: environment bond dimension. Default: ``20``
    :vartype chi: int
    :ivar opt_max_iter: maximal number of optimization steps. Default: ``100``
    :vartype opt_max_iter: int
    :ivar opt_resume: resume from checkpoint file. Default: ``None``
    :vartype opt_resume: str or Path
    :ivar opt_resume_override_params: override optimizer parameters stored in checkpoint. Default: ``False``
    :vartype opt_resume_override_params: bool
    :ivar seed: PRNG seed. Default: ``0``
    :vartype seed: int
    """
    def __init__(self):
        pass

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class GLOBALARGS():
    r"""
    Holds global configuration options. The default settings can be modified through 
    command line arguments as follows ``--GLOBALARGS_<variable-name> desired-value``

    :ivar dtype: data type of all torch.tensor. Default: ``float64``
    :vartype dtype: torch.dtype
    :ivar device: device on which all the torch.tensors are stored. Default: ``'cpu'``
    :vartype device: str
    :ivar offload_to_gpu: gpu used for optional acceleration. It might be desirable to store the model 
               and all the intermediates of CTM on CPU and compute only the core parts of the expensive
               CTM step on GPU. Default: ``'None'``
    :vartype offload_to_gpu: str
    """
    def __init__(self):
        self.tensor_io_format= "legacy"
        self.dtype= "float64"
        self.torch_dtype= torch.float64
        self.device= 'cpu'
        self.offload_to_gpu= 'None'

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class PEPSARGS():
    def __init__(self):
        self.build_dl= True
        self.build_dl_open= False
        self.quasi_gauge_max_iter= 10**6
        self.quasi_gauge_tol= 1.0e-8

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class CTMARGS():
    r"""
    Holds configuration of the CTM algorithm. The default settings can be modified through 
    command line arguments as follows ``--CTMARGS_<variable-name> desired-value``

    :ivar ctm_max_iter: maximum iterations of directional CTM algorithm. Default: ``50``
    :vartype ctm_max_iter: int
    :ivar ctm_env_init_type: default initialization method for ENV objects. Default: ``'CTMRG'``
    :vartype ctm_env_init_type: str
    :ivar ctm_conv_tol: threshold for convergence of CTM algorithm. Default: ``'1.0e-10'``
    :vartype ctm_conv_tol: float
    :ivar conv_check_cpu: execute CTM convergence check on cpu (if applicable). Default: ``False`` 
    :ivar ctm_absorb_normalization: normalization to use for new corner/T tensors. Either ``'fro'`` for usual
                                    L2 norm or ``'inf'`` for L-\infty norm. Default: ``'fro'``.  
    :vartype ctm_absorb_normalization: str
    :vartype conv_check_cpu: bool
    :ivar projector_method: method used to construct projectors which facilitate truncation
                            of environment bond dimension :math:`\chi` within CTM algorithm

                                * 4X4: Projectors are built from two halfs of 4x4 tensor
                                  network
                                * 4X2: Projectors are built from two enlarged corners (2x2)
                                  making up a 4x2 (or 2x4) tensor network

                            Default: ``'4X4'``  
    :vartype projector_method: str
    :ivar projector_svd_method: singular/eigen value decomposition algorithm used in the construction
                                of the projectors:

                                    * ``'GESDD'``: pytorch wrapper of LAPACK's gesdd
                                    * ``'RSVD'``: randomized SVD
                                    * ``'SYMEIG'``: pytorch wrapper of LAPACK's dsyev for symmetric matrices
                                    * ``'SYMARP'``: scipy wrapper of ARPACK's dsaupd for symmetric matrices
                                    * ``'ARP'``: scipy wrapper of ARPACK's svds for general matrices

                                Default: ``'SYMEIG'`` for c4v-symmetric CTM, otherwise ``'GESDD'``
    :vartype projector_svd_method: str
    :ivar projector_svd_reltol: relative threshold on the magnitude of the smallest elements of 
                                singular value spectrum used in the construction of projectors. 
                                Default: ``1.0e-8``
    :vartype projector_svd_reltol: float
    :ivar projector_svd_reltol_block: (relevant only for decompositions of blocks-sparse tensors) 
                                relative threshold on the magnitude of the smallest elements of 
                                singular value spectrum per block used in the construction of projectors.
                                Default: ``0.0``
    :vartype projector_svd_reltol_block: float
    :ivar projector_eps_multiplet: threshold for defining boundary of the multiplets
    :vartype projector_eps_multiplet: float
    :ivar projector_multiplet_abstol: absolute threshold for spectral values to be considered in multiplets 
    :vartype projector_multiplet_abstol: float
    :ivar radomize_ctm_move_sequence: If ``True``, then ``ctm_move_sequence`` is randomized in each optimization step
    :vartype radomize_ctm_move_sequence: bool
    :ivar ctm_move_sequence: sequence of directional moves within single CTM iteration. The possible 
                             directions are encoded as tuples(int,int) 
                                
                                * up = (0,-1)
                                * left = (-1,0)
                                * down = (0,1)
                                * right = (1,0)

                             Default: ``[(0,-1), (-1,0), (0,1), (1,0)]``
    :vartype ctm_move_sequence: list[tuple(int,int)]
    :ivar ctm_force_dl: precompute and use on-site double-layer tensors in CTMRG 
    :vartype ctm_force_dl: bool
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
    :ivar fwd_checkpoint_move: recompute forward pass of whole ``ctm_MOVE`` during backward pass. Default: ``False``
    :vartype fwd_checkpoint_move: bool

    FPCM related options

    :ivar fpcm_init_iter: minimal number of CTM steps before FPCM acceleration step is attempted.
                          Default: ``1``
    :vartype fpcm_init_iter: int
    :ivar fpcm_freq: frequency of FPCM steps per CTM steps. Default: ``-1``
    :vartype fpcm_freq: int
    :ivar fpcm_isogauge_tol: tolerance on gauging the uniform MPS built from half-row/-column tensor T.
                             Default: ``1.0e-14``.
    :vartype fpcm_isogauge_tol: float
    :ivar fpcm_fpt_tol: tolerance on convergence within FPCM step. Default: ``1.0e-8``
    :vartype fpcm_fpt_tol: float

    Logging and Debugging options

    :ivar ctm_logging: log debug statements into log file. Default: ``False`` 
    :vartype ctm_loggging: bool
    :ivar verbosity_initialization: verbosity of initialization method for ENV objects. Default: ``0``
    :vartype verbosity_initialization: int
    :ivar verbosity_ctm_convergence: verbosity of evaluation of CTM convergence criterion. Default: ``0``
    :vartype verbosity_ctm_convergence: int
    :ivar verbosity_projectors: verbosity of projector construction. Default: ``0``
    :vartype verbosity_projectors: int
    :ivar verbosity_ctm_move: verbosity of directional CTM moves. Default: ``0``
    :vartype verbosity_ctm_move: int
    :ivar verbosity_rdm: verbosity of reduced density matrix routines. Default: ``0``
    :vartype verbosity_rdm: int
    """
    def __init__(self):
        self.ctm_max_iter= 50
        self.ctm_env_init_type= 'CTMRG'
        self.ctm_conv_tol= 1.0e-8
        self.ctm_absorb_normalization= 'inf'
        self.fpcm_init_iter=1
        self.fpcm_freq= -1
        self.fpcm_isogauge_tol= 1.0e-14
        self.fpcm_fpt_tol= 1.0e-8
        self.conv_check_cpu = False
        self.projector_method = '4X4'
        self.projector_svd_method = 'DEFAULT'
        self.projector_svd_reltol = 1.0e-8
        self.projector_svd_reltol_block = 0.0
        self.projector_eps_multiplet = 1.0e-8
        self.projector_multiplet_abstol = 1.0e-14
        self.ad_decomp_reg= 1.0e-12
        self.ctm_move_sequence = [(0,-1), (-1,0), (0,1), (1,0)]
        self.randomize_ctm_move_sequence = False
        self.ctm_force_dl = False
        self.ctm_logging = False
        self.verbosity_initialization = 0
        self.verbosity_ctm_convergence = 0
        self.verbosity_projectors = 0
        self.verbosity_ctm_move = 0
        self.verbosity_fpcm_move = 0
        self.verbosity_rdm = 0
        self.fwd_checkpoint_c2x2 = False
        self.fwd_checkpoint_halves = False
        self.fwd_checkpoint_projectors = False
        self.fwd_checkpoint_absorb = False
        self.fwd_checkpoint_move = False
        self.fwd_checkpoint_loop_rdm = False

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class OPTARGS():
    r"""
    Holds configuration of the optimization. The default settings can be modified through 
    command line arguments as follows ``--OPTARGS_<variable-name> desired-value``

    General options

    :ivar opt_ctm_reinit: reinitialize environment from scratch within every loss 
                          function evaluation. Default: ``True``
    :vartype opt_ctm_reinit: bool
    :ivar lr: initial learning rate. Default: ``1.0``
    :vartype lr: float
    :ivar line_search: line search algorithm to use. L-BFGS supports ``'strong_wolfe'`` 
        and ``'backtracking'``. SGD supports just ``'backtracking'``. Default: ``None``.
    :vartype line_search: str
    :ivar line_search_ctm_reinit: recompute environment from scratch at each step within
        line search algorithm. Default: ``True``.
    :vartype line_search_ctm_reinit: bool
    :ivar line_search_svd_method:   eigen decompostion method to use within line search
        environment computation. See options in :class:`config.CTMARGS`. Default: ``'DEFAULT'`` which
        depends on the particular CTM algorithm.
    :vartype line_search_svd_method: str
    
    L-BFGS related options

    :ivar tolerance_grad: stopping criterion wrt. norm of the gradient (which norm ? See 
                          ``torch.optim.LBFGS``). Default: ``1.0e-5``
    :vartype tolerance_grad: float
    :ivar tolerance_change: stopping criterion wrt. change of the loss function. 
                            Default: ``1.0e-9``
    :vartype tolerance_change: float
    :ivar max_iter_per_epoch: maximum number of optimizer iterations per epoch. Default: ``1``
    :vartype max_iter_per_epoch: int
    :ivar history_size: number past of directions used to approximate inverse Hessian.
        Default: ``100``.
    :vartype history_size: int 
    
    SGD related options

    :ivar momentum: momentum used in the SGD step
    :vartype momentum: float
    :ivar dampening: dampening used in the SGD step
    :vartype dampening: float

    Gradients through finite differences

    :ivar fd_eps: magnitude of displacement when computing the forward difference 
        :math:`E(x_0 + \textrm{fd_eps})-E(x_0)/\textrm{fd_eps}`. Default: ``1.0e-4``
    :vartype fd_eps: float
    :ivar fd_ctm_reinit: recompute environment from scratch after applying the displacement.
        Default: ``True`` 

    Logging

    :ivar opt_logging: turns on recording of additional data from optimization, such as
                       CTM convergence, timings, gradients, etc. The information 
                       is logged in file ``{out_prefix}_log.json``. Default: ``True``
    :vartype opt_logging: bool
    :ivar opt_log_grad: log values of gradient. Default: ``False``
    :vartype opt_log_grad: bool
    :ivar verbosity_opt_epoch: verbosity within optimization epoch. Default: ``1``
    :vartype verbosity_opt_epoch: int
    """
    def __init__(self):
        self.lr= 1.0
        self.momentum= 0.
        self.dampening= 0.
        self.tolerance_grad= 1e-5
        self.tolerance_change= 1e-9
        self.opt_ctm_reinit= True
        self.env_sens_scale= 10.0
        self.line_search= "default"
        self.line_search_ctm_reinit= True
        self.line_search_svd_method= 'DEFAULT'
        self.line_search_tol= 1.0e-8
        self.fd_eps= 1.0e-4
        self.fd_ctm_reinit= True
        self.history_size= 100
        self.max_iter_per_epoch= 1
        self.verbosity_opt_epoch= 1
        self.opt_logging= True
        self.opt_log_grad= False

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

main_args= MAINARGS()
global_args= GLOBALARGS()
peps_args= PEPSARGS()
ctm_args= CTMARGS()
opt_args= OPTARGS()
