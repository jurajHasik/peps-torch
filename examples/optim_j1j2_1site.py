import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import j1j2
from groups.pg import *
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = j1j2.J1J2(j1=args.j1, j2=args.j2)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    def lattice_to_site(coord):
        return (0, 0)

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        state= IPEPS(dict(), lX=1, lY=1, vertexToSite=lattice_to_site)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)
        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        sites = {(0,0): A}
        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    # 2) select the "energy" function
    energy_f=model.energy_2x2_1site_BP

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            e_curr = energy_f(state, env)
            history.append(e_curr.item())

            if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
                or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
        return False, history

    # 3) choose C4v irrep (or their mix)
    def symmetrize(state):
        A= state.site((0,0))
        A_symm= make_c4v_symm_A1(A) 
        # A_symm= make_c4v_symm_A2(A) \
        #     + make_c4v_symm_B2(A) + make_c4v_symm_B2(A)
        symm_state= IPEPS({(0,0): A_symm}, vertexToSite=state.vertexToSite)
        return symm_state

    symm_state= symmetrize(state)
    ctm_env= ENV(args.chi, symm_state)
    init_env(symm_state, ctm_env)

    ctm_env, *ctm_log= ctmrg.run(symm_state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(symm_state, ctm_env)
    obs_values, obs_labels = model.eval_obs(symm_state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        symm_state= symmetrize(state)

        # possibly re-initialize the environment
        if cfg.opt_args.opt_ctm_reinit:
            init_env(symm_state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(symm_state, ctm_env_in, conv_check=ctmrg_conv_energy)
        loss = energy_f(symm_state, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    def obs_fn(state, ctm_env, opt_context):
        symm_state= symmetrize(state)
        epoch= len(opt_context["loss_history"]["loss"]) 
        loss= opt_context["loss_history"]["loss"][-1]
        obs_values, obs_labels = model.eval_obs(symm_state,ctm_env)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]+\
            [f"{torch.max(torch.abs(symm_state.site((0,0))))}"]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    symm_state= symmetrize(state)
    ctm_env = ENV(args.chi, symm_state)
    init_env(symm_state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(symm_state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(symm_state,ctm_env)
    obs_values, obs_labels = model.eval_obs(symm_state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3

    # basic tests
    def test_opt_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        main()