import context
import torch
import argparse
import config as cfg
from groups.pg import make_d2_symm, make_d2_antisymm
from ipeps_d2 import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import coupledLadders
from optim.ad_optim_d2 import optimize_state
import unittest
from ctm.generic import transferops
import json

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-alpha", type=float, default=0., help="inter-ladder coupling")
parser.add_argument("-top_freq", type=int, default=1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("-top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    
    model = coupledLadders.COUPLEDLADDERS_D2_BIPARTITE(alpha=args.alpha)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    

    if args.instate!=None:
        state = read_ipeps_d2(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        add_random_noise(state, args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype,device=cfg.global_args.device)
        # A= make_d2_symm(A)
        A= A/torch.max(torch.abs(A))

        sites = {(0,0): A}   
        state = IPEPS_D2SYM(sites)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    state.sites= state.build_onsite_tensors()
    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            e_curr = model.energy_2x1_1x2(state, env)
            history.append(e_curr.item())

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    loss = model.energy_2x1_1x2(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_args=cfg.opt_args):
        # symmetrize and normalize
        # with torch.no_grad():
        for c in state.parent_tensors.keys():
            state.parent_tensors[c]= make_d2_symm(state.parent_tensors[c])
            # state.parent_tensors[c]+= state.parent_tensors[c].permute(0,1,4,3,2)
            # state.parent_tensors[c]*= 1.0/torch.max(torch.abs(state.parent_tensors[c]))
            state.parent_tensors[c]= state.parent_tensors[c]/torch.max(torch.abs(state.parent_tensors[c]))

        # build on-site tensors
        state.sites= state.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy)
        loss = model.energy_2x1_1x2(state, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    # def top_spec_check(state,ctm_env,epoch,*other):
    #     with torch.no_grad():
    #         def _to_json(l):
    #             re=[l[i,0].item() for i in range(l.size()[0])]
    #             im=[l[i,1].item() for i in range(l.size()[0])]
    #             return dict({"re": re, "im": im})

    #         if epoch%args.top_freq==0:
    #             coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
    #             for c,d in coord_dir_pairs:
    #                 # transfer operator spectrum
    #                 print(f"TOP spectrum(T)[{c},{d}] ",end="")
    #                 l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
    #                 print("TOP "+json.dumps(_to_json(l)))

    # # optimize
    optimize_state(state, ctm_env, loss_fn, model, args)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_d2(outputstatefile)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = model.energy_2x1_1x2(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.alpha=1.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=2

    # basic tests
    def test_opt_GESDD(self):
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        main()

    def test_opt_fwd_checkpoint_move(self):
        args.fwd_checkpoint_move= True
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_gpu_fwd_checkpoint_move(self):
        args.GLOBALARGS_device="cuda:0"
        args.fwd_checkpoint_move= True
        main()