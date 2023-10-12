import os
import context
import torch
import argparse
import config as cfg
from ipeps.ipess_kagome import *
from ipeps.ipeps_kagome import *
from ctm.generic.env import *
from ctm.generic import ctmrg
# from ctm.generic import ctmrg_sl as ctmrg
from models import su3_kagome
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
import numpy as np

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional model-dependent arguments

# parser.add_argument("--j", type=float, default=0., help="coupling for 2-site exchange terms")
# parser.add_argument("--k", type=float, default=-1., help="coupling for 3-site ring exchange terms")
# parser.add_argument("--h", type=float, default=0., help="coupling for chiral terms")
parser.add_argument("--phi", type=float, default=0.5, help="parametrization between "\
    +"2-site exchange and 3-site permutation terms in the units of pi")
parser.add_argument("--theta", type=float, default=0., help="parametrization between "\
    +"normal and chiral terms in the units of pi")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", "IPESS_PG", "A_1,B", "A_2,B"])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bonds", action='store_false', dest='sym_bond_S',\
    help="same bond tensors for sites A,B and C")
parser.add_argument("--legacy_instate", action='store_true', dest='legacy_instate',\
    help="legacy format of states")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # 0) initialize model
    param_j = np.round(np.cos(np.pi*args.phi), decimals=12)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=12)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=12)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    model = su3_kagome.KAGOME_SU3(phys_dim=3, j=param_j, k=param_k, h=param_h)
    
    # 1) initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_1,B","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_1,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A1_B
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        
        if args.instate!=None:
            if args.ansatz=="IPESS":
                if not args.legacy_instate:
                    state= read_ipess_kagome_generic(args.instate)
                else:
                    state= read_ipess_kagome_generic_legacy(args.instate, ansatz=args.ansatz)
            elif args.ansatz in ["IPESS_PG","A_1,B","A_2,B"]:
                if not args.legacy_instate:
                    state= read_ipess_kagome_pg(args.instate)
                else:
                    state= read_ipess_kagome_generic_legacy(args.instate, ansatz=args.ansatz)

            # possibly symmetrize by PG
            if ansatz_pgs!=None:
                if type(state)==IPESS_KAGOME_GENERIC:
                    state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
                        SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
                elif type(state)==IPESS_KAGOME_PG:
                    if state.pgs==None or state.pgs==dict():
                        state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
                            SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
                    elif state.pgs==ansatz_pgs:
                        # nothing to do here
                        pass
                    elif state.pgs!=ansatz_pgs:
                        raise RuntimeError("instate has incompatible PG symmetry with "+args.ansatz)

            if args.bond_dim > state.get_aux_bond_dims():
                # extend the auxiliary dimensions
                state= state.extend_bond_dim(args.bond_dim)
            state.add_noise(args.instate_noise)
        elif args.opt_resume is not None:
            T_u= torch.zeros(args.bond_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            T_d= torch.zeros(args.bond_dim, args.bond_dim,\
                args.bond_dim, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_c= torch.zeros(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_a= torch.zeros(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_b= torch.zeros(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            if args.ansatz in ["IPESS_PG", "A_1,B", "A_2,B"]:
                state= IPESS_KAGOME_PG(T_u, B_c, T_d=T_d, B_a=B_a, B_b=B_b,\
                    SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            elif args.ansatz in ["IPESS"]:
                state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                    'B_b': B_b, 'B_c': B_c})
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=='RANDOM':
            bond_dim = args.bond_dim
            T_u= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            T_d= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_c= torch.rand(3, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_a= torch.rand(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_b= torch.rand(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            if args.ansatz in ["IPESS_PG", "A_1,B", "A_2,B"]:
                state = IPESS_KAGOME_PG(T_u, B_c, T_d=T_d, B_a=B_a, B_b=B_b,\
                    SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs,\
                    pg_symmetrize=True)
            elif args.ansatz in ["IPESS"]:
                state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                    'B_b': B_b, 'B_c': B_c})
    
    elif args.ansatz in ["IPEPS"]:    
        ansatz_pgs=None
        if args.instate!=None:
            state= read_ipeps_kagome(args.instate)

            if args.bond_dim > max(state.get_aux_bond_dims()):
                # extend the auxiliary dimensions
                state= state.extend_bond_dim(args.bond_dim)
            state.add_noise(args.instate_noise)
        elif args.opt_resume is not None:
            state= IPEPS_KAGOME(dict(), lX=1, lY=1)
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=='RANDOM':
            bond_dim = args.bond_dim
            A = torch.rand((model.phys_dim**3, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device) - 0.5
            A = A/torch.max(torch.abs(A))
            state= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)
    else:
        raise ValueError("Missing ansatz specification --ansatz "\
            +str(args.ansatz)+" is not supported")

    if not state.dtype == model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model = su3_kagome.KAGOME_SU3(phys_dim=3, j=param_j, k=param_k, h=param_h)

    print(state)
    # we want to use single triangle energy evaluation for CTM
    # convergence as its much cheaper than considering up triangles
    # which require 2x2 subsystem
    energy_f_down_t_1x1subsystem= model.energy_down_t_1x1subsystem
    energy_f= model.energy_per_site_2x2subsystem

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_curr = energy_f_down_t_1x1subsystem(state, env, force_cpu=args.force_cpu,\
            fail_on_check=False, warn_on_check=False)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    # 2) compute initial observables
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs_2x2subsystem(state, ctm_env, force_cpu=args.force_cpu)
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{loss0}"] + [f"{v}" for v in obs_values]))

    # 3) define loss function
    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # build on-site tensors
        if args.ansatz in ["IPESS", "IPESS_PG", "A_1,B", "A_2,B"]:
            if args.ansatz in ["IPESS_PG", "A_1,B", "A_2,B"]:
                # explicit rebuild of on-site tensors
                sym_state= to_PG_symmetric(state)
            else:
                sym_state= state
            # include normalization of new on-site tensor
            sym_state.sites= sym_state.build_onsite_tensors()
        else:
            A= state.sites[(0,0)]
            A= A/A.abs().max()
            sym_state= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(sym_state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(sym_state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        loss= energy_f(sym_state, ctm_env_out, force_cpu=args.force_cpu)

        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if args.ansatz in ["IPESS_PG","A_1,B","A_2,B"]:
            state_sym= to_PG_symmetric(state)
        else:
            state_sym= state
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
        obs_values, obs_labels = model.eval_obs_2x2subsystem(state_sym, ctm_env,\
            force_cpu=args.force_cpu, warn_on_check=False)
        print(", ".join([f"{epoch}", f"{loss}"] + [f"{v}" for v in obs_values]))
        log.info("Norm(sites): " + ", ".join([f"{t.norm()}" for c, t in state.sites.items()]))

    # 4) optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # 5) compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    if args.ansatz in ["IPESS"]:
        state= read_ipess_kagome_generic(outputstatefile)
    elif args.ansatz in ["IPESS_PG","A_1,B","A_2,B"]:
        state= read_ipess_kagome_pg(outputstatefile)
    elif args.ansatz in ["IPEPS"]:
        state= read_ipeps_kagome(outputstatefile)
    else:
        raise ValueError("Missing ansatz specification --ansatz "\
            +str(args.ansatz)+" is not supported")
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs_2x2subsystem(state, ctm_env, force_cpu=args.force_cpu)
    print(", ".join([f"{args.opt_max_iter}", f"{loss0}"] + [f"{v}" for v in obs_values]))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestOpt(unittest.TestCase):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        args.theta=0.0
        args.phi=0.5
        args.bond_dim=3
        args.chi=18
        args.instate= None
        args.GLOBALARGS_dtype= "complex128"
        args.OPTARGS_tolerance_grad= 1.0e-8
        args.OPTARGS_tolerance_change= 1.0e-8
        # args.OPTARGS_line_search= "backtracking"
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. ARNOLDISVD is not available.")
            self.SCIPY= False

    def test_opt_AKLT_IPESS(self):
        from io import StringIO
        from unittest.mock import patch
        
        args.instate= args.instate= self.DIR_PATH+"/../../test-input/AKLT_SU3_KAGOME_D3_IPESS_state.json"
        args.ansatz= "IPESS"
        args.instate_noise= 0.1
        args.seed= 43
        args.opt_max_iter= 500
        args.out_prefix= f"TEST_IPESS_D{args.bond_dim}-chi{args.chi}_AKLT"

        # run the main simulation
        main()

        args.instate= args.out_prefix+"_state.json"
        args.instate_noise= 0
        args.opt_max_iter= 1
        with patch('sys.stdout', new = StringIO()) as fake_out:
            main()
            output= fake_out.getvalue()

        print(output)

        # test final output against expected result
        final_obs= [float(x) for x in output.splitlines()[-1].split(',')[:2]]
        final_f_vecs= [complex(x) for x in output.splitlines()[-1].split(',')[-3:]]
        final_e= final_obs[1]
        self.assertTrue(abs(final_e + 2/3.) < 1.0e-3)
        self.assertTrue(abs(final_f_vecs[0]) < 1.0e-4)
        self.assertTrue(abs(final_f_vecs[1]) < 1.0e-4)
        self.assertTrue(abs(final_f_vecs[2]) < 1.0e-4)

    def test_opt_AKLT_IPESS_PG(self):
        from io import StringIO
        from unittest.mock import patch
        
        args.instate= args.instate= self.DIR_PATH+"/../../test-input/AKLT_SU3_KAGOME_D3_IPESS_PG_state.json"
        args.ansatz= "IPESS_PG"
        args.instate_noise= 0.1
        args.seed= 43
        args.opt_max_iter= 500
        args.out_prefix= f"TEST_IPESS_PG_D{args.bond_dim}-chi{args.chi}_AKLT"

        # run the main simulation
        main()

        args.instate= args.out_prefix+"_state.json"
        args.instate_noise= 0
        args.opt_max_iter= 1
        with patch('sys.stdout', new = StringIO()) as fake_out:
            main()
            output= fake_out.getvalue()

        print(output)

        # test final output against expected result
        final_obs= [float(x) for x in output.splitlines()[-1].split(',')[:2]]
        final_f_vecs= [complex(x) for x in output.splitlines()[-1].split(',')[-3:]]
        final_e= final_obs[1]
        self.assertTrue(abs(final_e + 2/3.) < 1.0e-3)
        self.assertTrue(abs(final_f_vecs[0]) < 1.0e-4)
        self.assertTrue(abs(final_f_vecs[1]) < 1.0e-4)
        self.assertTrue(abs(final_f_vecs[2]) < 1.0e-4)

    def test_opt_AKLT_A2B(self):
        from io import StringIO
        from unittest.mock import patch
        
        args.instate= args.instate= self.DIR_PATH+"/../../test-input/AKLT_SU3_KAGOME_D3_A2B_state.json"
        args.ansatz= "A_2,B"
        args.instate_noise= 0.1
        args.seed= 321431
        args.opt_max_iter= 500
        args.out_prefix= f"TEST_A2B_D{args.bond_dim}-chi{args.chi}_AKLT"

        # run the main simulation
        main()

        args.instate= args.out_prefix+"_state.json"
        args.instate_noise= 0
        args.opt_max_iter= 1
        with patch('sys.stdout', new = StringIO()) as fake_out:
            main()
            output= fake_out.getvalue()

        print(output)

        # test final output against expected result
        final_obs= [float(x) for x in output.splitlines()[-1].split(',')[:2]]
        final_f_vecs= [complex(x) for x in output.splitlines()[-1].split(',')[-3:]]
        final_e= final_obs[1]
        self.assertTrue(abs(final_e + 2/3.) < 1.0e-7)
        self.assertTrue(abs(final_f_vecs[0]) < 1.0e-5)
        self.assertTrue(abs(final_f_vecs[1]) < 1.0e-5)
        self.assertTrue(abs(final_f_vecs[2]) < 1.0e-5)