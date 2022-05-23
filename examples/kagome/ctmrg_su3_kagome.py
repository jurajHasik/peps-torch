import os
import context
import torch
import argparse
import config as cfg
from ipeps.ipess_kagome import *
from ipeps.ipeps_kagome import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import su3_kagome
import unittest
import numpy as np

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
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
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=8, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    param_j = np.round(np.cos(np.pi*args.phi), decimals=12)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=12)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=12)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    model = su3_kagome.KAGOME_SU3(phys_dim=3, j=param_j, k=param_k, h=param_h)

    # initialize the ipess/ipeps
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
            elif args.ansatz in ["IPESS_PG","A_1,B", "A_2,B"]:
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
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                state= IPESS_KAGOME_PG(T_u, B_c, T_d, T_d=T_d, B_a=B_a, B_b=B_b,\
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
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
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

    # 2) select the "energy" function
    # we want to use single triangle energy evaluation for CTM
    # converge as its much cheaper than considering up triangles
    # which require 2x2 subsystem
    energy_f_down_t_1x1subsystem= model.energy_down_t_1x1subsystem
    energy_f= model.energy_triangles_2x2subsystem

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history = []
            e_curr = energy_f_down_t_1x1subsystem(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()] + obs_values)
            print(", ".join([f"{len(history)}", f"{e_curr}"] + [f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0] - history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    # 3) compute and print observables from initial guess of the environment
    e_down, e_up= energy_f(state, ctm_env_init)
    e_curr0= (e_down+e_up)/3
    obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)
    print(", ".join(["epoch", "energy_down"] + obs_labels))
    print(", ".join([f"{-1}", f"{e_curr0}"] + [f"{v}" for v in obs_values0]))

    # 4) converge CTM
    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # 5) evaluate final observables
    e_down, e_up = energy_f(state, ctm_env_init)
    e_final= (e_down+e_up)/3
    obs_values, obs_labels = model.eval_obs(state, ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_final}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # 6) compute and print additional observables
    obs = dict()
    obs["energy_dn"] = e_down
    obs["energy_up"] = e_up
    c1 = model.eval_C1(state, ctm_env_init)
    for label, value in c1.items():
        obs[label] = value
    c2s = model.eval_C2(state, ctm_env_init)
    for label, value in c2s.items():
        obs[label] = value
    print("\n")
    print(obs)

    # corrSS = model.eval_corrf_SS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(1,0)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSS = model.eval_corrf_SS((0, 0), (0, 1), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(0,1)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSSSS = model.eval_corrf_SSSS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSSSS[(0,0),(1,0)] r " + " ".join([label for label in corrSSSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSSSS[label][i]}" for label in corrSSSS.keys()]))
    #

    # environment diagnostics
    print("\n")
    for c_loc, c_ten in ctm_env_init.C.items():
        u, s, v = torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")
    
    # transfer operator spectrum
    site_dir_list = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 1), (1, 0)), ((1, 1), (0, 1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l = transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i, 0]} {l[i, 1]}")


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg_IPESS_D3_AKLT(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-aklt"
    ANSATZE= [("IPESS","AKLT_SU3_KAGOME_D3_IPESS_state.json"),\
        ("IPESS_PG","AKLT_SU3_KAGOME_D3_IPESS_PG_state.json"),\
        ("A_2,B","AKLT_SU3_KAGOME_D3_A2B_state.json")]

    def setUp(self):
        args.phi= 0.5
        args.bond_dim=3
        args.chi=18
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_ipess_ansatze_d3_aklt(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                args.ansatz= ansatz[0]
                args.instate= self.DIR_PATH+"/../../test-input/"+ansatz[1]
                # args.sym_up_dn= ansatz[1]
                # args.sym_bond_S= ansatz[2]
                # args.out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}_"\
                #     +("T" if ansatz[1] else "F")+("T" if ansatz[2] else "F")
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}"
                
                # i) run ctmrg and compute observables
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                final_obs=None
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs

                # compare with the reference
                ref_data="""
                -0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                """
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [complex(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log"]:
                if os.path.isfile(f): os.remove(f)