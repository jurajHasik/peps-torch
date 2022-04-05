import os
import context
import argparse
import config as cfg
import math
import torch
import copy
from collections import OrderedDict
from ipeps.ipess_kagome import *
from ipeps.ipeps_kagome import *
from models import spin_half_kagome
from ctm.generic.env import *
from ctm.generic import ctmrg
from optim.fd_optim_lbfgs_mod import optimize_state
import su2sym.sym_ten_parser as tenSU2
import json
import unittest
import logging
import scipy.io as io
import numpy
import time

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=None, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--JD", type=float, default=0, help="two-spin DM interaction")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=complex, default=0, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
    choices=["A_2,B", "A_1,B"])
parser.add_argument("--disp_corre_len", action='store_true', dest='disp_corre_len',\
    help="display correlation length during optimization")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",\
    choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--force_cpu", action='store_true', dest='force_cpu', help="force RDM contractions on CPU")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    if not args.theta is None:
        args.jtrip= args.j1*math.sin(args.theta*math.pi)
        args.j1= args.j1*math.cos(args.theta*math.pi)
    print(f"j1={args.j1}; jD={args.JD}; j2={args.j2}; jtrip={args.jtrip}")
    model= spin_half_kagome.S_HALF_KAGOME(j1=args.j1, JD=args.JD,\
        j2=args.j2, jtrip=args.jtrip, jperm=args.jperm)

    # initialize the ipess/ipeps
    if args.ansatz in ["A_1,B","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        if args.ansatz=="A_1,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A1_B
        
        if args.instate!=None:
            state= read_ipess_kagome_pg_lc(args.instate)

            # if args.bond_dim > state.get_aux_bond_dims():
            #     # extend the auxiliary dimensions
            #     state= state.extend_bond_dim(args.bond_dim)
            state.add_noise(args.instate_noise)
        elif args.opt_resume is not None:
            if args.ansatz in ["A_1,B", "A_2,B"]:
                state= IPESS_KAGOME_PG_LC.create_from_checkpoint(args.opt_resume,\
                    pgs=ansatz_pgs)
        elif args.ipeps_init_type=='RANDOM':
            if args.bond_dim in [3,6]:
                p= 2 # phys dim
                D= args.bond_dim
                path= os.path.dirname(__file__)+"/../../su2sym/ipess_kagome/"

                tri_t_A1= tenSU2.import_sym_tensors_generic((D,D,D), "A_1",
                    path+f"D{D}_trivalent.txt", dtype=torch.float64, device=cfg.global_args.device)
                tri_t_A2= tenSU2.import_sym_tensors_generic((D,D,D), "A_2",
                    path+f"D{D}_trivalent.txt", dtype=torch.float64, device=cfg.global_args.device)
                bond_t_A= tenSU2.import_sym_tensors_generic((p,D,D), "A",
                    path+f"D{D}_bond.txt", dtype=torch.float64, device=cfg.global_args.device)
                bond_t_B= tenSU2.import_sym_tensors_generic((p,D,D), "B",
                    path+f"D{D}_bond.txt", dtype=torch.float64, device=cfg.global_args.device)
                tri_t= tri_t_A1+tri_t_A2
                bond_t= bond_t_A+bond_t_B

                if args.ansatz in ["A_1,B", "A_2,B"]:
                    state= IPESS_KAGOME_PG_LC(
                        T_u=(torch.rand( len(tri_t), dtype=torch.float64, device=cfg.global_args.device ),\
                            tri_t),\
                        B_c=(torch.rand( len(bond_t), dtype=torch.float64, device=cfg.global_args.device ),\
                            bond_t),
                    SYM_UP_DOWN=True,SYM_BOND_S=True, pgs=ansatz_pgs)
            else:
                raise RuntimeError("Bond dim "+str(args.bond_dim)+" is not supported")
            state.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing ansatz specification --ansatz "\
            +str(args.ansatz)+" is not supported")


    def energy_f(state, env, force_cpu=False, fail_on_check=False,\
        warn_on_check=True):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def energy_f_complex(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f_NoCheck(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        return e_dn

    @torch.no_grad()
    def print_corner_spectra(env):
        spectra = []
        for c_loc,c_ten in env.C.items():
            u,s,v= torch.svd(c_ten, compute_uv=False)
            if c_loc[1] == (-1, -1):
                label = 'LU'
            if c_loc[1] == (-1, 1):
                label = 'LD'
            if c_loc[1] == (1, -1):
                label = 'RU'
            if c_loc[1] == (1, 1):
                label = 'RD'
            spectra.append([label, s])
        return spectra

    if args.CTM_check=="Energy":
        @torch.no_grad()
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            e_curr = energy_f_complex(state, env, force_cpu=ctm_args.conv_check_cpu)
            history.append(e_curr.item())
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="Partial_energy":
        @torch.no_grad()
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            if len(history)>8:
                e_curr = dn_energy_f_NoCheck(state, env, force_cpu=ctm_args.conv_check_cpu)
                history.append(e_curr.item())
            else:
                history.append(len(history)+1)
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol*2) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="SingularValue":
        @torch.no_grad()
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history_spec = []
                history_ite=1
                history=[history_ite, history_spec]
            spect_new=print_corner_spectra(env)
            spec1_new=spect_new[0][1]
            spec1_new=spec1_new/spec1_new[0]
            spec2_new=spect_new[1][1]
            spec2_new=spec2_new/spec2_new[0]
            spec3_new=spect_new[2][1]
            spec3_new=spec3_new/spec3_new[0]
            spec4_new=spect_new[3][1]
            spec4_new=spec4_new/spec4_new[0]
            if len(history[1])==4:
                spec_ers=torch.zeros(4)
                spec_ers[0]=torch.linalg.norm(spec1_new-history[1][0])
                spec_ers[1]=torch.linalg.norm(spec2_new-history[1][1])
                spec_ers[2]=torch.linalg.norm(spec3_new-history[1][2])
                spec_ers[3]=torch.linalg.norm(spec4_new-history[1][3])
                #print(history[0])
                #print(torch.max(spec_ers))

            if (len(history[1])==4 and torch.max(spec_ers) < ctm_args.ctm_conv_tol*100) \
                    or (history[0] >= ctm_args.ctm_max_iter):
                log.info({"history_length": history[0], "history": spec_ers})
                return True, history
            history[1]=[spec1_new,spec2_new,spec3_new,spec4_new]
            history[0]=history[0]+1
            return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    ctm_env_init, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_init, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)
    
    loss0 = energy_f(state, ctm_env_init, force_cpu=cfg.ctm_args.conv_check_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=args.force_cpu,\
        disp_corre_len=args.disp_corre_len)
    print("\n\n",end="")
    print(", ".join(["epoch",f"loss"]+[label for label in obs_labels]))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    
    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # rebuild on-site tensors
        if args.ansatz in ["A_1,B", "A_2,B"]:
            state.update_()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        loss= energy_f(state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        return loss, ctm_env_out, history, (t_ctm, t_conv_check)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if args.ansatz in ["A_1,B", "A_2,B"]:
            # implicit rebuild of on-site tensors
            state.update_()
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        if opt_context["line_search"]:
            print(", ".join([f"{epoch}",f"{loss}"]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))
        else:
            obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu,\
                disp_corre_len=args.disp_corre_len)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]), end="")
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))
            print(", "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]) )


    def post_proc(state, ctm_env, opt_context):
        pass
    
    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn,
        post_proc=post_proc)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    if args.ansatz in ["A_1,B", "A_2,B"]:
        state= read_ipess_kagome_pg_lc(outputstatefile)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCheckpoint_IPESS_Ansatze(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-chck"
    ANSATZE= [("A_1,B",),("A_2,B",)]

    def reset_couplings(self):
        args.j1= 1.0
        args.theta=0.2
        args.JD=0.0

    def setUp(self):
        self.reset_couplings()
        args.bond_dim=3
        args.chi=27
        args.seed=100
        args.opt_max_iter= 10
        args.instate_noise=0
        args.GLOBALARGS_dtype= "complex128"

    def test_checkpoint_ipess_ansatze(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np
        from ipeps.ipess_kagome import write_ipess_kagome_pg_lc 

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                self.reset_couplings()
                args.opt_max_iter= 10
                args.ansatz= ansatz[0]
                args.opt_resume= None
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}_"
                args.instate= args.out_prefix[len("RESULT_"):]+"_instate.json"

                # create random state
                ansatz_pgs= None
                if args.ansatz=="A_1,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A1_B
                if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
                if args.bond_dim in [3,6]:
                    p= 2 # phys dim
                    D= args.bond_dim
                    path= os.path.dirname(__file__)+"/../../su2sym/ipess_kagome/"

                    tri_t_A1= tenSU2.import_sym_tensors_generic((D,D,D), "A_1",
                        path+f"D{D}_trivalent.txt", dtype=torch.float64, device=cfg.global_args.device)
                    tri_t_A2= tenSU2.import_sym_tensors_generic((D,D,D), "A_2",
                        path+f"D{D}_trivalent.txt", dtype=torch.float64, device=cfg.global_args.device)
                    bond_t_A= tenSU2.import_sym_tensors_generic((p,D,D), "A",
                        path+f"D{D}_bond.txt", dtype=torch.float64, device=cfg.global_args.device)
                    bond_t_B= tenSU2.import_sym_tensors_generic((p,D,D), "B",
                        path+f"D{D}_bond.txt", dtype=torch.float64, device=cfg.global_args.device)
                    tri_t= tri_t_A1+tri_t_A2
                    bond_t= bond_t_A+bond_t_B
                    if args.ansatz in ["A_1,B", "A_2,B"]:
                        state= IPESS_KAGOME_PG_LC(
                            T_u=(torch.rand( len(tri_t), dtype=torch.float64, device=cfg.global_args.device ),\
                                tri_t),\
                            B_c=(torch.rand( len(bond_t), dtype=torch.float64, device=cfg.global_args.device ),\
                                bond_t),
                        SYM_UP_DOWN=True,SYM_BOND_S=True, pgs=ansatz_pgs)
                        write_ipess_kagome_pg_lc(state, args.instate)
                else:
                    raise RuntimeError("Bond dim "+str(args.bond_dim)+" is not supported")
            

                # i) run optimization and store the optimization data
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                obs_opt_lines=[]
                final_obs=None
                OPT_OBS= OPT_OBS_DONE= False
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                        OPT_OBS_DONE= True
                        OPT_OBS=False
                    if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                        obs_opt_lines.append(l)
                    if "epoch, loss," in l and not OPT_OBS_DONE: 
                        OPT_OBS= True
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs
                assert len(obs_opt_lines)>0

                # compare the line of observables with lowest energy from optimization (i) 
                # and final observables evaluated from best state stored in *_state.json output file
                # drop the last column, not separated by comma
                best_e_line_index= np.argmin([ float(l.split(',')[1]) for l in obs_opt_lines ])
                opt_line_last= [complex(x) for x in obs_opt_lines[best_e_line_index].split(",")[1:-1]]
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                for val0,val1 in zip(opt_line_last, fobs_tokens):
                    assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

                # ii) run optimization for 3 steps
                # reset j1 which is otherwise set by main() if args.theta is used
                args.opt_max_iter= 3 
                self.reset_couplings()
                main()
        
                # iii) run optimization from checkpoint
                args.instate=None
                args.opt_resume= args.out_prefix+"_checkpoint.p"
                args.opt_max_iter= 7
                self.reset_couplings()
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                obs_opt_lines_chk=[]
                final_obs_chk=None
                OPT_OBS= OPT_OBS_DONE= False
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                        OPT_OBS_DONE= True
                        OPT_OBS=False
                    if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                        obs_opt_lines_chk.append(l)
                    if "checkpoint.loss" in l and not OPT_OBS_DONE: 
                        OPT_OBS= True
                    if "FINAL" in l:    
                        final_obs_chk= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs_chk
                assert len(obs_opt_lines_chk)>0

                # compare initial observables from checkpointed optimization (iii) and the observables 
                # from original optimization (i) at one step after total number of steps done in (ii)
                opt_line_iii= [complex(x) for x in obs_opt_lines_chk[0].split(",")[1:]]
                # drop (last) normalization column
                opt_line_i= [complex(x) for x in obs_opt_lines[4].split(",")[1:-1]]
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                for val3,val1 in zip(opt_line_iii, opt_line_i):
                    assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

                # compare final observables from optimization (i) and the final observables 
                # from the checkpointed optimization (iii)
                fobs_tokens_1= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                fobs_tokens_3= [complex(x) for x in final_obs_chk[len("FINAL"):].split(",")]
                for val3,val1 in zip(fobs_tokens_3, fobs_tokens_1):
                    assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}_"
            instate= out_prefix[len("RESULT_"):]+"_instate.json"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log",instate]:
                if os.path.isfile(f): os.remove(f)