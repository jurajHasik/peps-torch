import os
import context
import argparse
import config as cfg
import math
import torch
import copy
import pytest
from collections import OrderedDict
from ipeps.ipess_kagome import *
from ipeps.ipeps_kagome import *
from models import spin1_kagome
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.ad_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--j1sq", type=float, default=0, help="nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--j2sq", type=float, default=0, help="next-nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=float, default=0, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", "IPESS_PG", "A_2,B"])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= spin1_kagome.S1_KAGOME(j1=args.j1,j1sq=args.j1sq,j2=args.j2,j2sq=args.j2sq,\
        jtrip=args.jtrip,jperm=args.jperm)

    # initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        ansatz_pgs=None
        if args.ansatz=="A_2,B": ansatz_pgs= ("A_2", "A_2", "B")
        
        if args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate)
            elif args.ansatz in ["IPESS_PG","A_2,B"]:
                state= read_ipess_kagome_pg(args.instate)

            # possibly symmetrize by PG
            if ansatz_pgs!=None:
                if type(state)==IPESS_KAGOME_GENERIC:
                    state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
                elif type(state)==IPESS_KAGOME_PG:
                    if state.pgs==(None,None,None):
                        state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
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
            B_a= torch.zeros(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                state= IPESS_KAGOME_PG(T_u, B_a, T_d, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
            elif args.ansatz in ["IPESS"]:
                B_b= torch.zeros(3, args.bond_dim, args.bond_dim,\
                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
                B_c= torch.zeros(3, args.bond_dim, args.bond_dim,\
                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
                state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d, 'B_b': B_b, 'B_c': B_c})
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=='RANDOM':
            bond_dim = args.bond_dim
            T_u= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            T_d= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_a= torch.rand(3, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                state = IPESS_KAGOME_PG(T_u, B_a, T_d, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
            elif args.ansatz in ["IPESS"]:
                B_b= torch.rand(3, args.bond_dim, args.bond_dim,\
                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
                B_c= torch.rand(3, args.bond_dim, args.bond_dim,\
                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
                state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d, 'B_b': B_b, 'B_c': B_c})
    
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

    def energy_f(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
        history.append(e_curr.item())
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    ctm_env_init, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_init, \
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
    
    loss0 = energy_f(state, ctm_env_init, force_cpu=cfg.ctm_args.conv_check_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=False)
    print("\n\n",end="")
    print(", ".join(["epoch",f"loss"]+[label for label in obs_labels]))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # build on-site tensors
        if args.ansatz in ["IPESS", "IPESS_PG", "A_2,B"]:
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                # explicit rebuild of on-site tensors
                tmp_state= to_PG_symmetric(state, state.pgs)
            else:
                tmp_state= state
            # include normalization of new on-site tensor
            tmp_state.sites= tmp_state.build_onsite_tensors()
        else:
            A= state.sites[(0,0)]
            A= A/A.abs().max()
            tmp_state= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(tmp_state, ctm_env_in)

        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_conv_check = ctmrg.run(tmp_state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        loss0 = energy_f(tmp_state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        # loc_ctm_args = copy.deepcopy(ctm_args)
        # loc_ctm_args.ctm_max_iter = 1
        # ctm_env_out, history1, t_ctm1, t_obs1 = ctmrg.run(state, ctm_env_out, ctm_args=loc_ctm_args)
        # loss1 = energy_f(state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)
        # loss = torch.max(loss0, loss1)
        
        loss= loss0
        return loss, ctm_env_out, history, t_ctm, t_conv_check

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if args.ansatz in ["IPESS_PG", "A_2,B"]:
            state_sym= to_PG_symmetric(state, state.pgs)
        else:
            state_sym= state
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state_sym,ctm_env,force_cpu=False)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]), end="")
        log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))
        print(" "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]) )

    def post_proc(state, ctm_env, opt_context):
        pass

    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn,
        post_proc=post_proc)

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCheckpoint_IPESS_Ansatze(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-spin1-trimer"
    ANSATZE= [("IPESS",False,False), ("IPEPS",None,None)]
        #("IPESS_PG",False,False), ("IPESS_PG",True,True), ("A_2,B",True,True)]

    def reset_couplings(self):
        args.j1= 1.0
        args.j1sq=1.0
        # P_12 = -1 + SS + SS^2, here we have h_12 = SS + SS^2 = 1 + P_12

    def setUp(self):
        self.reset_couplings()
        args.bond_dim=3
        args.chi=9
        args.seed=300
        args.opt_max_iter= 100
        args.instate_noise=0
        args.GLOBALARGS_dtype= "float64"
        args.OPTARGS_tolerance_grad= 1.0e-9

    @pytest.mark.slow
    def test_checkpoint_ipess_ansatze(self):
        import builtins
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np
        from ipeps.ipess_kagome import write_ipess_kagome_generic, write_ipess_kagome_pg 

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                self.reset_couplings()
                args.opt_max_iter= 100
                args.ansatz= ansatz[0]
                args.sym_up_dn= ansatz[1]
                args.sym_bond_S= ansatz[2]
                args.opt_resume= None
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}_"\
                    +("T" if ansatz[1] else "F")+("T" if ansatz[2] else "F")
                args.instate= args.out_prefix[len("RESULT_"):]+"_instate.json"

                # create random state
                ansatz_pgs= None
                if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
                bd = args.bond_dim
                phys_dim= 3
                T_u= torch.rand(bd, bd, bd,dtype=torch.complex128, device='cpu')
                T_d= torch.rand(bd, bd, bd,dtype=torch.complex128, device='cpu')
                B_c= torch.rand(phys_dim, bd, bd,dtype=torch.complex128, device='cpu')
                B_a= torch.rand(phys_dim, bd, bd,dtype=torch.complex128, device='cpu')
                B_b= torch.rand(phys_dim, bd, bd,dtype=torch.complex128, device='cpu')
                if args.ansatz in ["IPESS_PG", "A_2,B"]:
                    state = IPESS_KAGOME_PG(T_u, B_c, T_d=T_d, B_a=B_a, B_b=B_b,\
                        SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
                    write_ipess_kagome_pg(state, args.instate)
                elif args.ansatz in ["IPESS"]:
                    state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                        'B_b': B_b, 'B_c': B_c})
                    write_ipess_kagome_generic(state, args.instate)
                elif args.ansatz in ["IPEPS"]:
                    A = torch.rand((phys_dim**3, bd, bd, bd, bd),\
                        dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
                    A = A/torch.max(torch.abs(A))
                    state= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)
                    state.write_to_file(args.instate)

                # i) run optimization and store the optimization data
                tmp_out= StringIO()
                original_print = builtins.print
                def passthrough_print(*args, **kwargs):
                    original_print(*args, **kwargs)
                    kwargs.update(file=tmp_out)
                    original_print(*args, **kwargs)
                
                with patch('builtins.print', new=passthrough_print) as tmp_print:
                    main()

                # parse FINAL observables
                obs_opt_lines=[]
                final_obs=None
                OPT_OBS= OPT_OBS_DONE= False
                tmp_out.seek(0)
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
                # assert final_obs
                assert len(obs_opt_lines)>0

                # compare the line of observables with lowest energy from optimization (i) 
                # and final observables evaluated from best state stored in *_state.json output file
                # drop the last column, not separated by comma
                best_e_line_index= np.argmin([ float(l.split(',')[1]) for l in obs_opt_lines ])
                opt_line_last= [float(x) for x in obs_opt_lines[best_e_line_index].split(",")[1:-1]]
                assert opt_line_last[0]<1.33 # energy
                assert abs( opt_line_last[1] - opt_line_last[2] ) > 3 # dimerization as 1,2 are energies from up and down triangle
                assert opt_line_last[3]<0.1 # m_0
                assert opt_line_last[4]<0.1 # m_1
                assert opt_line_last[5]<0.1 # m_2


    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}_"\
                +("T" if ansatz[1] else "F")+("T" if ansatz[2] else "F")
            instate= out_prefix[len("RESULT_"):]+"_instate.json"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log",instate]:
                if os.path.isfile(f): os.remove(f)