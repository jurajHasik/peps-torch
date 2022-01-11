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
from ctm.pess_kagome import rdm_kagome
from ctm.generic import ctmrg
import json
import unittest
import logging
import scipy.io as io
import numpy

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=None, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--JD", type=float, default=0, help="two-spin DM interaction")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=float, default=0, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", "IPESS_PG", 'A_2,B'])
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",\
     choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--force_cpu", action='store_true', dest='force_cpu', help="force RDM contractions on CPU")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    if not args.theta is None:
        args.j1= args.j1*math.cos(args.theta*math.pi)
        args.jtrip= args.j1*math.sin(args.theta*math.pi)
    print(f"j1={args.j1}; jD={args.JD}; j2={args.j2}; jtrip={args.jtrip}")
    model= spin_half_kagome.S_HALF_KAGOME(j1=args.j1,JD=args.JD,j2=args.j2,\
        jtrip=args.jtrip,jperm=args.jperm)

    # initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        
        if args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate)
            elif args.ansatz in ["IPESS_PG","A_2,B"]:
                state= read_ipess_kagome_pg(args.instate)

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
            B_c= torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_a= torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_b= torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
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
            B_c= torch.rand(model.phys_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_a= torch.rand(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_b= torch.rand(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                state = IPESS_KAGOME_PG(T_u, B_c, T_d=T_d, B_a=B_a, B_b=B_b,\
                    SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
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


    def energy_f(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=args.force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=args.force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def energy_f_complex(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=args.force_cpu)
        e_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=args.force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=args.force_cpu)
        return e_dn
        
    def eval_corner_spectra(env):
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

    def report_conv_fn(state, ctm_env, conv_step, conv_crit):
        if args.obs_freq>0 and \
            (conv_step%args.obs_freq==0 or (conv_step-1)%args.obs_freq==0):
            e_curr = energy_f(state, env, force_cpu=args.force_cpu)
            obs_values, obs_labels = model.eval_obs(state, env, force_cpu=args.force_cpu)
            print(", ".join([f"{conv_step}",f"{conv_crit}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
        else:
            print(f"{conv_step}, {conv_crit}")

    if args.CTM_check=="Energy":
        def ctmrg_conv_fn(state, env, history, ctm_args=cfg.ctm_args):
            if not history: history = []
            e_curr = energy_f(state, env, force_cpu=args.force_cpu)
            history.append(e_curr.item())

            # log conv_crit and observables
            report_conv_fn(state,env,len(history),\
                float('inf') if len(history) < 2 else abs(history[-1] - history[-2]))

            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="Partial_energy":
        def ctmrg_conv_fn(state, env, history, ctm_args=cfg.ctm_args):
            if not history: history = []
            if len(history)>8:
                e_curr = dn_energy_f(state, env, force_cpu=args.force_cpu)
                history.append(e_curr.item())
            else:
                history.append(float('inf'))

            # log conv_crit and observables
            report_conv_fn(state,env,len(history),\
                float('inf') if len(history) < 10 else abs(history[-1] - history[-2]))

            if (len(history) > 9 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol*2) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="SingularValue":
        def ctmrg_conv_fn(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history_spec = []
                history_ite=1
                history=[history_ite, history_spec]
            spect_new=eval_corner_spectra(env)
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

            # log conv_crit and observables
            report_conv_fn(state,env,history[0],torch.max(spec_ers))

            if (len(history[1])==4 and torch.max(spec_ers) < ctm_args.ctm_conv_tol*100) \
                    or (history[0] >= ctm_args.ctm_max_iter):
                log.info({"history_length": history[0], "history": spec_ers})
                return True, history
            history[1]=[spec1_new,spec2_new,spec3_new,spec4_new]
            history[0]=history[0]+1
            return False, history

    
    # 3) initialize environment 
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    # 4) (optional) compute observables as given by initial environment
    e_curr0 = energy_f(state, ctm_env_init,force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # 5) (main) execute CTM algorithm
    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_fn)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init, force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=args.force_cpu)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # 7) ----- additional observables ---------------------------------------------
    # chirality
    print("\n")
    Pijk= model.P_triangle
    Pijk_inv= model.P_triangle_inv
    chi_R= Pijk + Pijk_inv
    chi_I= 1.0j * (Pijk - Pijk_inv)

    ev_chi_R_downT= rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, ctm_env_init,\
        chi_R)
    ev_chi_I_downT= rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, ctm_env_init,\
        chi_I)
    print(f"Re(Chi) downT {ev_chi_R_downT} Im(Chi) downT {ev_chi_I_downT}")

    rho_upT= rdm_kagome.rdm2x2_up_triangle_open((0,0), state, ctm_env_init)
    ev_chi_R_upT= torch.einsum('ijkmno,mnoijk',rho_upT, chi_R)
    ev_chi_I_upT= torch.einsum('ijkmno,mnoijk',rho_upT, chi_I)
    print(f"Re(Chi) upT {ev_chi_R_upT} Im(Chi) upT {ev_chi_I_upT}")


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()



   
