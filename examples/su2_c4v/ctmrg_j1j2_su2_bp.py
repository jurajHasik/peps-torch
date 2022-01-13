import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_lc_bp import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic.rdm import rdm2x1
# from ctm.one_site_c4v import transferops_c4v
from models import j1j2
from linalg.custom_svd import truncated_svd_gesdd
# from optim.fd_optim_lbfgs_mod import optimize_state
import su2sym.sym_ten_parser as tenSU2
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--force_cpu", action="store_true", help="force energy and observale evalution on CPU")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"+
    " during CTM convergence")
parser.add_argument("--corrf_dd_v", action='store_true', help="compute vertical dimer-dimer"\
    + " correlation function")
parser.add_argument("--top2", action='store_true', help="compute transfer matrix for width-2 channel")
args, unknown_args= parser.parse_known_args()
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2(j1=args.j1, j2=args.j2)
    energy_f= model.energy_2x2_2site

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_lc_bp(args.instate)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors_FIX(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(su2sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_LC_1SITE_PG(su2sym_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors_FIX(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            su2sym_b= tenSU2.import_sym_bonds(args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))

        c_A= torch.rand(len(su2sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        c_b= torch.rand(len(su2sym_b), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        c_A= c_A/torch.max(torch.abs(c_A))
        c_b= c_b/torch.max(torch.abs(c_b))
        coeffs = {"site": c_A, "bond": c_b}
        state = IPEPS_LC_BP({"site": su2sym_t, "bond": su2sym_b}, coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # state.sites[(1,0)]=state.sites[(1,0)]*4

    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        _rdm2x1= rdm2x1((0,0), state, env)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(_rdm2x1, history["rdm"], p=2).item()

        # log dist and observables
        if args.obs_freq>0 and \
            (len(history["log"])%args.obs_freq==0 or 
            (len(history["log"])-1)%args.obs_freq==0):
            t0_energy= time.perf_counter()
            e_curr = energy_f(state, env)
            t1_energy= time.perf_counter()
            obs_values, obs_labels = model.eval_obs(state, env)
            print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]\
                +[f"{v}" for v in obs_values]+[f"{t1_energy-t0_energy}"]))
        else:
            print(f"{len(history['log'])}, {dist}")
        
        # update history
        history["rdm"]=_rdm2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return True, history
        elif len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return False, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, \
        conv_check=ctmrg_conv_f)

    e_curr0 = energy_f(state, ctm_env)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    for c_loc,c_ten in ctm_env.C.items(): 
        u,s,v= truncated_svd_gesdd(c_ten, c_ten.size(0))
        print(f"\n\nspectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()