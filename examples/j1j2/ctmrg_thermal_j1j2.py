import os
import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from ipeps.ipeps_c4v_thermal import *
from linalg.custom_eig import truncated_eig_sym
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v.env_c4v import _init_from_ipeps_pbc
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import ddA_rdm1x1
from ctm.one_site_c4v.rdm_c4v_thermal import entropy, rdm1x1_sl, rdm2x1_sl, rdm2x2
from ctm.one_site_c4v import transferops_c4v
from optim.exp_ad_optim_vtnr import optimize_state
from models import j1j2
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--l2d", type=int, default=1)
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--beta", type=float, default=0., help="inverse temperature")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--mode", type=str, default="dl")
args, unknown_args= parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # 0) initialize model
    model = j1j2.J1J2_C4V_BIPARTITE_THERMAL(j1=args.j1, j2=args.j2, j3=0, 
        hz_stag= 0.0, delta_zz=1.0, beta=0.)
    energy_f= model.energy_1x1
    eval_obs_f= model.eval_obs

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_c4v_thermal_ttn_v2(args.instate)
        # state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        state= IPEPS_C4V_THERMAL_TTN_V2.load_checkpoint(args.opt_resume, metadata=None,
            peps_args=cfg.peps_args, global_args=cfg.global_args)
        assert args.layers_Ds == state.iso_Ds, "Unexpected isometry dimensions"
    elif args.ipeps_init_type in ['CPEPO-SVD']:
        A= None #model.ipepo_trotter_suzuki(args.beta/(2**args.layers))
        if args.ipeps_init_type=='CPEPO-SVD':
            import subprocess
            dt= args.beta if args.mode=='dl' else args.beta/2
            p = subprocess.Popen(f"octave GetPEPO.m {dt:.5f} {args.j2:.5f}", \
                stdout=subprocess.PIPE, shell=True)
            p_status= p.wait()
            from scipy.io import loadmat
            A= torch.from_numpy(loadmat(f"hbpepo_dt{dt:.5f}_j2{args.j2:.5f}.mat")['T'])\
                .permute(4,5,0,1,2,3).contiguous()
            import pdb; pdb.set_trace()
        state= IPEPS_C4V_THERMAL(A)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    def get_logz_per_site(A, C, T):
        A_sl_scale= A.abs().max()
        if len(A.size())==5:
            # assume it is single-layer tensor with physica+ancilla fused
            auxD= A.size(4)
            A= torch.einsum('suldr,sefgh->uelfdgrh',A,A).contiguous()
            A= A.view([auxD**2]*4)
        elif len(A.size())==6:
            # assume it is single-layer tensor with physica+ancilla unfused
            if args.mode=="sl":
                auxD= A.size(5)
                A= torch.einsum('asuldr,asefgh->uelfdgrh',A,A).contiguous()
                A= A.view([auxD**2]*4)
                A_sl_scale= A_sl_scale**2
            elif args.mode=="dl":
                A= torch.einsum('ssuldr->uldr',A).contiguous()

        # C--T--C            / C--C
        # |  |  |   C--C    /  |  |   C--T--C
        # T--A--T * |  |   /   T--T * |  |  |
        # |  |  |   C--C  /    |  |   C--T--C
        # C--T--C        /     C--C

        # closed C^4
        C4= torch.einsum('ij,jk,kl,li',C,C,C,C)

        # closed rdm1x1
        CTC = torch.tensordot(C,T,([1],[0]))
        #   C--0
        # A |
        # | T--2->1
        # | 1
        #   0
        #   C--1->2
        CTC = torch.tensordot(CTC,C,([1],[0]))

        # closed CTCCTC
        #   C--0 2--C
        # A |       |
        # | T--1 1--T |
        #   |       | V
        #   C--2 0--C
        CTCCTC= torch.tensordot(CTC,CTC,([0,1,2],[2,1,0]))

        #   C--0
        # A |
        # | T--1
        # | |       2->3
        #   C--2 0--T--1->2
        CTC = torch.tensordot(CTC,T,([2],[0]))
        # rdm = torch.tensordot(rdm,A,([1,3],[1,2]))
        # rdm = torch.tensordot(rdm,A/(A_sl_scale**2),([1,3],[1,2]))
        rdm = torch.tensordot(CTC,A/A_sl_scale,([1,3],[1,2]))

        #   C--0 2--T-------C
        #   |       3       |
        # A |       2       |
        # | T-------A--3 1--T
        # | |       |       |
        # | |       |       |
        #   C-------T--1 0--C
        rdm = torch.tensordot(rdm,CTC,([0,1,2,3],[2,0,3,1]))

        log.info(f"get_z_per_site rdm {rdm.item()} CTCCTC {CTCCTC.item()} C4 {C4.item()}")

        # z_per_site= (rdm/CTC)*(CTC/C4)
        logz_per_site= torch.log(A_sl_scale) + torch.log(rdm) + torch.log(C4)\
            - 2*torch.log(CTCCTC)
        return logz_per_site
        # return z_per_site

    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        # unfuse ancilla+physical
        dims= state.site().size()
        _tmp_site= state.site().view( 2,2, *dims[1:])
        _tmp_state= IPEPS_C4V_THERMAL(_tmp_site)
        rdm2x1= rdm2x1_sl(_tmp_state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"]=rdm2x1
        history["log"].append(dist)

        e0= energy_f(_tmp_state, env, force_cpu=True) #args.mode,
        log_z0= get_logz_per_site(_tmp_state.site(), env.get_C(), env.get_T())
        obs_values, obs_labels = eval_obs_f(_tmp_state, env, force_cpu=True) #args.mode,
        print(", ".join([f"{len(history['log'])}",f"{dist}",f"{log_z0}",f"{e0}"]\
            +[f"{v}" for v in obs_values]))

        if dist<ctm_args.ctm_conv_tol and len(history['log']) < ctm_args.ctm_max_iter:
            return True, history
        return False, history
    
    def ctmrg_conv_Cspec(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        U, specC, V= torch.svd(env.get_C(), compute_uv=False)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(specC, history["specC"], p=2).item()
        history["specC"]=specC
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return True, history
        elif len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return False, history
        return False, history

    ctmrg_conv_f= ctmrg_conv_rdm2x1


    # 0) 
    #    fuse ancilla+physical index and create regular c4v ipeps with rank-5 on-site
    #    tensor
    state_fused= state.to_nophys_ipeps_c4v() if args.mode=='dl' else \
        state.to_fused_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)


    e0= energy_f(state, ctm_env, force_cpu=True) #args.mode,
    log_z0= get_logz_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    obs_values, obs_labels = eval_obs_f(state, ctm_env, force_cpu=True) #args.mode,
    
    print("\n\n",end="")
    print(", ".join(["epoch","dist","log_z","e0"]+obs_labels))
    print(", ".join([f"{-1}",f"{float('inf')}",f"{log_z0}",f"{e0}"]\
        +[f"{v}" for v in obs_values]))

    ctm_env, *ctm_log = ctmrg_c4v.run_dl(state_fused, ctm_env, conv_check=ctmrg_conv_f)
    history, t_ctm, t_obs= ctm_log 

    e0 = energy_f(state,ctm_env,force_cpu=True) #args.mode,
    log_z0= get_logz_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    obs_values, obs_labels = eval_obs_f(state, ctm_env,force_cpu=True) #args.mode, 
    S0, r2_0= 0, 0
    print("\n\n",end="")
    print(", ".join(["epoch","log_z0","e0","S0","r2_0"]+obs_labels))
    print(", ".join(["FINAL",f"{log_z0}",f"{e0}",f"{S0}",f"{r2_0}"]\
        +[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()