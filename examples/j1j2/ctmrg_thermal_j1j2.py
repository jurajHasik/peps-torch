import os
import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps import IPEPO, IPEPS, read_ipeps
from ctm.generic.env import *
from ctm.generic import ctmrg
import ctm.generic.rdm_thermal as rdm_thermal
from models import j1j2, spin_triangular
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
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
    model = j1j2.J1J2_THERMAL(j1=args.j1, j2=args.j2, beta=0.)
    energy_f= model.energy_2x2_1site_BP
    eval_obs_f= model.eval_obs_1site_BP
    # model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2)
    # energy_f=model.energy_1x3
    # eval_obs_f= model.eval_obs

    # initialize an ipeps
    if args.instate!=None:
        # state = read_ipeps_c4v_thermal_ttn_v2(args.instate)
        # state.add_noise(args.instate_noise)
        state= read_ipeps(args.instate, vertexToSite=None)
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
        state= IPEPO({(0,0): A})
        # import pdb; pdb.set_trace()
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    def get_logz_per_site(coord,state,env):
        # A= state.site(coord)
        # A_sl_scale= A.abs().max()
        # if len(A.size())==5:
        #     # assume it is single-layer tensor with physica+ancilla fused
        #     dimsA= A.size()
        #     A= contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
        #     A= view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
        # elif len(A.size())==6:
        #     # assume it is single-layer tensor with physica+ancilla unfused
        #     if args.mode=="sl":
        #         auxD= A.size(5)
        #         A= torch.einsum('asuldr,asefgh->uelfdgrh',A,A).contiguous()
        #         A= A.view([auxD**2]*4)
        #         A_sl_scale= A_sl_scale**2
        #     elif args.mode=="dl":
        #         A= torch.einsum('ssuldr->uldr',A).contiguous()

        # C1--T1--C2                                          / C1--C2
        # |   |   |    C1----------------C2(shift=(-1,0))    /  |   |    C1--T2--C2
        # T4--A---T3 * |                 |                  /   T4--T2 * |   |   |
        # |   |   |    C4(shift=(0,-1))--C3(shift=(-1,-1)) /    |   |    C4--T3--C3 shift=(0,-1)
        # C4--T3--C3                                      /     C4--C3
        #                                                           shift=(-1,0)

        # closed C^4
        C4= rdm_thermal.norm_2x2(coord,state,env)

        # closed 2x3 and 3x2
        norm_2x3= rdm_thermal.norm_2x3(coord,state,env)
        norm_3x2= rdm_thermal.norm_3x2(coord,state,env)

        # closed rdm1x1
        rdm= rdm_thermal.rdm1x1(coord,state,env,mode='dl',\
            operator=torch.eye(A.size(0),dtype=A.dtype,device=A.device))

        log.info(f"get_logz_per_site rdm {rdm.item()} norm_2x3 {norm_2x3.item()}"
            +f" norm_3x2 {norm_3x2.item()} C4 {C4.item()}")

        logz_per_site= torch.log(rdm) + torch.log(C4)\
            - torch.log(norm_2x3) - torch.log(norm_3x2)
        return logz_per_site

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

    def ctmrg_conv_specC(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history={'spec': [], 'diffs': []}
        # use corner spectra
        diff=float('inf')
        diffs=None
        spec= env.get_spectra()
        spec_nosym_sorted= { s_key : s_t.sort(descending=True)[0] \
                for s_key, s_t in spec.items() }
        if len(history['spec'])>0:
            s_old= history['spec'][-1]
            diffs= [ sum((spec_nosym_sorted[k]-s_old[k])**2).item() \
                for k in spec.keys() ]
            diff= sum(diffs)
        history['spec'].append(spec_nosym_sorted)
        history['diffs'].append(diffs)
        
        log_z0= float('NaN') #get_logz_per_site((0,0),state,env)
        if len(next(iter(state.sites.values())).size())==4:
            # iPEPS/iPEPO sites have no physical indices, hence no observables
            # are accessible
            print(", ".join([f"{len(history['diffs'])}",f"{diff}",f"{log_z0}"]))
        else:
            pass

        if (len(history['diffs']) > 1 and abs(diff) < ctm_args.ctm_conv_tol)\
            or len(history['diffs']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['diffs']), "history": history['diffs']})
            return True, history
        return False, history

    # 0) For regular iPEPO (mode='dl') contract ancilla and physical index 
    #    to create regular rank-4 (aux indices only) on-site tensor
    #    For double-layer iPEPO (mode='sl') ...
    state_fused= state.to_nophys_ipeps() if args.mode=='dl' else \
        state.to_fused_ipeps()
    # sitesDL=dict()
    # for coord,A in state.sites.items():
    #     dimsA = A.size()
    #     a = contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
    #     a= view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #     sitesDL[coord]=a
    # state_fused = IPEPS(sites=sitesDL)
    
    ctm_env = ENV(args.chi, state_fused)
    init_env(state_fused, ctm_env)

    # 1) choose convergence criterion
    if len(next(iter(state_fused.sites.values())).size())==4:
        ctmrg_conv_f= ctmrg_conv_specC
    else:
        ctmrg_conv_f= ctmrg_conv_rdm2x1

    e0= energy_f(state, ctm_env) #,force_cpu=True) #args.mode,
    log_z0= 0 #get_logz_per_site((0,0), state, ctm_env)
    obs_values, obs_labels = eval_obs_f(state, ctm_env)#, force_cpu=True) #args.mode,
    
    print("\n\n",end="")
    print(", ".join(["epoch","dist","log_z","e0"]+obs_labels))
    print(", ".join([f"{-1}",f"{float('inf')}",f"{log_z0}",f"{e0}"]\
        +[f"{v}" for v in obs_values]))

    ctm_env, *ctm_log = ctmrg.run(state_fused, ctm_env, conv_check=ctmrg_conv_f)
    history, t_ctm, t_obs= ctm_log 

    e0 = energy_f(state,ctm_env)#,force_cpu=True) #args.mode,
    log_z0= 0 #get_logz_per_site((0,0), state, ctm_env)
    obs_values, obs_labels = eval_obs_f(state, ctm_env)#,force_cpu=True) #args.mode, 
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