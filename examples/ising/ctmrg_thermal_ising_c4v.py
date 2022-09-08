import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from ipeps.ipeps_c4v_thermal import *
from linalg.custom_eig import truncated_eig_sym
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v_thermal import entropy, rdm1x1_sl, rdm2x1_sl, rdm2x2
# from ctm.one_site_c4v import transferops_c4v
from optim.ad_optim_lbfgs_mod import optimize_state
from models import ising
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--l2d", type=int, default=1)
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--hz", type=float, default=0., help="longitudinal field")
parser.add_argument("--q", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--beta", type=float, default=0., help="inverse temperature")
parser.add_argument("--init_beta", type=float, default=0., \
    help="initial inverse temperature for BETA initialization")
parser.add_argument("--init_layers", type=int, default=1)
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"+
    " during CTM convergence")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--mode", type=str, default="dl")
args, unknown_args= parser.parse_known_args()
# set to BETA
args.ipeps_init_type= "BETA"

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # 0) initialize model
    model = ising.ISING_C4V(hx=args.hx, hz=args.hz, q=args.q)
    assert args.q==0,"plaquette term is not supported"
    energy_f= model.energy_1x1_nn_thermal
    obs_f= model.eval_obs_thermal
    
    # 1) initialize an thermal ipepo
    if args.instate!=None:
        state = read_ipeps_c4v_thermal_ttn_v2(args.instate)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [4]:
            elem_t= _build_elem_t()
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(elem_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_C4V_THERMAL_LC(elem_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type in ['BETA']:
        if args.ipeps_init_type=='BETA':
            assert args.init_layers>0,"number of layers must be larger than 0"
            A0= model.ipepo_trotter_suzuki(args.init_beta/args.init_layers)
            A= A0.clone()
            for i in range(1,args.init_layers):
                A= torch.einsum("sxuldr,xpefgh->spuelfdgrh",A,A0).contiguous()
                A= A.view([model.phys_dim]*2 + [2**(i+1)]*4)
        state = IPEPS_C4V_THERMAL(A)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) define CTMRG convergence callbacks
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        # 2.1) unfuse ancilla+physical
        dims= state.site().size()
        _tmp_site= state.site().view( model.phys_dim,model.phys_dim, *dims[1:])
        _tmp_state= IPEPS_C4V_THERMAL(_tmp_site)
        rdm2x1= rdm2x1_sl(_tmp_state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        # 2.2) log dist and observables
        if args.obs_freq>0 and \
            (len(history["log"])%args.obs_freq==0 or 
            (len(history["log"])-1)%args.obs_freq==0):
            e_curr = energy_f(_tmp_state, env, force_cpu=ctm_args.conv_check_cpu)
            obs_values, obs_labels = obs_f(_tmp_state, env, force_cpu=True)
            print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
        else:
            print(f"{len(history['log'])}, {dist}")
        # 2.3) update history
        history["rdm"]=rdm2x1
        history["log"].append(dist)
        
        converged= dist<ctm_args.ctm_conv_tol
        if converged or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return converged, history
        return False, history

    def ctmrg_conv_f2(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        U, specC, V= torch.svd(env.get_C(), compute_uv=False)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(specC, history["specC"], p=2).item()
        history["specC"]=specC
        history["log"].append(dist)

        converged= dist<ctm_args.ctm_conv_tol
        if converged or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return converged, history
        return False, history


    # # 4) define auxiliary functions to evaluate various approximate entropies
    # def approx_S(state, ctm_env, ad_decomp_reg=cfg.ctm_args.ad_decomp_reg, force_cpu=False):
    #     return entropy( rdm2x2(state, ctm_env), ad_decomp_reg )/4.

    # def renyi2_site(a, D1, D2):
    #     # a rank-6 on-site tensor with indices [apuldr]
    #     # 0) build projectors
    #     #
    #     #    ||/p0       
    #     #  ==a*a== -> ==a*a== = P D P^H
    #     # p1/||          
    #     #
    #     tmp_M= torch.einsum('apuldr,apufdh->lfrh',a,a.conj()).contiguous().view(\
    #         a.size(3)**2,a.size(5)**2)
    #     D, P= truncated_eig_sym(tmp_M, D1, keep_multiplets=True,\
    #             verbosity=cfg.ctm_args.verbosity_projectors)
    #     # import pdb; pdb.set_trace()
    #     #
    #     # 1) build 1st layer
    #     #
    #     #      |
    #     #      P
    #     #      ||/p0        |/p0
    #     # --P==a*a==P-- = --l1--
    #     #   p1/||        p1/|
    #     #      P
    #     #      |
    #     #
    #     l1= torch.einsum('apuldr,asefgh->psuelfdgrh',a,a.conj()).contiguous().view(\
    #         a.size(1), a.size(1),a.size(2)**2,a.size(3)**2,a.size(4)**2,a.size(5)**2)
    #     l1= torch.einsum('psxywz,xu,yl,wd,zr->psuldr',l1, P, P, P, P).contiguous()
    #     #
    #     # 2) build 2nd layer
    #     #
    #     #   p0 
    #     #   |/        ||  
    #     # --l1-- => ==l2==
    #     #  /|         ||
    #     #   p1
    #     #   |/
    #     # --l1--
    #     #  /p0
    #     #
    #     l2= torch.einsum('psuldr,spxywz->uxlydwrz',l1,l1).contiguous().view(
    #         (D1**2,)*4)

    #     # prepare initial environment tensors
    #     #
    #     # C--, --T--
    #     # |      |
    #     #
    #     C_0= torch.einsum('psuldr,spulwz->dwrz',l1,l1).contiguous().view(
    #         (D1**2,)*2)
    #     T_0= torch.einsum('psuldr,spuywz->lydwrz',l1,l1).contiguous().view(
    #         (D1**2,)*3)

    #     #
    #     # 3) (optional) truncate auxiliary bonds
    #     if D2<D1**2:
    #         tmp_M= l2.view( (D1,D1,D1**2)*2 )
    #         tmp_M= torch.einsum('ijlijr->lr',tmp_M).contiguous()
    #         D, P= truncated_eig_sym(tmp_M, D2, keep_multiplets=True,\
    #             verbosity=cfg.ctm_args.verbosity_projectors)
    #         l2= torch.einsum('xywz,xu,yl,wd,zr->uldr',l2, P, P, P, P).contiguous()
    #         C_0= torch.einsum('wz,wd,zr',C_0,P,P).contiguous()
    #         T_0= torch.einsum('ywz,yl,wd,zr',T_0,P,P,P).contiguous()

    #     return l2, C_0, T_0

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

    # def approx_renyi2(state,env,D1,D2):
    #     # get intial renyi2 on-site tensor and env tensors
    #     # state.site() rank-6 tensor [apuldr]
    #     a= state.site()
    #     # l2 - rank-4 tensor [uldr]
    #     l2, C0, T0= renyi2_site(a,D1,D2)
    #     state_r2= IPEPS_C4V(l2)
    #     env_r2= ENV_C4V(args.chi, state_r2)
    #     init_env(None, env_r2, (C0,T0))

    #     # run CTM
    #     env_r2, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_r2, env_r2, \
    #         conv_check=ctmrg_conv_f2)
    #     # print(history)

    #     # renyi-2 = -log( (Tr \rho^2)/Z^2 ) = -log ( (Tr \rho^2) / (Tr \rho)^2 )
    #     #
    #     rho2_per_site= get_z_per_site( state_r2.site(), env_r2.get_C(), env_r2.get_T() )
        
    #     # double-layer
    #     l1= torch.einsum('apuldr,apxywz->uxlydwrz',a,a.conj())\
    #         .contiguous().view(a.size(2)**2,a.size(3)**2,a.size(4)**2,a.size(5)**2)
    #     rho1_per_site= get_z_per_site( l1, env.get_C(), env.get_T() )

    #     renyi2= -torch.log(rho2_per_site/(rho1_per_site**2))
    #     return renyi2


    # 5) fuse ancilla+physical index and create regular c4v ipeps with rank-5 on-site
    #    tensor
    state_fused= state.to_fused_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)


    # 6) evaluate observables on initial environment ansatz
    log_z0= get_logz_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    e0= energy_f(state, ctm_env, args.mode, force_cpu=True)
    obs_values, obs_labels= obs_f(state, ctm_env, args.mode, force_cpu=True)
    print("\n",end="")
    print(", ".join(["epoch","dist","log_z","e0"]+obs_labels))
    print(", ".join(["-1",f"{float('inf')}",f"{log_z0}",f"{e0}"]+[f"{v}" for v in obs_values]))


    # 7) compute environment by CTMRG
    ctm_env, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_fused, ctm_env, \
        conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

    # 8) evaluate final observables and entropies
    log_z0= get_logz_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    e0= energy_f(state, ctm_env, args.mode, force_cpu=True)
    obs_values, obs_labels= obs_f(state, ctm_env, args.mode, force_cpu=True)
    # S0= approx_S(state, ctm_env)
    # r2_0= approx_renyi2(state, ctm_env, args.l2d, args.bond_dim**2)
    # try:
    #     F_r2_0= e0 - 1./args.beta * r2_0
    # except ZeroDivisionError as e:
    #     F_r2_0= float('inf')
    S0=r2_0=F_r2_0= 0
    print("\n")
    print(", ".join(["epoch","log_z0","e0","F_r2_0","S0","r2_0"]+obs_labels))
    print("FINAL "+", ".join([f"{log_z0}",f"{e0}", f"{F_r2_0}", f"{S0}", f"{r2_0}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # ----- additional observables ---------------------------------------------
    # environment diagnostics
    print("\n\nspectrum(C)")
    u,s,v= torch.svd(ctm_env.C[ctm_env.keyC], compute_uv=False)
    for i in range(args.chi):
        print(f"{i} {s[i]}")

    # transfer operator spectrum 1-site-width channel
    # print("\n\nspectrum(T)")
    # l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env_init)
    # for i in range(l.size()[0]):
    #     print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
