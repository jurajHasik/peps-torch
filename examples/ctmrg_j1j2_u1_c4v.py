import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_lc import * 
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v, transferops_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import j1j2
import u1sym.sym_ten_parser as tenU1
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--u1_class", type=str, default="B")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j3", type=float, default=0., help="next-to-next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
# additional observables-related arguments
parser.add_argument("--corrf_canonical", action='store_true', help="align spin operators" \
    + " with the vector of spontaneous magnetization")
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"+
    " during CTM convergence")
parser.add_argument("--corrf_dd_v", action='store_true', help="compute vertical dimer-dimer"\
    + " correlation function")
parser.add_argument("--top2", action='store_true', help="compute transfer matrix for width-2 channel")
args, unknown_args= parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2, j3=args.j3, \
        hz_stag=args.hz_stag, delta_zz=args.delta_zz)
    energy_f= model.energy_1x1_lowmem
    # energy_f= model.energy_1x1_tiled
    # energy_f= model.energy_1x1

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_lc_1site_pg(args.instate)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [2,3,4,5,6,7,8]:
            u1sym_t= tenU1.import_sym_tensors(2,args.bond_dim,"A_1",\
                infile=f"u1sym/D{args.bond_dim}_U1_{args.u1_class}.txt",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(u1sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_LC_1SITE_PG(u1sym_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [2,3,4,5,6,7,8]:
            u1sym_t= tenU1.import_sym_tensors(2, args.bond_dim, "A_1", \
                infile=f"u1sym/D{args.bond_dim}_U1_{args.u1_class}.txt", \
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.rand(len(u1sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))
        coeffs = {(0,0): A}
        state = IPEPS_LC_1SITE_PG(u1sym_t, coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")
    
    print(state)

    @torch.no_grad()
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args): 
        if not history:
            history=dict({"log": []})
        rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        # log dist and observables
        if args.obs_freq>0 and \
            (len(history["log"])%args.obs_freq==0 or 
            (len(history["log"])-1)%args.obs_freq==0):
            e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
            obs_values, obs_labels = model.eval_obs(state, env, force_cpu=True)
            print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
        else:
            print(f"{len(history['log'])}, {dist}")
        # update history
        history["rdm"]=rdm2x1
        history["log"].append(dist)

        converged= dist<ctm_args.ctm_conv_tol
        if converged or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(env)})
            return converged, history
        return False, history

    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)

    e_curr0 = energy_f(state, ctm_env_init, force_cpu=True)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=True)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, \
        conv_check=ctmrg_conv_rdm2x1)

    e_curr0 = energy_f(state, ctm_env_init, force_cpu=True)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=True)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # ----- additional observables ---------------------------------------------
    corrSS= model.eval_corrf_SS(state, ctm_env_init, args.corrf_r, canonical=args.corrf_canonical)
    print("\n\nSS r "+" ".join([label for label in corrSS.keys()])+f" canonical {args.corrf_canonical}")
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrDD= model.eval_corrf_DD_H(state, ctm_env_init, args.corrf_r)
    print("\n\nDD r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    if args.corrf_dd_v:
        corrDD_V= model.eval_corrf_DD_V(state, ctm_env_init, args.corrf_r)
        print("\n\nDD_v r "+" ".join([label for label in corrDD_V.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrDD_V[label][i]}" for label in corrDD_V.keys()]))

    # environment diagnostics
    print("\n\nspectrum(C)")
    u,s,v= torch.svd(ctm_env_init.C[ctm_env_init.keyC], compute_uv=False)
    for i in range(args.chi):
        print(f"{i} {s[i]}")

    # transfer operator spectrum 1-site-width channel
    print("\n\nspectrum(T)")
    l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env_init)
    for i in range(l.size()[0]):
        print(f"{i} {l[i,0]} {l[i,1]}")

    # transfer operator spectrum 2-site-width channel
    if args.top2:
        print("\n\nspectrum(T2)")
        l= transferops_c4v.get_Top2_spec_c4v(args.top_n, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrg(unittest.TestCase):
    def setUp(self):
        args.instate=None 
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.GLOBALARGS_device="cpu"

    # basic tests
    def test_ctmrg_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()
