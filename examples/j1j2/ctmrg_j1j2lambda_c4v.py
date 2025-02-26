import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from ipeps.ipeps_lc import read_ipeps_lc_1site_pg 
from groups.pg import make_c4v_symm
from groups.su2 import SU2
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v, transferops_c4v, corrf_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl, rdm2x1, aux_rdm1x1
from models import j1j2
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--lmbd", type=float, default=0., help="chiral plaquette interaction")
parser.add_argument("--j3", type=float, default=0., help="next-to-next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
# additional observables-related arguments
parser.add_argument("--corrf_canonical", action='store_true', help="align spin operators" \
    + " with the vector of spontaneous magnetization")
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"
    + "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--corrf_dd_v", action='store_true', help="compute vertical dimer-dimer"\
    + " correlation function")
parser.add_argument("--top2", action='store_true', help="compute transfer matrix for width-2 channel")
parser.add_argument("--ff_ss", action='store_true', help="compute form-factors for spin-spin correlations")
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
parser.add_argument("--EH_n", type=int, default=1, help="number of leading eigenvalues "+
    "of EH to compute")
parser.add_argument("--EH_T_ED_L", type=int, default=0, help="max. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by ED")
parser.add_argument("--EH_T_ARP_minL", type=int, default=0, help="min. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by Arnoldi")
parser.add_argument("--EH_T_ARP_maxL", type=int, default=-1, help="max. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by Arnoldi")
args, unknown_args= parser.parse_known_args()

def main():
    # 0) parse command line arguments and configure simulation parameters
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2, j3=args.j3, \
        hz_stag=args.hz_stag, delta_zz=args.delta_zz, lmbd=args.lmbd)
    energy_f= model.energy_1x1

    # 1) initialize an ipeps - read from file or create a random one
    if args.instate!=None:
        try:
            state = read_ipeps_lc_1site_pg(args.instate)
            assert len(state.coeffs)==1, "Not a 1-site ipeps"
        except:
            state = read_ipeps_c4v(args.instate)
        state.add_noise(args.instate_noise)
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        state = IPEPS_C4V(A)
        state = to_ipeps_c4v(state)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    # 2) define convergence criterion for CTM algorithm. This function is to be 
    #    invoked at every CTM step. We also use it to evaluate observables of interest 
    #    during the course of CTM
    # 2a) convergence criterion based on on-site energy
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        
        e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
        history.append(e_curr.item())

        if args.obs_freq>0 and \
            (len(history)%args.obs_freq==0 or (len(history)-1)%args.obs_freq==0):
            obs_values, obs_labels = model.eval_obs(state, env)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
        else:
            print(", ".join([f"{len(history)}",f"{e_curr}"]))

        converged= len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol
        if converged or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history,
                "final_multiplets": compute_multiplets(env)})
            return converged, history
        return False, history

    # 2b) convergence criterion based on 2-site reduced density matrix 
    #     of nearest-neighbours
    @torch.no_grad()
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        rdm= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
        # rdm= rdm2x1(state, env, force_cpu=ctm_args.conv_check_cpu,
        #     verbosity=ctm_args.verbosity_rdm)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm, history["rdm"], p=2).item()
        # log dist and observables
        if args.obs_freq>0 and \
            (len(history["log"])%args.obs_freq==0 or 
            (len(history["log"])-1)%args.obs_freq==0):
            e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
            obs_values, obs_labels = model.eval_obs(state, env, force_cpu=ctm_args.conv_check_cpu)
            print(", ".join([f"{len(history['log'])}",f"{dist}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
        else:
            print(f"{len(history['log'])}, {dist}")
        # update history
        history["rdm"]=rdm
        history["log"].append(dist)

        converged= dist<ctm_args.ctm_conv_tol
        if converged or len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(env)})
            return converged, history
        return False, history

    # 3) initialize environment 
    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)

    # 4) (optional) compute observables as given by initial environment 
    e_curr0 = energy_f(state, ctm_env_init, force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # 5) (main) execute CTM algorithm
    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_rdm2x1)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init, force_cpu=args.force_cpu)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=args.force_cpu)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")


    # 7) ----- additional observables ---------------------------------------------
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
    return
    # transfer operator spectrum
    print("\n\nspectrum(T)")
    l,vecs= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env_init,\
        normalize=False,eigenvectors=True)
    r_0= vecs[:,0].view(ctm_env_init.chi,state.site().size(1)**2,ctm_env_init.chi)
    l_0= r_0.conj()
    _l_norm_fac= (l[0,0]+l[0,1]*1.0j).abs()
    for i in range(l.size()[0]):        
        print(f"{i} {l[i,0]/_l_norm_fac} {l[i,1]/_l_norm_fac} {l[i,0]} {l[i,1]}")


    corrSS= model.eval_corrf_SS(state, ctm_env_init, args.corrf_r, canonical=args.corrf_canonical,\
        rl_0=(r_0,l_0))
    print("\n\nSS r "+" ".join([label for label in corrSS.keys()])+f" canonical {args.corrf_canonical}")
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # transfer operator spectrum
    if args.top2:
        print("\n\nspectrum(T2)")
        l= transferops_c4v.get_Top2_spec_c4v(args.top_n, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

    # entanglement spectrum
    # for L in range(1,args.EH_T_ED_L):
    #     S= transferops.get_full_EH_spec_Ttensor(L, *sdp, state, ctm_env_init)
    #     print(f"\nEH_T_ED[{sdp[0]},{sdp[1]}] L={L}")
    #     for i in range(min(S.size(0),args.EH_n)):
    #         print(f"{i} {S.real[i]} {S.imag[i]}")

    for L in range(args.EH_T_ARP_minL,args.EH_T_ARP_maxL+1):
        S=transferops_c4v.get_EH_spec_Ttensor(args.EH_n, L, state, ctm_env_init)
        print(f"\nEH_T_ARP L={L}")
        for i in range(args.EH_n):
            print(f"{i} {S[i,0]} {S[i,1]}")

    if not args.ff_ss: return
    # form-factor analysis
    print("\n\nFormfactors(T) for S")
    # Solve eigenvalue problem
    #                        ----T---
    # <r_i| lmabda_i = <r_i| --A^+A--
    #                        ----T---
    #
    # Since transfer matrix TA^+AT is hermitian, left eigenvector is h.c. of right one
    #
    # ---T---- 
    # --A^+A-- |l_i> = lambda_i |l_i> = lambda_i (<r_i|)^+
    # ---T----
    #
    print("TM S.S resolved by formfactors at r=2")
    # compute normalization for r=2
    #
    #                          --T------T-----T------
    # <L(O)|TA^+AT|R(O)>= <r_0|--A^+OA--A^+A--A^+OA--|l_0> =\sum_i lambda_i <L(O)|l_i><r_i|R(O)> 
    #                          --T------T-----T------
    #
    irrep= SU2(2,dtype=torch.complex128,device=cfg.global_args.device)
    S_xyz= irrep.S()
    S_ops= {'id': irrep.I(), 'sz': S_xyz[0,:,:], 'sx': S_xyz[1,:,:], 'sy': S_xyz[2,:,:]}   
    ff_ops= {}
    for op_id,op in S_ops.items():
        #              ----T----
        # <L(O)|= <r_0|--A^+OA--
        #              ----T----
        r0TopT= corrf_c4v.apply_TM_1sO(state,ctm_env_init,r_0,op=op)
        ff_ops[op_id]= torch.zeros(args.top_n,dtype=torch.complex128,device=cfg.global_args.device)
        for i in range(args.top_n):
            #                   ----T----
            # <L(O)|l_i> = <r_0|--A^+OA-- (<r_i|)^+
            #                   ----T----
            ff_ops[op_id][i]= torch.dot(r0TopT.view(-1), vecs[:,i].conj())

    norm_r2= corrf_c4v.apply_TM_1sO(state,ctm_env_init,r_0)
    norm_r2= corrf_c4v.apply_TM_1sO(state,ctm_env_init,norm_r2)
    norm_r2= corrf_c4v.apply_TM_1sO(state,ctm_env_init,norm_r2)
    norm_r2= torch.tensordot(norm_r2,r_0.conj(),([0,1,2],[0,1,2]))
    print(f"norm r=2 {norm_r2.item()}")
    for i in range(args.top_n):
        if i==0: print("FF(r=2) "+" ".join([f"{op_id}{op_id}" for op_id in S_ops.keys()]))
        print(" ".join([f"{ff_ops[op_id][i]*(l[i,0]+1.0j*l[i,1])*ff_ops[op_id][i].conj()/norm_r2}" for op_id in S_ops.keys()]))
        

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestCtmrgBase(unittest.TestCase):
    def setUp(self):
        args.instate=None 
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.GLOBALARGS_device="cpu"
        args.GLOBALARGS_dtype="complex128"

    # basic tests
    def test_ctmrg_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

class TestCtmrgStates(unittest.TestCase):
    tol= 1.0e-6

    def setUp(self):
        import os
        self.DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        args.bond_dim=3
        args.j1, args.j2, args.lmbd= 1., 0, 0
        args.GLOBALARGS_device="cpu"
        args.GLOBALARGS_dtype="complex128"
        args.CTMARGS_ctm_max_iter=200

    # basic tests
    def test_ctmrg_RVB(self):
        args.instate=self.DIR_PATH+"/../../test-input/RVB_1x1.in"
        args.j2=0.5
        args.bond_dim=3
        args.chi=16
        
        cfg.configure(args)
        torch.set_num_threads(args.omp_cores)
        
        model = j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
        energy_f= model.energy_1x1

        state = read_ipeps_c4v(args.instate)

        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            with torch.no_grad():
                if not history:
                    history=[]
                e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
                history.append([e_curr.item()])

                if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                    return True, history
            return False, history

        ctm_env_init = ENV_C4V(args.chi, state)
        init_env(state, ctm_env_init)

        ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

        e_curr0 = energy_f(state, ctm_env_init)
        obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
        obs_dict=dict(zip(obs_labels,obs_values0))

        eps_e=1.0e-8
        eps_m=1.0e-14
        self.assertTrue(abs(e_curr0-(-0.47684229)) < eps_e)
        self.assertTrue(obs_dict["m"] < eps_m)
        for l in ["sz","sp","sm"]:
            self.assertTrue(abs(obs_dict[l]) < eps_m)

    def test_ctmrg_J1J2Lambda(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np
        
        args.instate= self.DIR_PATH+"/../../test-input/"+"BFGS_D3-chi81-PTL0.23423-run0-iD3PT3n0.0_state.json"
        args.out_prefix=f"{args.instate}".replace("BFGS","TEST_J1J2L_BFGS")
        args.j1, args.j2, args.lmbd= 1.7776001555035785, 0.856096294165775, 0.23423
        args.chi=81
                
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
        ref_data= """-0.9108441312441784, 0.07899239704953201, (-0.05711682700606768+0j), 
        (0.05456616959609545-1.1486641093887501e-17j), (0.05456616959609545+1.1486641093887501e-17j), 
        -0.26385027148964796, (0.06540441131450128+0j), (-0.3618962289515061+0j)"""
        fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
        ref_tokens= [complex(x) for x in ref_data.split(",")]
        for val,ref_val in zip(fobs_tokens, ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)
