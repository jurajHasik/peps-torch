import os
import warnings
import context
import argparse
import config as cfg
import math
import torch
import copy
from collections import OrderedDict
from ipeps.ipeps import IPEPS_WEIGHTED
from ipeps.ipess_kagome import *
from ipeps.ipeps_kagome import *
from models import spin_half_kagome
from ctm.generic.env import *
from ctm.pess_kagome import rdm_kagome
from ctm.generic import ctmrg
from ctm.generic import transferops
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
parser.add_argument("--jperm", type=complex, default=0+0j, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", "IPESS_PG", "A_2,B", "A_1,B"])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',\
    help="same bond site tensors")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",\
     choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--force_cpu", action='store_true', dest='force_cpu', help="force RDM contractions on CPU")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues "+
    "of transfer operator to compute")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--EH_n", type=int, default=1, help="number of leading eigenvalues "+
    "of EH to compute")
parser.add_argument("--EH_T_ED_L", type=int, default=0, help="max. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by ED")
parser.add_argument("--EH_T_ARP_minL", type=int, default=1, help="min. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by Arnoldi")
parser.add_argument("--EH_T_ARP_maxL", type=int, default=0, help="max. cylinder width "+
    "of EH constructed as T-tensor MPO and diagionalized by Arnoldi")
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
    model= spin_half_kagome.S_HALF_KAGOME(j1=args.j1,JD=args.JD,j2=args.j2,\
        jtrip=args.jtrip,jperm=args.jperm)

    # initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B","A_1,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        if args.ansatz=="A_1,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A1_B
        if ansatz_pgs in ["A_1,B", "A_2,B"]: 
            args.sym_bond_S= True
            args.sym_up_dn= True

        if args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate)
            elif args.ansatz in ["IPESS_PG","A_2,B","A_1,B"]:
                try: 
                    state= read_ipess_kagome_pg(args.instate)
                except Exception as e:
                    print(e)
                    warnings.warn(f"Attempting LC ansatz")
                    state= read_ipess_kagome_pg_lc(args.instate)

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
            if args.ansatz in ["IPESS_PG", "A_2,B", "A_1,B"]:
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
            if args.ansatz in ["IPESS_PG", "A_2,B", "A_1,B"]:
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


    def energy_f(state, env, force_cpu=False, fail_on_check=False,\
        warn_on_check=True):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=args.force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        e_up = model.energy_triangle_up(state, env, force_cpu=args.force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def energy_f_complex(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=args.force_cpu)
        e_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=args.force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=args.force_cpu)
        return e_dn
 

    def report_conv_fn(state, env, conv_step, conv_crit, e_curr=None):
        if args.obs_freq>0 and \
            (conv_step%args.obs_freq==0 or (conv_step-1)%args.obs_freq==0):
            if e_curr is None: 
                e_curr = energy_f(state, env, force_cpu=args.force_cpu, warn_on_check=False)
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
                float('inf') if len(history) < 2 else abs(history[-1] - history[-2]), e_curr=e_curr)

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
            _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)

            # log conv_crit and observables
            report_conv_fn(state,env,len(history['diffs']),history['conv_crit'][-1])

            return _conv_check, history

    # gauge, operates only IPEPS base and its sites tensors
    if args.gauge and args.ansatz=="IPEPS":
        state_g= IPEPS_WEIGHTED(state=state).gauge()
        state_g= state_g.absorb_weights()
        state= IPEPS_KAGOME(sites=state_g.sites, vertexToSite=state.vertexToSite, \
            lX=state.lX, lY=state.lY,\
            peps_args=cfg.peps_args, global_args=cfg.global_args)
    
    # 3) initialize environment 
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    # 4) (optional) compute observables as given by initial environment
    e_curr0 = energy_f(state, ctm_env_init, force_cpu=args.force_cpu, warn_on_check=False)
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

    norm_1x1= rdm_kagome.trace1x1_dn_kagome((0,0), state, ctm_env_init,\
        model.Id3_t.view([model.phys_dim]*6), force_cpu=args.force_cpu)
    print(f"Norm 1x1_dn {norm_1x1}")
    ev_chi_R_downT= rdm_kagome.trace1x1_dn_kagome((0,0), state, ctm_env_init,\
        chi_R, force_cpu=args.force_cpu)/norm_1x1
    ev_chi_I_downT= rdm_kagome.trace1x1_dn_kagome((0,0), state, ctm_env_init,\
        chi_I, force_cpu=args.force_cpu)/norm_1x1
    print(f"trace1x1_dn_kagome Re(Chi) downT {ev_chi_R_downT} Im(Chi) downT {ev_chi_I_downT}")

    ev_chi_R_downT,norm_2x2_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, ctm_env_init,\
        chi_R, force_cpu=args.force_cpu)
    ev_chi_I_downT,_= rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, ctm_env_init,\
        chi_I, force_cpu=args.force_cpu)
    print(f"Norm 2x2_dn {norm_2x2_dn}")
    print(f"Re(Chi) downT {ev_chi_R_downT} Im(Chi) downT {ev_chi_I_downT}")

    rho_upT= rdm_kagome.rdm2x2_up_triangle_open((0,0), state, ctm_env_init,\
        force_cpu=args.force_cpu)
    ev_chi_R_upT= torch.einsum('ijkmno,mnoijk',rho_upT, chi_R)
    ev_chi_I_upT= torch.einsum('ijkmno,mnoijk',rho_upT, chi_I)
    print(f"Re(Chi) upT {ev_chi_R_upT} Im(Chi) upT {ev_chi_I_upT}")

    # 7) ----- additional observables ---------------------------------------------    
    for x in [0,1,2]:
        corrSS= model.eval_corrf_SS((0,0), (1,0), state, ctm_env_init, args.corrf_r, site=x)
        print(f"\n\nSS[(0,0),(1,0),site={x}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

        corrSS= model.eval_corrf_SS((0,0), (0,1), state, ctm_env_init, args.corrf_r, site=x)
        print(f"\n\nSS[(0,0),(0,1),site={x}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)), ((0,0), (0,1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

        print(f"\n\nspectrum(T_w0)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_w0_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

    # entanglement spectrum
    site_dir_list=[((0,0), (1,0)), ((0,0), (0,1))]
    for sdp in site_dir_list:

        for L in range(1,args.EH_T_ED_L+1):
            S= transferops.get_full_EH_spec_Ttensor(L, *sdp, state, ctm_env_init)
            print(f"\nEH_T_ED[{sdp[0]},{sdp[1]}] L={L}")
            for i in range(min(S.size(0),args.EH_n)):
                print(f"{i} {S.real[i]} {S.imag[i]}")

        for L in range(args.EH_T_ARP_minL,args.EH_T_ARP_maxL+1):
            S=transferops.get_EH_spec_Ttensor(args.EH_n, L, *sdp, state, ctm_env_init)
            if S is None: continue

            print(f"\nEH_T_ARP[{sdp[0]},{sdp[1]}] L={L}")
            for i in range(args.EH_n):
                print(f"{i} {S[i,0]} {S[i,1]}")

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg_IPESS_D3_RVB(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-rvb"
    ANSATZE= [("IPESS","IPESS_KAGOME_D3_RVB.in"),\
        ("IPESS_PG","IPESS_PG_KAGOME_D3_RVB.in")]

    def setUp(self):
        args.j1= 1.0
        args.bond_dim=3
        args.chi=18
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_ipess_ansatze_d3_rvb(self):
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
                -0.3931221584692804, (-0.5896832690555696+0j), (-0.5896832063522717+0j), (7.59716523160245e-32+0j), 
                (-4.331814810151939e-31+0j), (8.592175414886632e-32+0j), (3.07218410812194e-16+0j), 
                (-3.674157727896386e-17+0j), (5.011080358949883e-16+0j), (-3.953569086037768e-16+0j), 
                (6.82165674627862e-16+0j), (-8.641428147458597e-16+0j), (-1.0617693286257969e-16+0j), 
                (-9.980184244909592e-16+0j), (-7.479642784634561e-17+0j), (-0.19656089027856907+0j), 
                (-0.19656149813919332+0j), (-0.1965608806378064+0j), (-0.19656141466722352+0j), 
                (-0.19656089010487604+0j), (-0.19656090158017214+0j)
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

   
