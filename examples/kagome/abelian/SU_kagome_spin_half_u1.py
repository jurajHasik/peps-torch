import os
import warnings
import context
import argparse
import numpy as np
import torch
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
from ipeps.ipess_kagome_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
import ctm.pess_kagome_abelian.rdm_kagome as rdm_kagome
from models.abelian import kagome_u1
from itevol.itebd_ipess_kagome_abelian import itebd 
#from optim.ad_optim_lbfgs_mod import optimize_state
import scipy.io as io
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--JD", type=float, default=0, help="two-spin DM interaction")
parser.add_argument("--j1sq", type=float, default=0, help="nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--j2sq", type=float, default=0, help="next-nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=complex, default=0+0j, help="triangle permutation")
parser.add_argument("--h", type=float, default=0, help="magnetic field")
parser.add_argument("--ansatz", type=str, default="IPESS", help="choice of the tensor ansatz",choices=["IPESS","IPESS_PG","A_2,B"])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',help="same bond site tensors")
parser.add_argument("--disp_corre_len", action='store_true', dest='disp_corre_len',help="display correlation length during optimization")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--force_cpu", action='store_true', dest='force_cpu', help="force RDM contractions on CPU")
parser.add_argument("--itebd_tol", type=float, default=1e-12, help="itebd truncation tol")
parser.add_argument("--no_keep_multiplets", action='store_false', dest='keep_multiplets',help="keep multiplets when performing svd")
# iTEBD schedule - a list of segments [time step dt, total (imag.) time tau to evolve by steps of dt]
# Hence, for a given segment tau/dt steps are executed
parser.add_argument("--SU_schedule", type=str, default="[[0.5,10],[0.1,20],[0.05,10],[0.01,1]]")
# for large bond dimension, observables from CTM approximation
# might be prohibitively expensive. Either reduce the frequency of evaluation
# or skip their computation by setting SU_ctm_obs_freq = 0
# To compute observables only at the end of SU, set SU_ctm_obs_freq = -1
parser.add_argument("--SU_ctm_obs_freq", type=int, default=0)
# iTEBD can be initialized by D=3 RVB state by passing 
# --ipeps_init_type RVB
args, unknown_args = parser.parse_known_args()

@torch.no_grad()
def main(args=args):
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    settings_U1.default_dtype=cfg.global_args.dtype
    settings_U1.default_device=cfg.global_args.device
    
    # 0) initialize model and construct the gate
    if not args.theta is None:
        args.jtrip= args.j1*math.sin(args.theta*math.pi)
        args.j1= args.j1*math.cos(args.theta*math.pi)
    model= kagome_u1.KAGOME_U1(settings_U1, j1=args.j1, JD=args.JD, j1sq=args.j1sq,\
        j2=args.j2, j2sq=args.j2sq, jtrip=args.jtrip, jperm=args.jperm, h=args.h)
    # elementary term of the Hamiltonian
    H=model.h_triangle

    # 1) load or select initial state
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        
        if args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate, settings_U1)
            #
            # TODO allow PGs
            #
            # elif args.ansatz in ["IPESS_PG","A_2,B"]:
            #     state= read_ipess_kagome_pg(args.instate)

            # possibly symmetrize by PG
            # if ansatz_pgs!=None:
            #     if type(state)==IPESS_KAGOME_GENERIC:
            #         state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
            #             SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            #     elif type(state)==IPESS_KAGOME_PG:
            #         if state.pgs==None or state.pgs==dict():
            #             state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
            #                 SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            #         elif state.pgs==ansatz_pgs:
            #             # nothing to do here
            #             pass
            #         elif state.pgs!=ansatz_pgs:
            #             raise RuntimeError("instate has incompatible PG symmetry with "+args.ansatz)
            state= state.add_noise(args.instate_noise)
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
            if args.ansatz in ["IPESS"]:
                state= IPESS_KAGOME_GENERIC(settings_U1, {'T_u': T_u, 'B_a': B_a,\
                    'T_d': T_d, 'B_b': B_b, 'B_c': B_c})
            #
            # TODO allow PGs
            # 
            # elif args.ansatz in ["IPESS_PG", "A_2,B"]:
            #     state= IPESS_KAGOME_PG(T_u, B_c, T_d, T_d=T_d, B_a=B_a, B_b=B_b,\
            #         SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=="RVB":
            args.ansatz= "IPESS"
            unit_block= np.ones((1,1,1), dtype=cfg.global_args.dtype)
            B_c= yastn.Tensor(settings_U1, s=(-1, 1, 1), n=0)
            B_c.set_block(ts=(1,1,0), Ds=unit_block.shape, val= unit_block)
            B_c.set_block(ts=(1,0,1), Ds=unit_block.shape, val= unit_block)
            B_c.set_block(ts=(-1,-1,0), Ds=unit_block.shape, val= unit_block)
            B_c.set_block(ts=(-1,0,-1), Ds=unit_block.shape, val= unit_block)
            B_b=B_c
            B_a=B_c

            unit_block= np.ones((1,1,1), dtype=cfg.global_args.dtype)
            T_u= yastn.Tensor(settings_U1, s=(-1, -1, -1), n=0)
            T_u.set_block(ts=(1,-1,0), Ds=unit_block.shape, val= unit_block)
            T_u.set_block(ts=(-1,1,0), Ds=unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(0,1,-1), Ds=unit_block.shape, val= unit_block)
            T_u.set_block(ts=(0,-1,1), Ds=unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(-1,0,1), Ds=unit_block.shape, val= unit_block)
            T_u.set_block(ts=(1,0,-1), Ds=unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(0,0,0), Ds=unit_block.shape, val= unit_block)
            T_d=T_u
            state= IPESS_KAGOME_GENERIC_ABELIAN(settings_U1, {'T_u': T_u, 'B_a': B_a,\
                'T_d': T_d,'B_b': B_b, 'B_c': B_c})

    print(state)

    # 2) define auxilliary functions
    # 2.1) evaluation of energies
    def energy_f(state, env, force_cpu=False, fail_on_check=False,\
        warn_on_check=True):
        #print(env)
        e_dn = model.energy_down_t_2x2subsystem(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        e_up = model.energy_up_t_2x2subsystem(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f_NoCheck(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        return e_dn

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

    # 2.2) CTM convergence check    
    if args.CTM_check=="Energy":
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu,\
                warn_on_check=False)
            history.append(e_curr.item())
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="Partial_energy":
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            if len(history)>8:
                e_curr = dn_energy_f_NoCheck(state, env, force_cpu=ctm_args.conv_check_cpu)
                history.append(e_curr.item())
            else:
                history.append(len(history)+1)
            #print(history)
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol*2) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="SingularValue":
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

    # 2.3) evaluation of observables
    def obs_fn(state, ctm_env, opt_context):
        state_sym= state
        if args.ansatz in ["IPESS_PG", "A_2,B"]:
            # symmetrization and implicit rebuild of iPEPS on-site tensors
            state_sym= to_PG_symmetric(state, state.pgs)
        elif args.ansatz in ["IPESS"]:
            state_sym.sites= state_sym.build_onsite_tensors()
        
        obs_values, obs_labels = model.eval_obs(state_sym,ctm_env,force_cpu=args.force_cpu,\
            disp_corre_len=args.disp_corre_len)
        return obs_values, obs_labels
        
    # 2.3) recompute CTM, energies, and observables
    def loss_fn(state, ctm_env_in, opt_context={"ctm_args": cfg.ctm_args,\
        "opt_args": cfg.opt_args}):
        
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # build on-site tensors
        if args.ansatz in ["IPESS_PG", "A_2,B"]:
            # explicit rebuild of on-site tensors
            state_sym= to_PG_symmetric(state, state.pgs)
        else:
            state_sym= state
            state_sym.sites= state_sym.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        loss0 = energy_f(state_sym, ctm_env_out, force_cpu=args.force_cpu)
        
        loss= loss0
        return loss, ctm_env_out, history, t_ctm, t_conv_check

    def weight_diff(w1,w2):
        ls1= w1.get_leg_structure(0)
        ls2= w2.get_leg_structure(0)
        diff=0
        for t in set( list(ls1.keys())+list(ls2.keys()) ):
            if t in ls1 and t in ls2:
                if ls1[t]==ls2[t]:
                    diff+= sum(abs(w1[t+t]-w2[t+t])**2)
                elif ls1[t]>ls2[t]:
                    diff+= sum(abs(w1[t+t][:ls2[t]]-w2[t+t])**2)
                else:
                    diff+= sum(abs(w1[t+t]-w2[t+t][:ls1[t]])**2)
            elif t in ls1 and not t in ls2:
                diff+= sum(abs(w1[t+t])**2)
            else: 
                diff+= sum(abs(w2[t+t])**2)
        return diff

    # 3) setup iTEBD
    itebd_list= json.loads(args.SU_schedule)
    
    # arrays to hold observables
    energies=np.array(torch.zeros(len(itebd_list)), dtype='float64')
    m=np.array(torch.zeros(len(itebd_list)), dtype='complex128')
    if args.disp_corre_len:
        correl_len_x=np.array(torch.zeros(len(itebd_list)), dtype='float64')
        correl_len_y=np.array(torch.zeros(len(itebd_list)), dtype='float64')

    H= H.fuse_legs(axes=((0,1,2),(3,4,5)))
    # initialize weights with identity matrices
    lambdas= state.generate_weights()
    # if the minimal bond dimension is smaller, than the desired one, issue warning
    init_min_D= min([ sum(l.get_leg_structure(0).values()) for l in lambdas.values() ])
    if args.bond_dim<init_min_D:
        warnings.warn("Minimal bond dim ("+str(init_min_D)+") of ansatz"\
            +" is larger then args.bond_dim "+str(args.bond_dim), RuntimeWarning)

    # 4) (optional) evaluate initial observables
    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    if args.SU_ctm_obs_freq>0:
        ctm_env, *ctm_log = ctmrg.run(state, ctm_env, \
            conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)
        loss0= energy_f(state, ctm_env, force_cpu=args.force_cpu)
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu,\
            disp_corre_len=args.disp_corre_len)
        print(", ".join(["epoch",f"energy"]+[label for label in obs_labels]))
        print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    # 4) start iTEBD
    print("\niTEBD")
    print(", ".join(["epoch","loss"]), end="")
    if args.SU_ctm_obs_freq>0: 
        print(", "+", ".join([label for label in obs_labels]))
    else:
        print("")
    for ctt in range(len(itebd_list)):
        tau=itebd_list[ctt][1]
        dt=itebd_list[ctt][0]
        lambdas0= { k: v for k,v in lambdas.items() }
        state, lambdas=itebd(state, lambdas, H, args.itebd_tol, tau, dt,\
            args.bond_dim, args.keep_multiplets)

        # evaluate figure of merit from weights only
        delta_lambdas= sum([ weight_diff(lambdas[k],lambdas0[k]) for k in lambdas0.keys() ])
        print(f"{(tau,dt)}, {delta_lambdas}", end="")

        if args.SU_ctm_obs_freq>0 or (args.SU_ctm_obs_freq<0 and itebd_list[ctt]==itebd_list[-1]):
            loss, ctm_env, history, t_ctm, t_conv_check= loss_fn(state, ctm_env)
            obs_values, obs_labels= obs_fn(state, ctm_env, None)
            print(", "+", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))

            energies[ctt]=loss.numpy()
            m[ctt]=(np.sqrt(obs_values[2])+np.sqrt(obs_values[3])+np.sqrt(obs_values[4]))/3
            if args.disp_corre_len:
                correl_len_x[ctt]=obs_values[26]
                correl_len_y[ctt]=obs_values[27]
        else:
            print("")

        # print structure of lambdas
        for l_id,l in lambdas.items():
            print(f"{l_id} {l.get_leg_structure(0)}")

    # 5) save the final state
    # print(state)
    write_ipess_kagome_generic(state, args.out_prefix+"_state.json",\
        peps_args=cfg.peps_args, global_args=cfg.global_args)

    #filenm='IPESS_D_'+str(D)+'_chi_'+str(chi)
    # if abs(args.JD)>0:
    # filenm='IPESS_J'+str(args.j1)+'_JD'+str(args.JD)+'_D'+str(args.bond_dim)+'_chi_'+str(args.chi)
    # filenm=filenm+'.mat'
    # if args.disp_corre_len:
    #     io.savemat(filenm,{'energies':energies,'m':m, 'correl_len_x':correl_len_x, 'correl_len_y':correl_len_y, 'itebd_list':itebd_list})
    # else:
    #     io.savemat(filenm,{'energies':energies,'m':m, 'itebd_list':itebd_list})
    # filenm='IPESS_J'+str(args.j1)+'_JD'+str(args.JD)+'_D'+str(args.bond_dim)+'.json'


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestSU_RVB_Ansatz(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-SU_RVB"

    def setUp(self):
        args.j1= 1.0
        args.theta=0.2
        args.JD=0.0
        args.bond_dim=6
        args.chi=36
        args.instate_noise=0
        args.ipeps_init_type= "RVB"
        args.GLOBALARGS_dtype= "complex128"
        args.SU_ctm_obs_freq= -1
        args.SU_schedule= "[[0.5,10],[0.1,20],[0.05,10],[0.01,1]]"

    def test_su_ipess_rvb(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        # i) run optimization and store the optimization data
        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        final_obs=None
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if "(1, 0.01)" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs
 
        ref_data="""
        (1, 0.01), 0.00024571275573539693, -0.37415279185615224, 
        (-0.5615371279641982+7.681199294136548e-17j), (-0.5609212476042584+0j)
        """
        # compare final observables against expected reference.
        ref_tokens= [complex(x) for x in ref_data.replace('\n','')\
            .strip()[len("(1, 0.01),"):].split(",")]
        fobs_tokens= [complex(x) for x in final_obs[len("(1, 0.01),"):].split(",")]
        for val,ref_val in zip(fobs_tokens[:4], ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)