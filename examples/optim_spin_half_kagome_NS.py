import context
import argparse
import config as cfg
import math
import torch
import copy
from collections import OrderedDict
from ipeps.ipess_kagome_NS import IPESS_KAGOME, read_ipess_kagome, extend_bond_dim, to_PG_symmetric
from ipeps.ipeps_kagome import IPEPS_KAGOME, read_ipeps_kagome
from models import spin_half_kagome_NS
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.ad_optim_lbfgs_mod_NS import optimize_state
import json
import unittest
import logging
import scipy.io as io
import numpy
import time

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
parser.add_argument("--jperm", type=float, default=0, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", 'A_2,B'])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',\
    help="same bond site tensors")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",\
     choices=["Energy", "SingularValue", "Partial_energy"])
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= spin_half_kagome_NS.S1_KAGOME(j1=args.j1, JD=args.JD, j1sq=args.j1sq,j2=args.j2,j2sq=args.j2sq,\
        jtrip=args.jtrip,jperm=args.jperm)

    #sen niu
    def extend_bond_dim_PEPS(state, bond_dim):
        Atensor=state.sites[(0,0)]
        Atensor=Atensor/(torch.norm(Atensor))
        sz=torch.Tensor.size(Atensor)
        if sz[1]<bond_dim:
            R=torch.rand([sz[0],bond_dim,bond_dim,bond_dim,bond_dim], dtype=Atensor.dtype, device=Atensor.device)
            R=R/(torch.norm(R))
            A_new=torch.zeros(sz[0],bond_dim,bond_dim,bond_dim,bond_dim, dtype=Atensor.dtype, device=Atensor.device)
            A_new[:,0:sz[1],0:sz[1],0:sz[1],0:sz[1]]=Atensor
            A_new=A_new+R*0.01
        else:
            A_new=Atensor
        state.sites[(0,0)]=A_new
        return state

    # initialize the ipeps
    if args.ansatz in ["IPESS","A_2,B"]:
        ansatz_pgs=None
        if args.ansatz=="A_2,B": ansatz_pgs= ("A_2", "A_2", "B")
    
        if args.instate!=None:
            state= read_ipess_kagome(args.instate)

            # possibly symmetrize by PG
            if ansatz_pgs!=None and state.pgs==(None,None,None):
                state= to_PG_symmetric(state, ansatz_pgs)
            elif ansatz_pgs!=None and state.pgs==ansatz_pgs:
                pass
            elif ansatz_pgs!=None and state.pgs!=ansatz_pgs:
                raise RuntimeError("instate has incompatible PG symmetry with "+args.ansatz)

            if args.bond_dim > max(state.get_aux_bond_dims()):
                # extend the auxiliary dimensions
                state= extend_bond_dim(state, args.bond_dim)
            state.add_noise(args.instate_noise)
        elif args.opt_resume is not None:
            T_U= torch.zeros(args.bond_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            T_D= None if args.sym_up_dn else (torch.zeros(args.bond_dim, args.bond_dim,\
                args.bond_dim, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)
            B_S1= torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            B_S2= None if args.sym_bond_S else (torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))  
            B_S3= None if args.sym_bond_S else (torch.zeros(model.phys_dim, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))
            state= IPESS_KAGOME(T_U, B_S1, T_D, B_S2, B_S3, SYM_UP_DOWN=args.sym_up_dn, SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=='RANDOM':
            bond_dim = args.bond_dim
            T_U= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            T_D= None if args.sym_up_dn else (torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)
            B_S1= torch.rand(model.phys_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            B_S2= None if args.sym_bond_S else (torch.rand(model.phys_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)  
            B_S3= None if args.sym_bond_S else (torch.rand(model.phys_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)

            state = IPESS_KAGOME(T_U, B_S1, T_D, B_S2, B_S3, SYM_UP_DOWN=args.sym_up_dn, SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            
    elif args.ansatz in ["IPEPS"]:    
        ansatz_pgs=None
        if args.instate!=None:
            state= read_ipeps_kagome(args.instate)

            if args.bond_dim > max(state.get_aux_bond_dims()):
                # extend the auxiliary dimensions
                #state= extend_bond_dim(state, args.bond_dim)
                #sen niu:
                state= extend_bond_dim_PEPS(state, args.bond_dim)
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
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

        

    def energy_f(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def energy_f_NoCheck(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f_NoCheck(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        return e_dn

    @torch.no_grad()
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

    if args.CTM_check=="Energy":
        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            e_curr = energy_f_NoCheck(state, env, force_cpu=ctm_args.conv_check_cpu)
            history.append(e_curr.item())
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="Partial_energy":
        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
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
        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
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
        if args.ansatz in ["IPESS", "A_2,B"]:
            # explicit rebuild of on-site tensors
            state_sym= to_PG_symmetric(state, state.pgs)
            state_sym.sites= state_sym.build_onsite_tensors()
        else:
            A= state.sites[(0,0)]
            A= A/A.abs().max()
            state_sym= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        loss0 = energy_f(state_sym, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        # loc_ctm_args = copy.deepcopy(ctm_args)
        # loc_ctm_args.ctm_max_iter = 1
        # ctm_env_out, history1, t_ctm1, t_obs1 = ctmrg.run(state, ctm_env_out, ctm_args=loc_ctm_args)
        # loss1 = energy_f(state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)
        # loss = torch.max(loss0, loss1)
        
        loss= loss0
        return loss, ctm_env_out, history, t_ctm, t_conv_check

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if args.ansatz in ["IPESS", "A_2,B"]:
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

        # print time
        #print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 


        # 格式化成2016-03-20 11:45:39形式
        #print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 

        # #senniu
        # chi=torch.Tensor.size(ctm_env.T[((0,0), (-1, 0))])[0]
        # D=round((torch.Tensor.size(ctm_env.T[((0,0), (-1, 0))])[2])**0.5)
        # if args.ansatz in ["IPESS"]:
        #     filenm='IPESS_D_'+str(D)+'_chi_'+str(chi)
        # elif args.ansatz in ["IPEPS"]:
        #     filenm='IPEPS_D_'+str(D)+'_chi_'+str(chi)
        # filenm=filenm+'.mat'
        # #print(filenm)
        # #print(type(state))
        # #print(type(state.sites))
        # #print(type(state.sites[(0,0)]))
        # #print(D)
        # #print(chi)
        # #print(ctm_env)
        # Cmm=ctm_env.C[((0,0),(-1,-1))].numpy()
        # Cmp=ctm_env.C[((0,0),(-1,1))].numpy()
        # Cpm=ctm_env.C[((0,0),(1,-1))].numpy()
        # Cpp=ctm_env.C[((0,0),(1,1))].numpy()
        # T0p=ctm_env.T[((0,0),(0,1))].numpy()
        # Tp0=ctm_env.T[((0,0),(1,0))].numpy()
        # T0m=ctm_env.T[((0,0),(0,-1))].numpy()
        # Tm0=ctm_env.T[((0,0),(-1,0))].numpy()
        # Atensor=state.sites[(0,0)].numpy()
        # io.savemat(filenm,{'chi':chi,'D':D,'Atensor':Atensor,
        #        'Cmm':Cmm,'Cmp':Cmp,'Cpm':Cpm,'Cpp':Cpp,'T0m':T0m,'T0p':T0p,'Tm0':Tm0,'Tp0':Tp0,
        #        obs_labels[0]:obs_values[0].numpy(),
        #        obs_labels[1]:obs_values[1].numpy(),
        #        obs_labels[2]:obs_values[2],
        #        obs_labels[3]:obs_values[3],
        #        obs_labels[4]:obs_values[4],
        #        obs_labels[5]:obs_values[5].numpy(),
        #        obs_labels[6]:obs_values[6].numpy(),
        #        obs_labels[7]:obs_values[7].numpy(),
        #        obs_labels[8]:obs_values[8].numpy(),
        #        obs_labels[9]:obs_values[9].numpy(),
        #        obs_labels[10]:obs_values[10].numpy(),
        #        obs_labels[11]:obs_values[11].numpy(),
        #        obs_labels[12]:obs_values[12].numpy(),
        #        obs_labels[13]:obs_values[13].numpy(),
        #        obs_labels[14]:obs_values[14].numpy(),
        #        obs_labels[15]:obs_values[15].numpy(),
        #        obs_labels[16]:obs_values[16].numpy(),
        #        obs_labels[17]:obs_values[17].numpy(),
        #        obs_labels[18]:obs_values[18].numpy(),
        #        obs_labels[19]:obs_values[19].numpy()})
        

    def post_proc(state, ctm_env, opt_context):
        pass
    
    print('optimization start')

    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn,
        post_proc=post_proc)

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
