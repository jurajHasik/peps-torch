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
from ctm.generic import ctmrg
import json
import unittest
import logging
import scipy.io as io
import numpy

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
     choices=["IPEPS", "IPESS", "IPESS_PG", 'A_2,B'])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',\
    help="same bond site tensors")
parser.add_argument("--large_chis", action='store_true', dest='large_chis',\
    help="same trivalent tensors for up and down triangles")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",\
     choices=["Energy", "SingularValue", "Partial_energy"])
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= spin_half_kagome.S_HALF_KAGOME(j1=args.j1,JD=args.JD,j1sq=args.j1sq,j2=args.j2,\
        j2sq=args.j2sq,jtrip=args.jtrip,jperm=args.jperm)

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
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def energy_f_complex(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f(state, env, force_cpu=False):
        #print(env)
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
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

    if args.CTM_check=="Energy":
        def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            e_curr = energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
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
                e_curr = dn_energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
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


    def obs_fn(state, ctm_env, opt_context):
        if args.ansatz in ["IPESS_PG", "A_2,B"]:
            state_sym= to_PG_symmetric(state, state.pgs)
        else:
            state_sym= state
        obs_values, obs_labels = model.eval_obs(state_sym,ctm_env,force_cpu=False)
        return obs_values, obs_labels
    
    #senniu
    chi_origin=args.chi
    ctmargs=cfg.ctm_args
    if args.large_chis:
        chi_set=numpy.array([10,20,40,80,100,120,160])
    else:
        chi_set=numpy.array([10,20,40,80])
    energies=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    e_t_dns=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    e_t_ups=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    m_0=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    m_1=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    m_2=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sz_0=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sp_0=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sm_0=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sz_1=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sp_1=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sm_1=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sz_2=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sp_2=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    sm_2=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_dn_01=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_up_01=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_dn_02=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_up_02=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_dn_12=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    SS_up_12=numpy.array(torch.zeros(numpy.size(chi_set)), dtype='complex128')
    CTMspectras= numpy.empty([4,numpy.size(chi_set)], dtype=object)
    CTMspectra_name= numpy.empty(4, dtype=object)
    for cchi in range(0, numpy.size(chi_set)):
        args.chi=chi_set[cchi]
        ctmargs.chi=chi_set[cchi]
        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)
        ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_init, \
             conv_check=ctmrg_conv_energy, ctm_args=ctmargs)
        spect=print_corner_spectra(ctm_env)
        energies[cchi]=energy_f_complex(state, ctm_env, force_cpu=False).numpy()
        obs_values, obs_labels=obs_fn(state, ctm_env, None)
        e_t_dns[cchi]=obs_values[0]
        e_t_ups[cchi]=obs_values[1]
        m_0[cchi]=numpy.sqrt(obs_values[2])
        m_1[cchi]=numpy.sqrt(obs_values[3])
        m_2[cchi]=numpy.sqrt(obs_values[4])
        sz_0[cchi]=obs_values[5]
        sp_0[cchi]=obs_values[6]
        sm_0[cchi]=obs_values[7]
        sz_1[cchi]=obs_values[8]
        sp_1[cchi]=obs_values[9]
        sm_1[cchi]=obs_values[10]
        sz_2[cchi]=obs_values[11]
        sp_2[cchi]=obs_values[12]
        sm_2[cchi]=obs_values[13]
        SS_dn_01[cchi]=obs_values[14]
        SS_dn_12[cchi]=obs_values[15]
        SS_dn_02[cchi]=obs_values[16]
        SS_up_01[cchi]=obs_values[17]
        SS_up_12[cchi]=obs_values[18]
        SS_up_02[cchi]=obs_values[19]
        for n_corner in range(0,4):
            CTMspectras[n_corner,cchi]=numpy.array([spect[n_corner][1].numpy()])
            CTMspectra_name[n_corner]=spect[n_corner][0]
    D=args.bond_dim
    chi=chi_origin
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        filenm='ob_IPESS_D_'+str(D)+'_chi_'+str(chi)
        # if abs(args.JD)>0:
        #     filenm='ob_IPESS_J'+str(args.j1)+'_JD'+str(args.JD)+'_D'+str(D)+'_chi'+str(chi)
    elif args.ansatz in ["IPEPS"]:
        filenm='ob_IPEPS_D_'+str(D)+'_chi_'+str(chi)
        # if abs(args.JD)>0:
        #     filenm='ob_IPEPS_J'+str(args.j1)+'_JD'+str(args.JD)+'_D'+str(D)+'_chi'+str(chi)
    filenm=filenm+'.mat'
    io.savemat(filenm,{'chi_set':chi_set,'e_t_dns':e_t_dns,'e_t_ups':e_t_ups,'m_0':m_0,'m_1':m_1,'m_2':m_2,\
        'sz_0':sz_0,'sp_0':sp_0,'sm_0':sm_0,'sz_1':sz_1,'sp_1':sp_1,'sm_1':sm_1,'sz_2':sz_2,'sp_2':sp_2,'sm_2':sm_2,\
        'SS_dn_01':SS_dn_01,'SS_dn_12':SS_dn_12,'SS_dn_02':SS_dn_02,'SS_up_01':SS_up_01,'SS_up_12':SS_up_12,'SS_up_02':SS_up_02,\
        'CTMspectras':CTMspectras,'CTMspectra_name':CTMspectra_name,'energies':energies})



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

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()



   
