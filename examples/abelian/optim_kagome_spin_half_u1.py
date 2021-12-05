import context
import argparse
import numpy as np
import torch
import config as cfg
import examples.abelian.settings_U1_torch as settings_U1
from ipeps.ipeps_kagome_abelian import *
from ipeps.ipess_kagome_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
import ctm.pess_kagome_abelian.rdm_kagome as rdm_kagome
import models.abelian.kagome_spin_half_u1 as model
from optim.ad_optim_lbfgs_mod import optimize_state
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
parser.add_argument("--jperm", type=float, default=0, help="triangle permutation")
parser.add_argument("--h", type=float, default=0, help="magnetic field")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",choices=["IPEPS", "IPESS", "IPESS_PG", 'A_2,B'])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',help="same bond site tensors")
parser.add_argument("--disp_corre_len", action='store_true', dest='disp_corre_len',help="display correlation length during optimization")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--itebd_tol", type=float, default=1e-12, help="itebd truncation tol")
parser.add_argument("--initial_RVB", type=float, default=0,help="D=3 RVB state")
args, unknown_args = parser.parse_known_args()

@torch.no_grad()
def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    settings_U1.default_dtype=args.GLOBALARGS_dtype
    settings_U1.default_device=args.GLOBALARGS_device
    model_u1= model.KAGOME_U1(settings_U1, j1=args.j1, JD=args.JD, j1sq=args.j1sq, j2=args.j2, j2sq=args.j2sq, jtrip=args.jtrip, jperm=args.jperm, h=args.h)
    H=model_u1.h_triangle
    #print(H)


    # import config_U1
    # import config_Z2
    # from yamps.yast.tensor._output import show_properties, to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
    # a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),\
    #               t=((0, 1), (0, 1), (0, 1), (0, 1)),\
    #               D=((1, 1), (1, 1), (1, 1), (1, 1)))
    #print(a)
    #import pdb; pdb.set_trace()
    



    # initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B
        

        if args.initial_RVB==1:
            
            unit_block= np.ones((1,1,1), dtype=args.GLOBALARGS_dtype)
            B_c= yast.Tensor(settings_U1, s=(-1, 1, 1), n=0)
            B_c.set_block(ts=(1,1,0), val= unit_block)
            B_c.set_block(ts=(1,0,1), val= unit_block)
            B_c.set_block(ts=(-1,-1,0), val= unit_block)
            B_c.set_block(ts=(-1,0,-1), val= unit_block)
            B_b=B_c
            B_a=B_c

            unit_block= np.ones((1,1,1), dtype=args.GLOBALARGS_dtype)
            T_u= yast.Tensor(settings_U1, s=(-1, -1, -1), n=0)
            T_u.set_block(ts=(1,-1,0), val= unit_block)
            T_u.set_block(ts=(-1,1,0), val= -1*unit_block)
            T_u.set_block(ts=(0,1,-1), val= unit_block)
            T_u.set_block(ts=(0,-1,1), val= -1*unit_block)
            T_u.set_block(ts=(-1,0,1), val= unit_block)
            T_u.set_block(ts=(1,0,-1), val= -1*unit_block)
            T_u.set_block(ts=(0,0,0), val= unit_block)
            T_d=T_u
            state= IPESS_KAGOME_GENERIC_ABELIAN(settings_U1, {'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                'B_b': B_b, 'B_c': B_c})
            
            #import pdb; pdb.set_trace()
        elif args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate, settings_U1)
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

            # if args.bond_dim > state.get_aux_bond_dims():
            #     # extend the auxiliary dimensions
            #     state= state.extend_bond_dim(args.bond_dim)
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

            #su(2) sectors
            if bond_dim==3:
                B_c = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))
                B_b = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))
                B_a = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))

                T_u = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1, 1), (1, 1, 1), (1, 1, 1)))
                T_d = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1, 1), (1, 1, 1), (1, 1, 1)))
            if bond_dim==6:
                B_c = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                B_b = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                B_a = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))

                T_u = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 2, 1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                T_d = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 2, 1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
            if bond_dim==8:
                B_c = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                B_b = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                B_a = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))

                T_u = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 2, 2, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                T_d = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 2, 2, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
            if bond_dim==9:
                B_c = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                B_b = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                B_a = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))

                T_u = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 3, 2, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                T_d = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 3, 2, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))            

            #non-su(2) sectors
            elif bond_dim==4:
                B_c = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))
                B_b = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))
                B_a = yast.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))

                T_u = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 2, 1), (1, 2, 1), (1, 2, 1)))
                T_d = yast.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 2, 1), (1, 2, 1), (1, 2, 1)))
            state= IPESS_KAGOME_GENERIC_ABELIAN(settings_U1, {'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                'B_b': B_b, 'B_c': B_c})


    def energy_f(state, env, force_cpu=False):
        #print(env)
        e_dn = model_u1.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model_u1.energy_triangle_up(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        #print(e_dn)
        #print(e_up)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    
    @torch.no_grad()
    def energy_f_NoCheck(state, env, force_cpu=False):
        #print(env)
        e_dn = model_u1.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        e_up = model_u1.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3
    def dn_energy_f_NoCheck(state, env, force_cpu=False):
        #print(env)
        e_dn = model_u1.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
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


    ctm_env_init= ENV_ABELIAN(args.chi, state=state, init=True)
    ctm_env_init, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)

    # loss0 = energy_f_NoCheck(state, ctm_env_init, force_cpu=cfg.ctm_args.conv_check_cpu).real
    # obs_values, obs_labels = model_u1.eval_obs(state,ctm_env_init,force_cpu=False, disp_corre_len=args.disp_corre_len)
    # print("\n\n",end="")
    # print(", ".join(["epoch",f"loss"]+[label for label in obs_labels]))
    # print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    
    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # build on-site tensors
        if args.ansatz in ["IPESS", "IPESS_PG", "A_2,B"]:
            if args.ansatz in ["IPESS_PG", "A_2,B"]:
                # explicit rebuild of on-site tensors
                state_sym= to_PG_symmetric(state, state.pgs)
            else:
                state_sym= state
            # include normalization of new on-site tensor
            state_sym.sites= state_sym.build_onsite_tensors()
        else:
            A= state.sites[(0,0)]
            A= A/A.abs().max()
            state_sym= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)

        # build double-layer open on-site tensors
        state.build_sites_dl_open()

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
        if args.ansatz in ["A_2,B"]:
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
        if opt_context["line_search"]:
            print(", ".join([f"{epoch}",f"{loss}"]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))
        else:
            obs_values, obs_labels = model_u1.eval_obs(state,ctm_env_init,force_cpu=False, disp_corre_len=args.disp_corre_len)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]), end="")
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))
            print(" "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]) )


        
    # optimize
    ctm_env_init= ENV_ABELIAN(args.chi, state=state, init=True)
    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn)
    
    

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
        
    main()
