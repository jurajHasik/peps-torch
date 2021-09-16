import context
import argparse
import config as cfg
import math
import torch
import copy
from collections import OrderedDict
from ipeps.ipess_kagome import IPESS_KAGOME, read_ipess_kagome, extend_bond_dim, to_PG_symmetric
from ipeps.ipeps_kagome import IPEPS_KAGOME, read_ipeps_kagome
from models import spin1_kagome
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.ad_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--j1sq", type=float, default=0, help="nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--j2sq", type=float, default=0, help="next-nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=float, default=0, help="triangle permutation")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=["IPEPS", "IPESS", 'A_2,B'])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= spin1_kagome.S1_KAGOME(j1=args.j1,j1sq=args.j1sq,j2=args.j2,j2sq=args.j2sq,\
        jtrip=args.jtrip,jperm=args.jperm)

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
            B_S= torch.zeros(3, args.bond_dim, args.bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            state= IPESS_KAGOME(T_U, B_S, T_D, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
            state.load_checkpoint(args.opt_resume)
        elif args.ipeps_init_type=='RANDOM':
            bond_dim = args.bond_dim
            T_U= torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            T_D= None if args.sym_up_dn else (torch.rand(bond_dim, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)
            B_S= torch.rand(3, bond_dim, bond_dim,\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
            state = IPESS_KAGOME(T_U, B_S, T_D, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
    
    elif args.ansatz in ["IPEPS"]:    
        ansatz_pgs=None
        if args.instate!=None:
            state= read_ipeps_kagome(args.instate, lX=1, lY=1)

            if args.bond_dim > max(state.get_aux_bond_dims()):
                # extend the auxiliary dimensions
                state= extend_bond_dim(state, args.bond_dim)
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
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn)/3 #+ e_nnn) / 3

    @torch.no_grad()
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

    def post_proc(state, ctm_env, opt_context):
        pass

    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn,
        post_proc=post_proc)

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
