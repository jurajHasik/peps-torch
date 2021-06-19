import context
import argparse
import config as cfg
import math
import torch
import copy
from collections import OrderedDict
from ipeps.ipeps_kagome import IPEPS_KAGOME, read_ipeps_kagome, extend_bond_dim, to_PG_symmetric
from models import SU3_chiral
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
parser.add_argument("--j1", type=float, default=0, help="nearest-neighbor exchange coupling")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",\
     choices=[None, 'A_2,B'])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',\
    help="same trivalent tensors for up and down triangles")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= SU3_chiral.SU3_CHIRAL(Kr=math.cos(args.theta * math.pi),\
        Ki=math.sin(args.theta * math.pi), j1=args.j1, j2=args.j2)

    # initialize the ipeps
    if args.ansatz=="A_2,B":
        ansatz_pgs= ("A_2", "A_2", "B")
    elif args.ansatz== None or args.ansatz== "":
        ansatz_pgs= None
    if args.instate!=None:
        state= read_ipeps_kagome(args.instate)

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
        state= IPEPS_KAGOME(T_U, B_S, T_D, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        T_U= torch.rand(bond_dim, bond_dim, bond_dim,\
            dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
        T_D= None if args.sym_up_dn else (torch.rand(bond_dim, bond_dim, bond_dim,\
            dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0)
        B_S= torch.rand(3, bond_dim, bond_dim,\
            dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)-1.0
        state = IPEPS_KAGOME(T_U, B_S, T_D, SYM_UP_DOWN=args.sym_up_dn, pgs=ansatz_pgs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")


    def energy_f(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn_v2(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up_v2(state, env, force_cpu=force_cpu)
        e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn + e_nnn) / 3

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
        print(f"\n\nspectrum C[{spectra[0][0]}]             spectrum C[{spectra[1][0]}]             spectrum C[{spectra[2][0]}]             spectrum C[{spectra[3][0]}] ")
        for i in range(args.chi):
            print("{:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}".format(i, spectra[0][1][i], i, spectra[1][1][i], i, spectra[2][1][i], i, spectra[3][1][i]))

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_dn = model.energy_triangle_dn_v2(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_up = model.energy_triangle_up_v2(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_nnn = model.energy_nnn(state, env)
        e_curr = (e_up + e_dn + e_nnn) / 3
        history.append(e_curr.item())
        #print_corner_spectra(env)
        print(f'CTM_step {len(history)} {e_curr.item()} {e_up.item()} {e_dn.item()}')
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    ctm_env_init, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_init, \
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
    
    # evaluate expectation values of SU(3) generators
    print("\n\n",end="")
    model.eval_su3_gens(state, ctm_env_init)


    loss0 = energy_f(state, ctm_env_init, force_cpu=cfg.ctm_args.conv_check_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env_init,force_cpu=False)
    print("\n\n",end="")
    print(", ".join(["epoch",f"loss"]+[label for label in obs_labels]))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))


    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # build on-site tensors
        if state.pgs!=(None,None,None):
            # explicit symmetrize (which rebuilds the on-site tensor)
            state_sym= to_PG_symmetric(state, state.pgs)
        else:
            # explicit rebuild of on-site tensors
            state_sym= state
            state_sym.sites= state_sym.build_onsite_tensors()

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
        state_sym= to_PG_symmetric(state, state.pgs)
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state_sym,ctm_env,force_cpu=False)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn)

    # ctm_env_final, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
    # # energy per site
    # e_dn_final = model.energy_triangle_dn(state, ctm_env_final, force_cpu=True)
    # e_up_final = model.energy_triangle_up(state, ctm_env_final, force_cpu=True)
    # e_nnn_final = model.energy_nnn(state, ctm_env_final, force_cpu=True)
    # e_tot_final = (e_dn_final + e_up_final + e_nnn_final) / 3

    # # P operators
    # P_up = model.P_up(state, ctm_env_final, force_cpu=True)
    # P_dn = model.P_dn(state, ctm_env_final, force_cpu=True)

    # # bond operators
    # Pnn_23, Pnn_13, Pnn_12 = model.P_bonds_nn(state, ctm_env_final)
    # Pnnn = model.P_bonds_nnn(state, ctm_env_final, force_cpu = True)

    # # magnetization
    # lambda3, lambda8 = model.eval_lambdas(state, ctm_env_final)

    # print('\n\n Energy density')
    # print(f' E_up={e_up_final.item()}, E_dn={e_dn_final.item()}, E_tot={e_tot_final.item()}')
    # print('\n Triangular permutations')
    # print(f' Re(P_up)={torch.real(P_up).item()}, Im(P_up)={torch.imag(P_up).item()}')
    # print(f' Re(P_dn)={torch.real(P_dn).item()}, Im(P_dn)={torch.imag(P_dn).item()}')
    # print('\n Nearest-neighbor permutations')
    # print(' P_23={:01.14f} \n P_13={:01.14f} \n P_12={:01.14f}'.format(Pnn_23.item(), Pnn_13.item(), Pnn_12.item()))
    # print('\n Next-nearest neighbor permutations')
    # print(' P_23_a={:01.14f}, P_23_b={:01.14f} \n P_31_a={:01.14f}, P_31_b={:01.14f} \n P_12_a={:01.14f}, '
    #       'P_12_b={:01.14f}'.format(Pnnn[4].item(), Pnnn[5].item(), Pnnn[0].item(), Pnnn[1].item(), Pnnn[2].item(),
    #                                 Pnnn[3].item()))
    # print('\n Magnetization')
    # print(
    #     f' Lambda_3 = {torch.real(lambda3[0]).item()}, {torch.real(lambda3[1]).item()}, {torch.real(lambda3[2]).item()}')
    # print(
    #     f' Lambda_8 = {torch.real(lambda8[0]).item()}, {torch.real(lambda8[1]).item()}, {torch.real(lambda8[2]).item()}')

    # # environment diagnostics
    # print("\n")
    # print("Final environment")
    # print_corner_spectra(ctm_env_final)



if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
