import context
import argparse
import config as cfg
import math
import torch
import copy
from random import randint
from collections import OrderedDict
from u1sym.ipeps_u1 import IPEPS_U1SYM, write_coeffs, read_coeffs
from read_write_SU3_tensors import *
from models import SU3_chiral
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.fd_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--frac_theta", type=float, default=0., help="angle parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=0., help="nearest-neighbor exchange coupling")
parser.add_argument("--j2", type=float, default=0., help="next-nearest-neighbor exchange coupling")
parser.add_argument("--ansatz", type=str, default="A1+iA2, B", help="choice of the tensor ansatz")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    t_device = torch.device(args.GLOBALARGS_device)

    # Import all elementary tensors
    elementary_tensors = []
    for name in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'L0', 'L1', 'L2']:
        tens = load_SU3_tensor(name)
        tens = tens.to(t_device)
        if name in ['S0', 'S1', 'S2', 'L2']:
            elementary_tensors.append(1j * tens)
        else:
            elementary_tensors.append(tens)

    # define initial coefficients and which coefficients will vary / will be added a noise
    if args.ansatz == "A1+iA2, B":
        #coeffs = {(0, 0): torch.tensor([1.0000,  0.3563,  4.4882, -0.3494, -3.9341, 0., 0., 1.0000, 0.2429, 0.], dtype=torch.float64, device=t_device)}
        coeffs = {(0, 0): torch.tensor([1., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=torch.float64, device=t_device)}
        var_coeffs_allowed = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 1, 0], dtype=torch.float64, device=t_device)
    elif args.ansatz == "A1+iA2, B+iA":
        coeffs = {(0, 0): torch.tensor([1., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=torch.float64, device=t_device)}
        var_coeffs_allowed = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 1, 1], dtype=torch.float64, device=t_device)
    elif args.ansatz == "E1, B":
        coeffs = {(0, 0): torch.tensor([0., 0., 0., 0., 0., 1., 0., 1., 0., 0.], dtype=torch.float64, device=t_device)}
        var_coeffs_allowed = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float64, device=t_device)
    elif args.ansatz == "E1, B+iA":
        coeffs = {(0, 0): torch.tensor([0., 0., 0., 0., 0., 1., 0., 1., 0., 0.], dtype=torch.float64, device=t_device)}
        var_coeffs_allowed = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=torch.float64, device=t_device)

    state = IPEPS_U1SYM(elementary_tensors, coeffs, var_coeffs_allowed)
    state.add_noise(args.instate_noise)
    print(f'Current state: {state.coeffs[(0, 0)].data}')

    model = SU3_chiral.SU3_CHIRAL(theta=math.pi * args.frac_theta / 100.0, j1=args.j1, j2=args.j2)

    def energy_f(state, env, force_cpu=False):
        e_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=force_cpu)
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
        e_dn = model.energy_triangle_dn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_nnn = model.energy_nnn(state, env)
        e_curr = (e_up + e_dn + e_nnn) / 3
        history.append(e_curr.item())
        #print_corner_spectra(env)
        print(f'Step n°{len(history)}    E_site ={e_curr.item()}   (E_up={e_up.item()}, E_dn={e_dn.item()})')
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]
        # build on-site tensors
        state.sites = state.build_onsite_tensors()
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
        loss0 = energy_f(state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        loc_ctm_args = copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter = 1
        ctm_env_out, history1, t_ctm1, t_obs1 = ctmrg.run(state, ctm_env_out, ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        timings = (t_ctm+t_ctm1, t_obs+t_obs1)
        loss = torch.max(loss0, loss1)
        return loss, ctm_env_out, history, timings

    optimize_state(state, ctm_env_init, loss_fn)
    ctm_env_final, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # energy per site
    e_dn_final = model.energy_triangle_dn(state, ctm_env_final, force_cpu=True)
    e_up_final = model.energy_triangle_up(state, ctm_env_final, force_cpu=True)
    e_nnn_final = model.energy_nnn(state, ctm_env_final, force_cpu=True)
    e_tot_final = (e_dn_final + e_up_final + e_nnn_final) / 3

    # P operators
    P_up = model.P_up(state, ctm_env_final, force_cpu=True)
    P_dn = model.P_dn(state, ctm_env_final, force_cpu=True)

    # bond operators
    Pnn_23, Pnn_13, Pnn_12 = model.P_bonds_nn(state, ctm_env_final)
    Pnnn = model.P_bonds_nnn(state, ctm_env_final, force_cpu = True)

    print('\n\n Energy density')
    print(f' E_up={e_up_final.item()}, E_dn={e_dn_final.item()}, E_tot={e_tot_final.item()}')
    print('\n Triangular permutations')
    print(f' Re(P_up)={torch.real(P_up).item()}, Im(P_up)={torch.imag(P_up).item()}')
    print(f' Re(P_dn)={torch.real(P_dn).item()}, Im(P_dn)={torch.imag(P_dn).item()}')
    print('\n Nearest-neighbor permutations')
    print(' P_23={:01.14f} \n P_13={:01.14f} \n P_12={:01.14f}'.format(Pnn_23.item(), Pnn_13.item(), Pnn_12.item()))
    print('\n Next-nearest neighbor permutations')
    print(' P_23_a={:01.14f}, P_23_b={:01.14f} \n P_31_a={:01.14f}, P_31_b={:01.14f} \n P_12_a={:01.14f}, '
          'P_12_b={:01.14f}'.format(Pnnn[4].item(), Pnnn[5].item(), Pnnn[0].item(), Pnnn[1].item(), Pnnn[2].item(),
                                    Pnnn[3].item()))

    colors3, colors8 = model.eval_lambdas(state, ctm_env_final)
    print(
        f'\n Lambda_3 = {torch.real(colors3[0]).item()}, {torch.real(colors3[1]).item()}, {torch.real(colors3[2]).item()}')
    print(
        f' Lambda_8 = {torch.real(colors8[0]).item()}, {torch.real(colors8[1]).item()}, {torch.real(colors8[2]).item()}')

    # environment diagnostics
    print("\n")
    print("Final environment")
    print_corner_spectra(ctm_env_final)



if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
