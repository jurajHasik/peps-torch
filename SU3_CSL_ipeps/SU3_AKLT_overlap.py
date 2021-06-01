import torch
import math
import numpy as np
import config as cfg
import copy
from collections import OrderedDict
from u1sym.ipeps_u1 import IPEPS_U1SYM
from read_write_SU3_tensors import *
from models import SU3_chiral
from ctm.generic.env import *
from ctm.generic import ctmrg, transferops
from ctm.generic import rdm
import json
import unittest
import logging
from tn_interface import view, contiguous

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0., help="angle parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=0., help="nearest-neighbor exchange coupling")
parser.add_argument("--j2", type=float, default=0., help="next-nearest-neighbor exchange coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="frequency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2,
                    help="number of leading eigenvalues of transfer operator to compute")
parser.add_argument("--instate_1", type=str, default=None, help="input state 1")
parser.add_argument("--instate_2", type=str, default=None, help="input state 2")
args, unknown_args = parser.parse_known_args()


def main():

    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    t_device = torch.device(args.GLOBALARGS_device)

    # Import all elementary tensors
    tensors_site_D7 = []
    tensors_triangle_D7 = []
    path = "SU3_CSL_ipeps/SU3_D7_tensors/"
    for name in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'L0', 'L1', 'L2']:
        tens = load_SU3_tensor(path+name)
        tens = tens.to(t_device)
        if name in ['S0', 'S1', 'S2']:
            tensors_triangle_D7.append(1j * tens)
        if name in ['S3', 'S4', 'S5', 'S6']:
            tensors_triangle_D7.append(tens)
        if name in ['L0', 'L1']:
            tensors_site_D7.append(tens)
        if name in ['L2']:
            tensors_site_D7.append(1j * tens)

    # Import all elementary tensors
    tensors_site_D6 = []
    tensors_triangle_D6 = []
    path = "SU3_CSL_ipeps/SU3_D6_tensors/"
    for name in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'L0', 'L1', 'L2', 'L3']:
        tens = load_SU3_tensor(path+name)
        tens = tens.to(t_device)
        if name in ['M4', 'M5']:
            tensors_triangle_D6.append(1j * tens)
        if name in ['M0', 'M1', 'M2', 'M3']:
            tensors_triangle_D6.append(tens)
        if name in ['L0', 'L1', 'L2']:
            tensors_site_D6.append(tens)
        if name == 'L3':
            tensors_site_D6.append(1j * tens)


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
        print(f"spectrum C[{spectra[0][0]}]             spectrum C[{spectra[1][0]}]             spectrum C[{spectra[2][0]}]             spectrum C[{spectra[3][0]}] ")
        for i in range(args.chi):
            print("{:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}".format(i, spectra[0][1][i], i, spectra[1][1][i], i, spectra[2][1][i], i, spectra[3][1][i]))


    def ctmrg_conv_overlap(state1, state2, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        overlap_curr = overlap(state1, state2, env)
        history.append(overlap_curr.item())
        if len(history)==1:
            overlap_prev = 0
        else:
            overlap_prev = history[-2]
        #print_corner_spectra(env)
        print('Step n°{:2}   O12 = {:01.14f}  delta_O12 = {:01.6f}'.format(len(history), overlap_curr.item(), (overlap_curr.item()-overlap_prev)/overlap_curr.item()))
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            print("converged")
            return True, history
        return False, history

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_dn = model.energy_triangle_dn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_nnn = model.energy_nnn(state, env)
        e_curr = (e_up + e_dn + e_nnn) / 3
        history.append(e_curr.item())
        if len(history) == 1:
            e_prev = 0
        else:
            e_prev = history[-2]
        #print_corner_spectra(env)
        print(
            'Step n°{:2}    E_site ={:01.14f}   (E_up={:01.14f}, E_dn={:01.14f}, E_nnn={:01.14f})  delta_E={:01.14f}'.format(
                len(history), e_curr.item(), e_up.item(), e_dn.item(), e_nnn, e_curr.item() - e_prev))
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    def overlap(state1, state2, env):
        O12 = torch.abs(rdm.rdm1x1_id_overlap((0,0), state1, state2, env, force_cpu=True))
        return O12

    def compute_obervables(state, ctm_env_final):
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
        Pnnn = model.P_bonds_nnn(state, ctm_env_final, force_cpu=True)

        # magnetization
        lambda3, lambda8 = model.eval_lambdas(state, ctm_env_final)

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
        print('\n Magnetization')
        print(
            f' Lambda_3 = {torch.real(lambda3[0]).item()}, {torch.real(lambda3[1]).item()}, {torch.real(lambda3[2]).item()}')
        print(
            f' Lambda_8 = {torch.real(lambda8[0]).item()}, {torch.real(lambda8[1]).item()}, {torch.real(lambda8[2]).item()}')

        # environment diagnostics
        print("\n")
        print("Final environment")
        print_corner_spectra(ctm_env_final)

    def build_states_D7():
        if args.instate_1 is not None:
            checkpoint = torch.load(args.instate_1)
            coeffs = checkpoint["parameters"]
            coeffs_triangle_up_1, coeffs_triangle_dn_1, coeffs_site_1 = coeffs[(0, 0)]
            for coeff_t in coeffs_triangle_dn_1.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_triangle_up_1.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_site_1.values(): coeff_t.requires_grad_(False)
            state1 = IPEPS_U1SYM(tensors_triangle_D7, tensors_site_D7,
                                 coeffs_triangle_up=coeffs_triangle_up_1, coeffs_site=coeffs_site_1)
        else:
            # AKLT + small noise
            coeffs_triangle_1 = {
                (0, 0): torch.tensor([1., 0., 0., 0., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            coeffs_site_1 = {(0, 0): torch.tensor([1., 0., 0.], dtype=torch.float64, device=t_device)}
            var_coeffs_site = torch.tensor([0, 1, 0], dtype=torch.float64, device=t_device)
            var_coeffs_triangle = torch.tensor([0, 1, 1, 1, 1, 0, 0], dtype=torch.float64, device=t_device)

            state1 = IPEPS_U1SYM(tensors_triangle_D7, tensors_site_D7,
                                 coeffs_triangle_up=coeffs_triangle_1, coeffs_site=coeffs_site_1,
                                 var_coeffs_site=var_coeffs_site, var_coeffs_triangle=var_coeffs_triangle)
            #state1.add_noise(args.instate_noise)

        if args.instate_2 is not None:
            checkpoint = torch.load(args.instate_2)
            coeffs = checkpoint["parameters"]
            coeffs_triangle_up_2, coeffs_triangle_dn_2, coeffs_site_2 = coeffs[(0, 0)]
            for coeff_t in coeffs_triangle_dn_2.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_triangle_up_2.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_site_2.values(): coeff_t.requires_grad_(False)
            state2 = IPEPS_U1SYM(tensors_triangle_D7, tensors_site_D7,
                                 coeffs_triangle_up=coeffs_triangle_up_2, coeffs_site=coeffs_site_2)
        else:
            # AKLT + small noise
            coeffs_triangle_2 = {
                (0, 0): torch.tensor([ 1.0000, -0.0215,  1.0923, 0., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            coeffs_site_2 = {(0, 0): torch.tensor([1., -0.7737, 0.], dtype=torch.float64, device=t_device)}
            var_coeffs_site = torch.tensor([0, 1, 0], dtype=torch.float64, device=t_device)
            var_coeffs_triangle = torch.tensor([0, 1, 1, 1, 1, 0, 0], dtype=torch.float64, device=t_device)
            state2 = IPEPS_U1SYM(tensors_triangle_D7, tensors_site_D7,
                                 coeffs_triangle_up=coeffs_triangle_2, coeffs_site=coeffs_site_2,
                                 var_coeffs_site=var_coeffs_site, var_coeffs_triangle=var_coeffs_triangle)
            state2.add_noise(args.instate_noise)
        return(state1, state2)



    def build_states_D6():
        if args.instate_1 is not None:
            checkpoint = torch.load(args.instate_1)
            coeffs = checkpoint["parameters"]
            coeffs_triangle_up_1, coeffs_triangle_dn_1, coeffs_site_1 = coeffs[(0, 0)]
            for coeff_t in coeffs_triangle_dn_1.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_triangle_up_1.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_site_1.values(): coeff_t.requires_grad_(False)
            state1 = IPEPS_U1SYM(tensors_triangle_D6, tensors_site_D6,
                                 coeffs_triangle_up=coeffs_triangle_up_1, coeffs_site=coeffs_site_1)
        else:
            # AKLT + small noise
            coeffs_triangle_1 = {(0, 0): torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            coeffs_site_1 = {(0, 0): torch.tensor([1., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            var_coeffs_site = torch.tensor([0, 1, 1, 1], dtype=torch.float64, device=t_device)
            var_coeffs_triangle = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.float64, device=t_device)

            state1 = IPEPS_U1SYM(tensors_triangle_D6, tensors_site_D6,
                                 coeffs_triangle_up=coeffs_triangle_1, coeffs_site=coeffs_site_1,
                                 var_coeffs_site=var_coeffs_site, var_coeffs_triangle=var_coeffs_triangle)
            #state1.add_noise(args.instate_noise)

        if args.instate_2 is not None:
            checkpoint = torch.load(args.instate_2)
            coeffs = checkpoint["parameters"]
            coeffs_triangle_up_2, coeffs_triangle_dn_2, coeffs_site_2 = coeffs[(0, 0)]
            for coeff_t in coeffs_triangle_dn_2.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_triangle_up_2.values(): coeff_t.requires_grad_(False)
            for coeff_t in coeffs_site_2.values(): coeff_t.requires_grad_(False)
            state2 = IPEPS_U1SYM(tensors_triangle_D6, tensors_site_D6,
                                 coeffs_triangle_up=coeffs_triangle_up_2, coeffs_site=coeffs_site_2)
        else:
            # AKLT + small noise
            coeffs_triangle_2 = {(0, 0): torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            coeffs_site_2 = {(0, 0): torch.tensor([1., 0., 0., 0.], dtype=torch.float64, device=t_device)}
            var_coeffs_site = torch.tensor([0, 1, 1, 1], dtype=torch.float64, device=t_device)
            var_coeffs_triangle = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.float64, device=t_device)
            state2 = IPEPS_U1SYM(tensors_triangle_D6, tensors_site_D6,
                                 coeffs_triangle_up=coeffs_triangle_2, coeffs_site=coeffs_site_2,
                                 var_coeffs_site=var_coeffs_site, var_coeffs_triangle=var_coeffs_triangle)
            state2.add_noise(args.instate_noise)
        return(state1, state2)



    state1, state2 = build_states_D7()
    print('\n State 1')
    state1.print_coeffs()
    print('\n State 2')
    state2.print_coeffs()

    model = SU3_chiral.SU3_CHIRAL(Kr = math.cos(args.theta * math.pi/180), Ki=math.sin(args.theta * math.pi/180), j1=args.j1, j2=args.j2)


    # CTMRG for the double-layer 1-2
    print('________________________________________')
    print(' \n CTMRG for the double-layer TN 1-2')
    ctm_env_12 = ENV(args.chi, state1)
    init_prod_overlap(state1, state2, ctm_env_12)
    # is O12 the right convergence criterion ?
    ctm_env_final_12, history_O12, *ctm_log = ctmrg.run_overlap(state1, state2, ctm_env_12, ctmrg_conv_overlap)
    O12 = overlap(state1, state2, ctm_env_final_12)
    print_corner_spectra(ctm_env_final_12)
    print('\n')

    # CTMRG for state 1
    print('________________________________________')
    print(' \n CTMRG for the double-layer TN 1-1')
    ctm_env_1 = ENV(args.chi, state1)
    init_env(state1, ctm_env_1)
    ctm_env_final_1, *ctm_log = ctmrg.run_overlap(state1, state1, ctm_env_1, ctmrg_conv_overlap)
    print('\nObservables state 1')
    compute_obervables(state1, ctm_env_final_1)
    O11 = overlap(state1, state1, ctm_env_final_1)
    print('\n')

    # CTMRG for state 2
    print('________________________________________')
    print(' \n CTMRG for the double-layer TN 2-2')
    ctm_env_2 = ENV(args.chi, state2)
    init_env(state2, ctm_env_2)
    ctm_env_final_2, *ctm_log = ctmrg.run_overlap(state2, state2, ctm_env_2, ctmrg_conv_overlap)
    print('\nObservables state 2')
    compute_obervables(state1, ctm_env_final_2)
    O22 = overlap(state2, state2, ctm_env_final_2)
    print('\n')

    # Compute overlap_12
    overlap_12 = O12**2 / (O11 * O22)
    print('_____________\n Overlap between wavefunctions 1 and 2: {:01.14f}'.format(overlap_12))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
