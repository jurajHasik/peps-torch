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
parser.add_argument("--import_state", type=str, default=None, help="input state for ctmrg")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    t_device = torch.device(args.GLOBALARGS_device)

    # Import all elementary tensors
    tensors_site = []
    tensors_triangle = []
    path = "SU3_CSL_ipeps/SU3_D7_tensors/"
    for name in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'L0', 'L1', 'L2']:
        tens = load_SU3_tensor(path+name)
        tens = tens.to(t_device)
        if name in ['S0', 'S1', 'S2']:
            tensors_triangle.append(1j * tens)
        if name in ['S3', 'S4', 'S5', 'S6']:
            tensors_triangle.append(tens)
        if name in ['L0', 'L1']:
            tensors_site.append(tens)
        if name in ['L2']:
            tensors_site.append(1j * tens)

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
        print(f"\n spectrum C[{spectra[0][0]}]             spectrum C[{spectra[1][0]}]             spectrum C[{spectra[2][0]}]             spectrum C[{spectra[3][0]}] ")
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
        print(f'Step n°{len(history)}    E_site ={e_curr.item()}   (E_up={e_up.item()}, E_dn={e_dn.item()})')
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

    # AKLT state
    coeffs_triangle_1 = {
        (0, 0): torch.tensor([1., 0.12, 0.582, 0., 0., 0., 0.], dtype=torch.float64, device=t_device)}
    coeffs_site_1 = {(0, 0): torch.tensor([1., 0.35, 0.], dtype=torch.float64, device=t_device)}
    # Ji-Yao's state for theta = pi/4
    coeffs_triangle_2 = {
        (0, 0): torch.tensor([1.0000, 0.3563, 4.4882, -0.3494, -3.9341, 0., 0.], dtype=torch.float64,
                             device=t_device)}
    coeffs_site_2 = {(0, 0): torch.tensor([1.0000, 0.2429, 0.], dtype=torch.float64, device=t_device)}

    state1 = IPEPS_U1SYM(tensors_triangle, tensors_site, coeffs_triangle_up=coeffs_triangle_1,
                         coeffs_site=coeffs_site_1)
    state1.print_coeffs()
    state2 = IPEPS_U1SYM(tensors_triangle, tensors_site, coeffs_triangle_up=coeffs_triangle_2,
                         coeffs_site=coeffs_site_2)
    state2.print_coeffs()

    model = SU3_chiral.SU3_CHIRAL(theta=args.theta, j1=args.j1, j2=args.j2)

    ctm_env_init = ENV(args.chi, state1)
    init_prod_overlap(state1, state2, ctm_env_init)

    ctm_env_final, history_O12, *ctm_log = ctmrg.run_overlap(state1, state2,  ctm_env_init, ctmrg_conv_overlap)

    O12 = history_O12[-1]


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
