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

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0., help="angle, in degrees, parametrizing the ratio K/J1")
parser.add_argument("--phi", type=float, default=0., help="angle, in degrees, parametrizing the ratio J2/K")
parser.add_argument("--C", type=float, default=0., help="amplitude/sign of the J2 curve")
parser.add_argument("--npts", type=int, default=9, help="number of points along each direction in parameter space")
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
    for name in ['S0', 'S1', 'S2', 'L0', 'L1']:
        tens = load_SU3_tensor(path+name)
        tens = tens.to(t_device)
        if name in ['S0', 'S1', 'S2']:
            tensors_triangle.append(tens)
        elif name in ['L0', 'L1']:
            tensors_site.append(tens)

    model = SU3_chiral.SU3_CHIRAL(Kr=math.sin(args.theta * math.pi/180) * math.cos(args.phi/2 * math.pi/180), Ki=0., j1=math.cos(args.theta * math.pi/180), j2=args.C * math.sin(args.phi *math.pi/180))


    def energy_f(state, env):
        e_dn = model.energy_triangle_dn(state, env, force_cpu=True)
        e_up = model.energy_triangle_up(state, env, force_cpu=True)
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
            spectra.append([label, s/s[0]])
        print(f"\n spectrum C[{spectra[0][0]}]             spectrum C[{spectra[1][0]}]             spectrum C[{spectra[2][0]}]             spectrum C[{spectra[3][0]}] ")
        for i in range(args.chi):
            print("{:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}".format(i, spectra[0][1][i], i, spectra[1][1][i], i, spectra[2][1][i], i, spectra[3][1][i]))


    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_dn = model.energy_triangle_dn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_up = model.energy_triangle_up(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_nnn = model.energy_nnn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_curr = (e_up + e_dn + e_nnn)/3
        history.append(e_curr.item())
        if len(history)==1:
            e_prev = 0
        else:
            e_prev = history[-2]
        #print_corner_spectra(env)
        print('Step n°{:2}    E_site ={:01.14f}   (E_up={:01.14f}, E_dn={:01.14f}, E_nnn={:01.14f})  delta_E={:01.14f}'.format(len(history), e_curr.item(), e_up.item(), e_dn.item(), e_nnn, e_curr.item()-e_prev))
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    def ctmrg_conv_corners(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_dn = torch.tensor(0.)# model.energy_triangle_dn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_up = torch.tensor(0.)# model.energy_triangle_up(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_nnn = torch.tensor(0.)# model.energy_nnn(state, env, force_cpu=ctm_args.conv_check_cpu)
        e_curr = (e_up + e_dn + e_nnn)/3
        spectra = []
        for c_loc, c_ten in env.C.items():
            u, s, v = torch.svd(c_ten, compute_uv=False)
            spectra += list(s/s[0])
        spectra = torch.tensor(spectra)
        history.append([e_curr.item(), spectra])
        if len(history)==1:
            delta_s = 0
        else:
            delta_s = torch.norm(history[-2][1] - history[-1][1]).item()
        print_corner_spectra(env)
        print('Step n°{:2}    E_site ={:01.14f}   (E_up={:01.14f}, E_dn={:01.14f}, E_nnn={:01.14f})  delta_s={:01.14f}'.format(len(history), e_curr.item(), e_up.item(), e_dn.item(), e_nnn, delta_s))
        if (len(history) > 1 and delta_s < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

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

    def overlap(state1, state2, env):
        O12 = torch.abs(rdm.rdm1x1_id_overlap((0,0), state1, state2, env, force_cpu=True))
        return O12


    state_AKLT = IPEPS_U1SYM(tensors_triangle, tensors_site,
                             coeffs_triangle_up={(0, 0): torch.tensor([1., 0, 0], dtype=torch.float64, device=t_device)},
                             coeffs_site={(0, 0): torch.tensor([1., 1.], dtype=torch.float64, device=t_device)})

    def state_U1(mu1, compute_overlap =False):
        coeffs_triangle = {(0, 0): torch.tensor([1., 0, mu1], dtype=torch.float64, device=t_device)}
        coeffs_site = {(0, 0): torch.tensor([1., 1.], dtype=torch.float64, device=t_device)}
        state = IPEPS_U1SYM(tensors_triangle, tensors_site, coeffs_triangle_up=coeffs_triangle, coeffs_site=coeffs_site)
        print('\n \n')
        state.print_coeffs()
        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)
        ctm_env_final, history, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
        if compute_overlap:
            O11 = overlap(state, state, ctm_env_final)
            ctm_env_12 = ENV(args.chi, state)
            init_prod_overlap(state, state_AKLT, ctm_env_12)
            ctm_env_final_12, history_O12, *ctm_log = ctmrg.run_overlap(state, state_AKLT, ctm_env_12, ctmrg_conv_overlap)
            O12 = history_O12[-1]
            return(history[-1], O12**2/O11)
        return(history[-1])

    def state_Z3(mu1, mu2, compute_overlap=False):
        coeffs_triangle = {(0, 0): torch.tensor([1., mu2, mu1], dtype=torch.float64, device=t_device)}
        coeffs_site = {(0, 0): torch.tensor([1., 1.], dtype=torch.float64, device=t_device)}
        state = IPEPS_U1SYM(tensors_triangle, tensors_site, coeffs_triangle_up=coeffs_triangle, coeffs_site=coeffs_site)
        print('\n \n')
        state.print_coeffs()
        ctm_env_init = ENV(args.chi, state)
        init_env(state, ctm_env_init)
        ctm_env_final, history, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
        if compute_overlap:
            O11 = overlap(state, state, ctm_env_final)
            ctm_env_12 = ENV(args.chi, state)
            init_prod_overlap(state, state_AKLT, ctm_env_12)
            ctm_env_final_12, history_O12, *ctm_log = ctmrg.run_overlap(state, state_AKLT, ctm_env_12, ctmrg_conv_overlap)
            O12 = history_O12[-1]
            return (history[-1], O12 ** 2 / O11)
        return (history[-1])


    def Z3_landscape(Mu1, Mu2, n_points):
        list_mu1 = np.linspace(-Mu1, Mu1, n_points)
        list_mu2 = np.linspace(-Mu2, Mu2, n_points)
        filename = 'Z3_chi{}_1Mu{}_2Mu{}'.format(args.chi, Mu1, Mu2)
        list_energies = []
        for mu1 in list_mu1:
            list_energies_mu1 = []
            for mu2 in list_mu2:
                energy, *observables = state_Z3(mu1, mu2, True)
                list_energies_mu1.append(energy)
            list_energies.append(list_energies_mu1)
            np.save(filename, np.array(list_energies))
        list_energies = np.array(list_energies)
        print('\n ' + filename)
        return(list_energies)


    def U1_landscape(Mu1, n_points):
        list_mu1 = np.linspace(0, Mu1, n_points)
        filename = 'U1_chi{}_1Mu{}'.format(args.chi, Mu1)
        list_energies = []
        list_overlaps = []
        for mu1 in list_mu1:
            energy = state_U1(mu1, False)
            list_energies.append(energy)
            #list_overlaps.append(aklt_overlap)
            np.save(filename+'_energies', np.array(list_energies))
            #np.save(filename + '_overlaps', np.array(list_overlaps))
        print('\n '+filename)
        return(list_energies)


    U1_landscape(2, args.npts)




if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
