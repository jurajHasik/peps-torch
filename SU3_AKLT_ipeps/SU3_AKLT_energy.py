import context
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models.SU3_AKLT import *
import unittest
import torch
import numpy as np

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues" +
                                                         "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

model = SU3_AKLT()


def lattice_to_site(coord): return (0, 0)


state = read_ipeps('SU3_AKLT_ipeps/SU3_AKLT_ipeps.json', vertexToSite=lattice_to_site)
state.add_noise(args.instate_noise)


def main():
    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    def ctmrg_conv_rdm(state, env, history, ctm_args=cfg.ctm_args):
        """
        Convergence is defined wrt. the reduced density matrix.
        We define the following quantity at step i:
            Delta_rho[i] = Norm( rho[i+1] - rho[i] )
        and the algoritm stops when Delta_rho < tolerance.
        """
        with torch.no_grad():
            if not history:
                history = []
            e_curr = model.energy_triangle(state, env)
            rdm_curr = model.rdm1x1(state, env)
            print("\n")
            for c_loc, c_ten in env.C.items():
                u, s, v = torch.svd(c_ten, compute_uv=False)
                print(f"spectrum C[{c_loc}]")
                for i in range(args.chi):
                    print(f"{i} {s[i]}")
            history.append([e_curr, rdm_curr])
            if len(history) <= 1:
                Delta_rho = 'not defined'
            else:
                Delta_rho = torch.norm(history[-1][1] - history[-2][1]).item()
            print('Step n°' + str(len(history)) + '     E_down = ' + str(e_curr.item()) + '     Delta_rho = ' + str(
                Delta_rho))
            if len(history) > 1 and Delta_rho < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    def print_corner_spectra(env):
        spectra = []
        for c_loc, c_ten in env.C.items():
            u, s, v = torch.svd(c_ten, compute_uv=False)
            if c_loc[1] == (-1, -1):
                label = 'LU'
            if c_loc[1] == (-1, 1):
                label = 'LD'
            if c_loc[1] == (1, -1):
                label = 'RU'
            if c_loc[1] == (1, 1):
                label = 'RD'
            spectra.append([label, s])
        print(
            f"\n\nspectrum C[{spectra[0][0]}]             spectrum C[{spectra[1][0]}]             spectrum C[{spectra[2][0]}]             spectrum C[{spectra[3][0]}] ")
        for i in range(args.chi):
            print("{:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}        {:2} {:01.14f}".format(i,
                                                                                                            spectra[0][
                                                                                                                1][i],
                                                                                                            i,
                                                                                                            spectra[1][
                                                                                                                1][i],
                                                                                                            i,
                                                                                                            spectra[2][
                                                                                                                1][i],
                                                                                                            i,
                                                                                                            spectra[3][
                                                                                                                1][i]))


    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        """
        Convergence is defined wrt. the energy
        """
        with torch.no_grad():
            if not history:
                history = []
            e_curr = model.energy_triangle(state, env)
            history.append([e_curr])
            if len(history) <= 1:
                Delta_E = 'not defined'
            else:
                Delta_E = torch.norm(history[-1][0] - history[-2][0]).item()
            print_corner_spectra(env)
            print('Step n°' + str(len(history)) + '     E_down = ' + str(e_curr.item()))
            if len(history) > 1 and Delta_E < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    # initializes an environment for the ipeps
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print_corner_spectra(ctm_env_init)

    # initial energy of up and down triangles
    e_init_dn = model.energy_triangle(state, ctm_env_init)
    print('*** Energy per site (before CTMRG) -- down triangles: ' + str(e_init_dn.item()))
    e_init_up = model.energy_triangle_up(state, ctm_env_init)
    print('*** Energy per site (before CTMRG) -- up triangles: ' + str(e_init_up.item()))

    # performs CTMRG and computes the obervables afterwards (energy and lambda_3,8)
    ctm_env_fin, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    e_final_dn = model.energy_triangle(state, ctm_env_fin)
    print('*** Energy per site (after CTMRG) -- down triangles: ' + str(e_final_dn.item()))
    e_final_up = model.energy_triangle_up(state, ctm_env_fin)
    print('*** Energy per site (after CTMRG) -- up triangles: ' + str(e_final_up.item()))

    colors3, colors8 = model.eval_lambdas(state, ctm_env_fin)
    print('*** <Lambda_3> (after CTMRG): ' + str(colors3[0].item()) + ', ' + str(colors3[1].item()) + ', ' + str(
        colors3[2].item()))
    print('*** <Lambda_8> (after CTMRG): ' + str(colors8[0].item()) + ', ' + str(colors8[1].item()) + ', ' + str(
        colors8[2].item()))

    # environment diagnostics
    print("\n")
    print_corner_spectra(ctm_env_fin)


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
