import torch
import math
import config as cfg
import copy
from collections import OrderedDict
from ipeps import ipeps_kagome
from models import SU3_chiral
from ctm.generic.env import *
from ctm.generic import ctmrg
import json
from ipeps.tensor_io import *
import unittest
import logging
from tn_interface import einsum
from tn_interface import conj

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--chiral_angle", type=float, default=0., help="angle parametrizing the chiral Hamiltonian")
parser.add_argument("--top_freq", type=int, default=-1, help="frequency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2,
                    help="number of leading eigenvalues of transfer operator to compute")
parser.add_argument("--input_filename", type=str, default=None, help="input state for ctmrg")
parser.add_argument("--show_corner_spectra", type=bool, default=False, help="plot the corner spectra at each CTM step")
args, unknown_args = parser.parse_known_args()

def main():

    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    t_device = torch.device(args.GLOBALARGS_device)


    # Import tensors and build the ipeps object
    location = 'SU3_CSL_ipeps/unconstrained_D9_tensors/'
    with open(location+args.input_filename+'.json') as j:
        # load tensor as a json file
        tensor_set = json.load(j)
        triangle_up = tensor_set['elem_tensors']['UP_T']
        triangle_dn = tensor_set['elem_tensors']['DOWN_T']
        bond_site = tensor_set['elem_tensors']['BOND_S']
    triangle_up = torch.tensor(read_bare_json_tensor_np_legacy(triangle_up), dtype=torch.complex128)
    triangle_dn = torch.tensor(read_bare_json_tensor_np_legacy(triangle_dn), dtype=torch.complex128)
    bond_site = torch.tensor(read_bare_json_tensor_np_legacy(bond_site), dtype=torch.complex128)
    state = ipeps_kagome.IPEPS_KAGOME(triangle_up, bond_site, triangle_dn)

    # investigate underlying virtual space

    # Build double-layer matrix with bra-ket indices uncontracted

    a_1layer = state.sites[(0,0)]
    #      /|
    # /---A*----
    # |  /| |       :=     --M--
    # | | |/
    # \-|-A-----
    #   |/
    M = einsum('mibcd,mjbcd->ij', a_1layer, conj(a_1layer))

    M_eigenvalues, eigv = torch.eig(M, eigenvectors=False)
    print('M eigenvalues')
    print(M_eigenvalues)
    print('\n')
    u, M_singvalues, v = torch.svd(M, compute_uv=False)
    print('M singluar values')
    print(M_singvalues)
    print('\n')


    # CTMRG to compute associated observables

    model = SU3_chiral.SU3_CHIRAL(Kr=math.cos(args.chiral_angle * math.pi/180), Ki=math.sin(args.chiral_angle * math.pi/180))

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
        if args.show_corner_spectra:
            print_corner_spectra(env)
        print('Step n°{:2}    E_site ={:01.14f}   (E_up={:01.14f}, E_dn={:01.14f}, E_nnn={:01.14f})  delta_E={:01.14f}'.format(len(history), e_curr.item(), e_up.item(), e_dn.item(), e_nnn, e_curr.item()-e_prev))
        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

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

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
