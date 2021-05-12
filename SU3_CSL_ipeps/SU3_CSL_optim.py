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
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    print('\n')
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # Import all elementary tensors and build initial state
    elementary_tensors = []
    for name in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'L0', 'L1', 'L2']:
        ts = load_SU3_tensor(name)
        elementary_tensors.append(ts)
    # define initial coefficients
    #coeffs = {(0, 0): torch.tensor([-0.418167, -0.1490097, -1.87683, 0.146103, 1.64509, 0., 0., 1.14427, 0.277921, 0.],dtype=torch.float64)}
    coeffs = {(0, 0): torch.tensor([1., 1., 1., 1., 1., 0., 0., 1., 1., 1.], dtype=torch.float64)}
    # define which coefficients will be added a noise
    var_coeffs_allowed = torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
    state = IPEPS_U1SYM(elementary_tensors, coeffs, var_coeffs_allowed)
    state.add_noise(args.instate_noise)
    print(f'Current state: {state.coeffs[(0, 0)].data}')

    model = SU3_chiral.SU3_CHIRAL(theta=math.pi * args.frac_theta / 100.0, j1=args.j1, j2=args.j2)

    def energy_f(state, env):
        e_dn = model.energy_triangle_dn(state, env)
        e_up = model.energy_triangle_up(state, env)
        e_nnn = model.energy_nnn(state, env)
        return (e_up + e_dn + e_nnn) / 3

    # @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_dn = model.energy_triangle_dn(state, env)
        e_up = model.energy_triangle_up(state, env)
        e_nnn = model.energy_nnn(state, env)
        e_curr = (e_up + e_dn + e_nnn) / 3
        history.append(e_curr.item())
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
        # build on-site tensors from su2sym components
        state.sites = state.build_onsite_tensors()
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy,
                                                       ctm_args=ctm_args)
        loss = energy_f(state, ctm_env_out)
        timings = (t_ctm, t_obs)
        return loss, ctm_env_out, history, timings

    optimize_state(state, ctm_env_init, loss_fn)
    ctm_env_final, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # energy per site
    e_dn_final = model.energy_triangle_dn(state, ctm_env_final)
    e_up_final = model.energy_triangle_up(state, ctm_env_final)
    e_nnn_final = model.energy_nnn(state, ctm_env_final)
    e_tot_final = (e_dn_final + e_up_final + e_nnn_final) / 3

    # P operators
    P_up = model.P_up(state, ctm_env_final)
    P_dn = model.P_dn(state, ctm_env_final)

    print(f'\n\n E_up={e_up_final.item()}, E_dn={e_dn_final.item()}, E_tot={e_tot_final.item()}')
    print(f' Re(P_up)={torch.real(P_up).item()}, Im(P_up)={torch.imag(P_up).item()}')
    print(f' Re(P_dn)={torch.real(P_dn).item()}, Im(P_dn)={torch.imag(P_dn).item()}')


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
