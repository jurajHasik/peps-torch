import context
import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1xU1_torch as settings_U1xU1
from ipeps.ipess_kagome_abelian import read_ipess_kagome_generic
from linalg.custom_svd import truncated_svd_gesdd
from models.abelian import su3_kagome
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
# from ctm.generic import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--phi", type=float, default=0.5, help="arctan(K/J): J -> 2-site coupling; K -> 3-site coupling")
parser.add_argument("--theta", type=float, default=0., help="arctan(H/K): K -> 3-site coupling; K -> chiral coupling")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
parser.add_argument("--symmetry", default=None, help="symmetry structure", choices=["NONE","U1xU1"])
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    param_j = np.round(np.cos(np.pi*args.phi), decimals=15)
    param_k = np.round(np.sin(np.pi*args.phi) * np.cos(np.pi*args.theta), decimals=15)
    param_h = np.round(np.sin(np.pi*args.phi) * np.sin(np.pi*args.theta), decimals=15)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))\
    # TODO(?) choose symmetry group
    if not args.symmetry or args.symmetry=="NONE":
        settings= settings_full
    elif args.symmetry=="U1xU1":
        settings= settings_U1xU1
    # override default device specified in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    # override default dtype specified in settings
    settings.default_dtype= cfg.global_args.dtype
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= su3_kagome.KAGOME_SU3_U1xU1(settings,j=param_j,k=param_k,h=param_h,global_args=cfg.global_args)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.instate!=None:
        state= read_ipess_kagome_generic(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    import pdb; pdb.set_trace()

    # 2) define convergence criterion for ctmrg
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        # simply use single down triangle energy to evaluate the CTMRG convergence
        e_curr = model.energy_down_t_1x1subsystem(state, env)
        history.append(e_curr.item())
        obs_values, obs_labels = model.eval_obs(state, env)
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            return True, history
        return False, history

    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    print(ctm_env)
    import pdb; pdb.set_trace()

    # 3) evaluate observables for initial environment
    loss= model.energy_per_site_2x2subsystem(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    # 5) compute final observables and timings
    loss= model.energy_per_site_2x2subsystem(state, ctm_env)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # # environment diagnostics
    # for c_loc,c_ten in ctm_env.C.items():
    #     u,s,v= truncated_svd_gesdd(c_ten, c_ten.size(0))
    #     print(f"\n\nspectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")

    # # transfer operator spectrum
    # site_dir_list=[((0,0), (1,0)),((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
    # for sdp in site_dir_list:
    #     print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
    #     l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env)
    #     for i in range(l.size()[0]):
    #         print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()