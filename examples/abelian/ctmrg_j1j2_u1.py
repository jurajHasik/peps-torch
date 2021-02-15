import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
# import yamps.tensor as TA
import yamps.yast as TA
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import j1j2
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
parser.add_argument("--symmetry", default=None, help="symmetry structure", choices=["NONE","U1"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    # TODO(?) choose symmetry group
    if not args.symmetry or args.symmetry=="NONE":
        settings= settings_full
    elif args.symmetry=="U1":
        settings= settings_U1
    # override default device specified in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)
    
    # the model (in particular operators forming Hamiltonian) is defined in a dense form
    # with no symmetry structure
    model= j1j2.J1J2_NOSYM(settings_full,j1=args.j1,j2=args.j2)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "1SITE":
        def lattice_to_site(coord):
            return (0, 0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    elif args.tiling == "8SITE":
        def lattice_to_site(coord):
            shift_x = coord[0] + 2*(coord[1] // 2)
            vx = shift_x % 4
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 1SITE, 2SITE, 4SITE, 8SITE")

    if args.instate!=None:
        state= read_ipeps(args.instate, settings, vertexToSite=lattice_to_site)
        state= state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=2, lY=1)
            state= IPEPS(dict(), lX=2, lY=1)
        elif args.tiling == "1SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=1, lY=1)
        elif args.tiling == "4SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=2, lY=2)
        elif args.tiling == "8SITE":
            state= IPEPS_ABELIAN(settings, dict(), vertexToSite=lattice_to_site,\
                lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= j1j2.J1J2_NOSYM(settings_full,j1=args.j1,j2=args.j2)

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling=="BIPARTITE" or args.tiling=="2SITE" or args.tiling=="4SITE" \
        or args.tiling=="8SITE":
        energy_f=model.energy_2x1_or_2Lx2site_2x2rdms
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env).to_number()
        obs_values, obs_labels = model.eval_obs(state, env)
        history.append([e_curr]+obs_values)
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    
    # 3) evaluate observables for initial environment
    loss0 = energy_f(state, ctm_env).to_number()
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    # 5) compute final observables and timings
    e_curr0 = energy_f(state, ctm_env).to_number()
    obs_values0, obs_labels = model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    for c_loc,c_ten in ctm_env.C.items(): 
        u,s,v= c_ten.split_svd(([0],[1]))
        print(f"\n\nspectrum C[{c_loc}]")
        for charges, sector in s.A.items():
            print(charges)
            for i in range(len(sector)):
                print(f"{i} {sector[i]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()