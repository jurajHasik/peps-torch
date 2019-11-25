import context
import torch
import argparse
import config as cfg
from ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import jq

if __name__=='__main__':
    # parse command line args and build necessary configuration objects
    parser= cfg.get_args_parser()
    # additional model-dependent arguments
    parser.add_argument("-j1", type=float, default=0.0, help="nearest-neighbour coupling")
    parser.add_argument("-q", type=float, default=1.0, help="plaquette interaction strength")
    # additional observables-related arguments
    parser.add_argument("-corrf_r", type=int, default=1, help="maximal correlation function distance")
    args = parser.parse_args()
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    
    model = jq.JQ(j1=args.j1, q=args.q)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    

    if args.instate!=None:
        state = read_ipeps(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        add_random_noise(state, args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype)
        D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype)

        sites = {(0,0): A, (1,0): B, (0,1): C, (1,1): D}

        for k in sites.keys():
            sites[k] = sites[k]/torch.max(torch.abs(sites[k]))
        state = IPEPS(sites, lX=2, lY=2)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            e_curr = model.energy_2x2_4site(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()]+obs_values)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_2x2_4site(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, history, t_ctm = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # ---- S.S(r) -----
    corrSS= model.eval_corrf_SS((0,0), (0,1), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(1,0)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS= model.eval_corrf_SS((0,0), (1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(0,-1)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS= model.eval_corrf_SS((0,0), (-1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(1,0)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS= model.eval_corrf_SS((0,0), (0,-1), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(0,-1)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ---(S.S)(S.S)(r) -----
    corrDD= model.eval_corrf_DD_H((0,0), (0,1), state, ctm_env_init, args.corrf_r)
    print("\n\nDD[(0,0),(0,1)] r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    corrDD= model.eval_corrf_DD_H((0,0), (1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nDD[(0,0),(1,0)] r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    corrDD= model.eval_corrf_DD_H((0,0), (-1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nDD[(0,0),(-1,0)] r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    corrDD= model.eval_corrf_DD_H((0,0), (0,-1), state, ctm_env_init, args.corrf_r)
    print("\n\nDD[(0,0),(0,-1)] r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")