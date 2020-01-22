import context
import torch
import argparse
import config as cfg
from su2sym.ipeps_su2 import * 
from groups.c4v import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v, transferops_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import j1j2
import su2sym.sym_ten_parser as tenSU2 
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
# additional observables-related arguments
parser.add_argument("-corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("-top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args = parser.parse_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    
    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_su2(args.instate, vertexToSite=None)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"
        add_random_noise(state, args.instate_noise)
        state.coeffs[(0,0)]= state.coeffs[(0,0)]/torch.max(torch.abs(state.coeffs[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))
        A= torch.rand(len(su2sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))

        coeffs = {(0,0): A}

        state = IPEPS_SU2SYM(su2_tensors=su2sym_t, coeffs=coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    # initialize on-site tensors
    state.build_onsite_tensors()
    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            rdm2x1= rdm2x1_sl(state, env)
            # print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))
            dist= float('inf')
            if len(history) > 1:
                dist= torch.dist(rdm2x1, history[-1][0], p=2)
                if dist<ctm_args.ctm_conv_tol:
                    return True
            history.append([rdm2x1,dist])
            print(f"{len(history)}, {dist}")
        return False

    ctm_env_init = ENV_C4V(args.chi, state, log=args.out_prefix+"_env.log")
    init_env(state, ctm_env_init)

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg_c4v.run(state, ctm_env_init, \
        conv_check=ctmrg_conv_energy)

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    # ----- additional observables ---------------------------------------------
    corrSS= model.eval_corrf_SS(state, ctm_env_init, args.corrf_r)
    print("\n\nSS r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrDD= model.eval_corrf_DD_H(state, ctm_env_init, args.corrf_r)
    print("\n\nDD r "+" ".join([label for label in corrDD.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()]))

    # environment diagnostics
    print("\n\nspectrum(C)")
    u,s,v= torch.svd(ctm_env_init.C[ctm_env_init.keyC], compute_uv=False)
    for i in range(args.chi):
        print(f"{i} {s[i]}")

    # transfer operator spectrum
    print("\n\nspectrum(T)")
    l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env_init)
    for i in range(l.size()[0]):
        print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    main()