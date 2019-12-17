import context
import torch
import argparse
import config as cfg
from ipeps import *
from c4v import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v import transferops_c4v
from models import j1j2

if __name__=='__main__':
    # parse command line args and build necessary configuration objects
    parser= cfg.get_args_parser()
    # additional model-dependent arguments
    parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
    parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
    # additional observables-related arguments
    parser.add_argument("-corrf_canonical", action='store_true', help="align spin operators" \
        + " with the vector of spontaneous magnetization")
    parser.add_argument("-corrf_r", type=int, default=1, help="maximal correlation function distance")
    parser.add_argument("-top_n", type=int, default=2, help="number of leading eigenvalues"+
        "of transfer operator to compute")
    args = parser.parse_args()
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    
    model = j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    
    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=None)
        assert len(state.sites)==1, "Not a 1-site ipeps"
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        add_random_noise(state, args.instate_noise)
        state.sites[(0,0)]= make_c4v_symm(state.sites[(0,0)])
        state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype)
        A= make_c4v_symm(A)
        A= A/torch.max(torch.abs(A))

        sites = {(0,0): A}

        state = IPEPS(sites)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            e_curr = model.energy_1x1(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()]+obs_values)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env_init = ENV_C4V(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = model.energy_1x1(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, history, t_ctm = ctmrg_c4v.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    corrSS= model.eval_corrf_SS(state, ctm_env_init, args.corrf_r, canonical=args.corrf_canonical)
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