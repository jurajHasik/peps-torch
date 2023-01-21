import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ipeps.ipeps_trgl_pg import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import spin_triangular
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim_sgd_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j4", type=float, default=0., help="plaquette coupling")
parser.add_argument("--jchi", type=float, default=0., help="scalar chirality")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice", \
    choices=["1SITE", "1SITE_NOROT", "1STRIV", "1SPG"])
parser.add_argument("--corrf_canonical", action='store_true', help="align spin operators" \
    + " with the vector of spontaneous magnetization")
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--compressed_rdms", type=int, default=-1, help="use compressed RDMs for 2x3 and 3x2 patches"\
        +" with chi lower that chi x D^2")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling in ["1SITE", "1STRIV", "1SPG"]:
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["1SITE_NOROT"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE")

    if args.instate!=None:
        if args.tiling in ["1STRIV"]:
            state= read_ipeps_trgl_1s_ttphys_pg(args.instate)
        elif args.tiling in ["1SPG"]:
            state= read_ipeps_trgl_1s_tbt_pg(args.instate)
        else:
            state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling in ["1SITE", "1SITE_NOROT"]:
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TTPHYS_PG()
        elif args.tiling == "1SPG":
            state= IPEPS_TRGL_1S_TBT_PG()
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4)

    print(state)
    
    # gauge, operates only IPEPS base and its sites tensors
    if args.gauge:
        state_g= IPEPS_WEIGHTED(state=state).gauge()
        state= state_g.absorb_weights()

    # 2) select the "energy" function 
    if args.tiling in ["1SITE", "1SITE_NOROT", "1STRIV", "1SPG"]:
        energy_f=model.energy_1x3
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env, compressed=args.compressed_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        history.append([e_curr.item()]+obs_values)
        print(", ".join([f"{len(history)}"]+[f"{e_curr}"]*2+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol):
            return True, history
        return False, history

    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        e_curr= energy_f(state, env, compressed=args.compressed_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}",\
            f"{e_curr}"]+[f"{v}" for v in obs_values]))
        return _conv_check, history

    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    
    loss0= energy_f(state, ctm_env_init, compressed=args.compressed_rdms)
    obs_values, obs_labels = eval_obs_f(state,ctm_env_init)
    print(", ".join(["epoch","conv_crit","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init, compressed=args.compressed_rdms)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    # print("\n")
    # for c_loc,c_ten in ctm_env_init.C.items(): 
    #     u,s,v= torch.svd(c_ten, compute_uv=False)
    #     print(f"spectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")

    # chirality
    # obs= model.eval_obs_chirality(state, ctm_env_init, compressed=args.compressed_rdms,\
    #     looped=args.loop_rdms)
    # print("\n\n")
    # for label,val in obs.items():
    #     print(f"{label} {val}")

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    dir_to_ind={ (0,-1): 1, (-1,0): 2, (0,1):3, (1,0):4 }
    evecs, evecs_maps=dict(), dict()
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        # direction gives direction of "growth" of the channel
        # i.e. for (1,0) the channel grows to right corresponding to solution of eq.
        #
        #  -- --T--           --
        # E-- --A-- = lambda E--  <=>  solving for *left* eigenvector E(TAT) = lambda E
        #  -- --T--           --    
        l, evecs_left= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init, eigenvectors=True)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

        # compute right eigenvector (TAT)E = lambda E (by reversing the direction of growth), 
        # since A = P^-1 diag(lambda) P i.e. P[:,0] and P^-1[0,:] are not simply related.
        l, evecs_right= transferops.get_Top_spec(1, sdp[0], (-sdp[1][0],-sdp[1][1]), \
            state, ctm_env_init, eigenvectors=True)

        assert evecs_left[:,0].imag.abs().max()<1.0e-14,"Leading eigenvector is not real"
        assert evecs_right[:,0].imag.abs().max()<1.0e-14,"Leading eigenvector is not real"
        evecs[sdp]= (evecs_left[:,0].real.view(ctm_env_init.chi,\
            state.site(sdp[0]).size(dir_to_ind[(-sdp[1][0],-sdp[1][1])])**2,ctm_env_init.chi).clone(),
            evecs_right[:,0].real.view(ctm_env_init.chi,\
                state.site(sdp[0]).size(dir_to_ind[sdp[1]])**2,ctm_env_init.chi).clone())

        # Pass in tuple of leading eigenvector-generating functions. First element gives
        # the left eigenvector, second gives right eigenvector
        evecs_maps[sdp]= (lambda x: evecs[sdp][0], lambda x: evecs[sdp][1])

    # ----- S(0).S(r) -----
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r, \
            canonical=args.corrf_canonical, rl_0=evecs_maps[sdp])
        print(f"\n\nSRSRt[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()])\
            +f" canonical {args.corrf_canonical}")
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r, \
            canonical=args.corrf_canonical, conj_s=False, rl_0=evecs_maps[sdp])
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()])\
            +f" canonical {args.corrf_canonical}")
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ----- S(0).Id(r) -----
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        corrSId= model.eval_corrf_SId(*sdp, state, ctm_env_init, 0, rl_0=evecs_maps[sdp])
        print(f"\n\nSId[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSId.keys()]))
        for i in range(len(next(iter(corrSId.values())))):
            print(f"{i} "+" ".join([f"{corrSId[label][i]}" for label in corrSId.keys()]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()