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
parser.add_argument("--tiling", default="3SITE", help="tiling of the lattice", \
    choices=["1SITE", "1STRIV", "1SPG", "3SITE", "4SITE"])
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--compressed_rdms", action='store_true', help="use compressed RDMs for 2x3 and 3x2 patches")
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
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4)
        lattice_to_site=None
    elif args.tiling == "3SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4)
        def lattice_to_site(coord):
            vx = coord[0] % 3
            vy = coord[1]
            return ((vx - vy) % 3, 0)
    elif args.tiling == "4SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4)
        def lattice_to_site(coord):
            vx = coord[0] % 2
            vy = ( coord[1] + ((coord[0]%4)//2) ) % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE, 3SITE, 4SITE")

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
        if args.tiling == "1SITE":
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TTPHYS_PG()
        elif args.tiling == "1SPG":
            state= IPEPS_TRGL_1S_TBT_PG()
        elif args.tiling == "3SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=3, lY=3)
        elif args.tiling == "4SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=4, lY=2)
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
    if args.tiling in ["1SITE", "1STRIV", "1SPG"]:
        energy_f=model.energy_1x3
        eval_obs_f= model.eval_obs
    elif args.tiling in ["3SITE", "4SITE"]:
        energy_f=model.energy_per_site if not args.compressed_rdms else \
            model.energy_per_site_compressed
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE, 3SITE, 4SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env)
        obs_values, obs_labels = eval_obs_f(state,env)
        history.append([e_curr.item()]+obs_values)
        print(", ".join([f"{len(history)}"]+[f"{e_curr}"]*2+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol):
            return True, history
        return False, history

    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        e_curr= energy_f(state, env)
        obs_values, obs_labels = eval_obs_f(state,env)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}",\
            f"{e_curr}"]+[f"{v}" for v in obs_values]))
        return _conv_check, history

    ctmrg_conv_f= ctmrg_conv_specC_loc

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)
    
    loss0= energy_f(state, ctm_env_init)
    obs_values, obs_labels = eval_obs_f(state,ctm_env_init)
    print(", ".join(["epoch","conv_crit","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()