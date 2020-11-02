import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import baTiOCu2Po4
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest neighbour interaction (strong plaquettes)")
parser.add_argument("--j2", type=float, default=0., help="next-nearest neighbour interaction (strong plaquettes)")
parser.add_argument("--jp2", type=float, default=0., help="next-nearest neighbour interaction (weak plaquettes)")
parser.add_argument("--jp11", type=float, default=0., help="nearest neighbour interaction (weak plaquettes)")
parser.add_argument("--jp12", type=float, default=0., help="nearest neighbour interaction (weak plaquettes)")
parser.add_argument("--tiling", default="8SITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model= baTiOCu2Po4.BaTiOCu2Po44(j1=args.j1, j2=args.j2, jp2=args.jp2,\
        jp11=args.jp11, jp12=args.jp12)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "8SITE":
        def lattice_to_site(coord):
            shift_x = coord[0] + 2*(coord[1] // 2)
            vx = shift_x % 4
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"8SITE")

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "8SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        sites={}
        for i in range(8):
            tmp= torch.rand([model.phys_dim]+ [bond_dim]*4,\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            tmp= tmp/tmp.abs().max()
            sites[(i%4,i//4)]= tmp
            
        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
            +" the model")
        baTiOCu2Po4.BaTiOCu2Po44(j1=args.j1, j2=args.j2, jp2=args.jp2,\
            jp11=args.jp11, jp12=args.jp12)

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling == "8SITE":
        energy_f=model.energy_2x2_8site
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state, env)
        history.append(e_curr.item())
        obs_values, obs_labels = model.eval_obs(state, env)
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and \
            abs( 2*(history[-1]-history[-2])/(history[-1]+history[-2]) ) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # next-nearest neighbours
    print("\n")
    obs_nnn_SS= model.eval_nnn_SS(state,ctm_env)
    print(", ".join([f"{k}" for k in obs_nnn_SS.keys()]))
    print(", ".join([f"{obs_nnn_SS[k]}" for k in obs_nnn_SS.keys()]))

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()