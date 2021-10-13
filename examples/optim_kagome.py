import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import context
import torch
import argparse
import config as cfg
# from ipeps.ipeps import *
from ipeps.ipeps_kagome import *
from ctm.generic.env import *
from ctm.generic import ctmrg
# from ctm.generic import ctmrg_sl as ctmrg
from models import kagome

from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
import numpy as np

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional model-dependent arguments

# parser.add_argument("--j", type=float, default=0., help="coupling for 2-site exchange terms")
# parser.add_argument("--k", type=float, default=-1., help="coupling for 3-site ring exchange terms")
# parser.add_argument("--h", type=float, default=0., help="coupling for chiral terms")
parser.add_argument("--theta", type=float, default=0.5, help="parametrization between 2- and 3-site terms. theta * pi")
parser.add_argument("--phi", type=float, default=1., help="parametrization between normal and chiral terms. phi * pi")

parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    param_j = np.round(np.cos(np.pi*args.theta), decimals=12)
    param_k = np.round(np.sin(np.pi*args.theta) * np.cos(np.pi*args.phi), decimals=12)
    param_h = np.round(np.sin(np.pi*args.theta) * np.sin(np.pi*args.phi), decimals=12)
    print("J = {}; K = {}; H = {}".format(param_j, param_k, param_h))
    model = kagome.KAGOME(phys_dim=3, j=param_j, k=param_k, h=param_h)
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
    # elif args.tiling == "8SITE":
    #     def lattice_to_site(coord):
    #         shift_x = coord[0] + 2*(coord[1] // 2)
    #         vx = shift_x % 4
    #         vy = coord[1] % 2
    #         return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 1SITE, 2SITE, 4SITE")

    if args.instate != None:
        state = read_ipeps_kagome(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim_kagome(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state = IPEPS_KAGOME(dict(), lX=2, lY=1)
        elif args.tiling == "1SITE":
            state = IPEPS_KAGOME(dict(), lX=1, lY=1)
        elif args.tiling == "4SITE":
            state = IPEPS_KAGOME(dict(), lX=2, lY=2)
        # elif args.tiling == "8SITE":
        #     state = IPEPS_KAGOME(dict(), lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type =='RANDOM':
        bond_dim = args.bond_dim
        A = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        C = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        RD = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        RU = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        B = B/torch.max(torch.abs(B))
        C = C/torch.max(torch.abs(C))
        RD = RD/torch.max(torch.abs(RD))
        RU = RU/torch.max(torch.abs(RU))
        kagome_sites = {(0, 0, 0): A, (0, 0, 1): B, (0, 0, 2): C, (0, 0, 3): RD, (0, 0, 4): RU}
        if args.tiling in ["BIPARTITE", "2SITE", "4SITE"]:
            A2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            B2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            C2 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                           device=cfg.global_args.device)
            RD2 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RU2 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            kagome_sites[(1, 0, 0)] = A2/torch.max(torch.abs(A2))
            kagome_sites[(1, 0, 1)] = B2/torch.max(torch.abs(B2))
            kagome_sites[(1, 0, 2)] = C2/torch.max(torch.abs(C2))
            kagome_sites[(1, 0, 3)] = RD2/torch.max(torch.abs(RD2))
            kagome_sites[(1, 0, 4)] = RU2/torch.max(torch.abs(RU2))
        if args.tiling in ["4SITE"]:
            A3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            B3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            C3 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RD3 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            RU3 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            kagome_sites[(0, 1, 0)] = A3 / torch.max(torch.abs(A3))
            kagome_sites[(0, 1, 1)] = B3 / torch.max(torch.abs(B3))
            kagome_sites[(0, 1, 2)] = C3 / torch.max(torch.abs(C3))
            kagome_sites[(0, 1, 3)] = RD3 / torch.max(torch.abs(RD3))
            kagome_sites[(0, 1, 4)] = RU3 / torch.max(torch.abs(RU3))

            A4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            B4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            C4 = torch.rand((model.phys_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                            device=cfg.global_args.device)
            RD4 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            RU4 = torch.rand((bond_dim, bond_dim, bond_dim), dtype=cfg.global_args.torch_dtype,
                             device=cfg.global_args.device)
            kagome_sites[(1, 1, 0)] = A4 / torch.max(torch.abs(A4))
            kagome_sites[(1, 1, 1)] = B4 / torch.max(torch.abs(B4))
            kagome_sites[(1, 1, 2)] = C4 / torch.max(torch.abs(C4))
            kagome_sites[(1, 1, 3)] = RD4 / torch.max(torch.abs(RD4))
            kagome_sites[(1, 1, 4)] = RU4 / torch.max(torch.abs(RU4))

        # if args.tiling == "8SITE":
        #     E= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
        #         dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        #     F= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
        #         dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        #     G= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
        #         dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        #     H= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
        #         dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        #     sites[(2,0)] = E/torch.max(torch.abs(E))
        #     sites[(3,0)] = F/torch.max(torch.abs(F))
        #     sites[(2,1)] = G/torch.max(torch.abs(G))
        #     sites[(3,1)] = H/torch.max(torch.abs(H))
        state = IPEPS_KAGOME(kagome_sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype == model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model = kagome.KAGOME(phys_dim=3, j=param_j, k=param_k, h=param_h)

    # print(state)

    # 2) select the "energy" function
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f = model.energy_1site
        eval_obs_f= model.eval_obs
    elif args.tiling == "1SITE":
        energy_f= model.energy_1site
        # TODO include eval_obs with rotation on B-sublattice
        eval_obs_f= model.eval_obs
    elif args.tiling == "4SITE":
        energy_f = model.energy_1site
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = []
        e_curr = energy_f(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(state, ctm_env)
    print(", ".join(["epoch", "energy"]))
    print(", ".join([f"{-1}", f"{loss0}"]))

    obs_values, obs_labels = model.eval_obs(state, ctm_env)
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{loss0}"] + [f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        state.sites = state.build_unit_cell_tensors_kagome()
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, \
                                          conv_check=ctmrg_conv_energy, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        loss = energy_f(state, ctm_env_out)

        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
                or not "line_search" in opt_context.keys():
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state, ctm_env)
            print(", ".join([f"{epoch}", f"{loss}"] + [f"{v}" for v in obs_values]))
            log.info("Norm(sites): " + ", ".join([f"{t.norm()}" for c, t in state.sites.items()]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile = args.out_prefix + "_state.json"
    state = read_ipeps_kagome(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state, ctm_env)
    # obs_values, obs_labels = model.eval_obs(state, ctm_env)
    # print(", ".join([f"{args.opt_max_iter}", f"{opt_energy}"] + [f"{v}" for v in obs_values]))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
