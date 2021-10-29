import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
# from ctm.generic import ctmrg_sl as ctmrg
from models import bbh_j1j2

from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--theta", type=float, default=0., help="theta")
parser.add_argument("--ratio", type=float, default=1., help="y/x ratio")
parser.add_argument("--j1", type=float, default=1., help="nn bilinear coupling")
parser.add_argument("--k1", type=float, default=0., help="nn biquadratic coupling")
parser.add_argument("--j2", type=float, default=0., help="nnn bilinear coupling")
parser.add_argument("--k2", type=float, default=0., help="nnn biquadratic coupling")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = bbh_j1j2.BBH_J1J2(j1=args.j1, k1=args.k1, j2=args.j2, k2=args.k2)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            # vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, 0)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    # elif args.tiling == "8SITE":
    #     def lattice_to_site(coord):
    #         shift_x = coord[0] + 2 * (coord[1] // 2)
    #         vx = shift_x % 4
    #         vy = coord[1] % 2
    #         return (vx, vy)
    else:
        raise ValueError("Invalid tiling: " + str(args.tiling) + " Supported options: " \
                         + "BIPARTITE, 2SITE, 4SITE")

    if args.instate != None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state = IPEPS(dict(), lX=2, lY=1)
        elif args.tiling == "4SITE":
            state = IPEPS(dict(), lX=2, lY=2)
        # elif args.tiling == "8SITE":
        #     state = IPEPS(dict(), lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type == 'RANDOM':
        bond_dim = args.bond_dim

        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        # A = torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
        #                 dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        # A[:, 0, :, 0, :] = torch.rand_like(A[:, 0, :, 0, :], dtype=A.torch_dtype, device=A.device)
        # A = (1 - args.ratio) * A + args.ratio * torch.rand_like(A, dtype=A.torch_dtype, device=A.device)
        #
        # B = torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
        #                 dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        # B[:, 0, :, 0, :] = torch.rand_like(B[:, 0, :, 0, :], dtype=B.torch_dtype, device=B.device)
        # B = (1 - args.ratio) * B + args.ratio * torch.rand_like(B, dtype=B.torch_dtype, device=B.device)

        # normalization of initial random tensors
        A = A / torch.max(torch.abs(A))
        B = B / torch.max(torch.abs(B))
        sites = {(0, 0): A, (1, 0): B}
        if args.tiling == "4SITE" or args.tiling == "8SITE":
            C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                           dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                           dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            sites[(0, 1)] = C / torch.max(torch.abs(C))
            sites[(1, 1)] = D / torch.max(torch.abs(D))

        # if args.tiling == "8SITE":
        #     E = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
        #                    dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        #     F = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
        #                    dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        #     G = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
        #                    dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        #     H = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
        #                    dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        #     sites[(2, 0)] = E / torch.max(torch.abs(E))
        #     sites[(3, 0)] = F / torch.max(torch.abs(F))
        #     sites[(2, 1)] = G / torch.max(torch.abs(G))
        #     sites[(3, 1)] = H / torch.max(torch.abs(H))
        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= " \
                         + str(args.ipeps_init_type) + " is not supported")
    
    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= bbh_j1j2.BBH_J1J2(j1=args.j1, k1=args.k1, j2=args.j2, k2=args.k2)

    print(state)

    # 2) select the "energy" function
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f = model.energy_2x2
    elif args.tiling == "4SITE":
        energy_f = model.energy_2x2
    # elif args.tiling == "8SITE":
    #     energy_f = model.energy_2x2_8site
    else:
        raise ValueError("Invalid tiling: " + str(args.tiling) + " Supported options: " \
                         + "BIPARTITE, 2SITE, 4SITE, 8SITE")

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
    obs_values, obs_labels = model.eval_obs(state, ctm_env)
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{loss0}"] + [f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

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
    state = read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state, ctm_env)
    print(", ".join([f"{args.opt_max_iter}", f"{opt_energy}"] + [f"{v}" for v in obs_values]))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
