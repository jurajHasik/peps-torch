from math import cos, sin
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from ctm.generic import corrf
from models import hb_anisotropy
import unittest


# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--theta", type=float, default=0., help="theta")
parser.add_argument("--ratio", type=float, default=1., help="y/x ratio")
parser.add_argument("--j1_x", type=float, default=1., help="nn x bilinear coupling")
parser.add_argument("--j1_y", type=float, default=1., help="nn y bilinear coupling")
parser.add_argument("--k1_x", type=float, default=0., help="nn x biquadratic coupling")
parser.add_argument("--k1_y", type=float, default=0., help="nn y biquadratic coupling")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    if args.theta:
        args.j1_x= cfg.main_args.j1_x= 1.0 * cos( args.theta )
        args.k1_x= cfg.main_args.k1_x= 1.0 * sin( args.theta )
        args.j1_y= cfg.main_args.j1_y= args.j1_x * args.ratio
        args.k1_y= cfg.main_args.k1_y= args.k1_x * args.ratio
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = hb_anisotropy.COUPLEDCHAINS(j1_x=args.j1_x, j1_y=args.j1_y, \
        k1_x=args.k1_x, k1_y=args.k1_y)

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
            return (vx, 0)
    else:
        raise ValueError("Invalid tiling: " + str(args.tiling) + " Supported options: " \
                         + "BIPARTITE, 2SITE")

    # 2) initialize an ipeps
    if args.instate != None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type == 'RANDOM':
        bond_dim = args.bond_dim

        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)

        # normalization of initial random tensors
        A = A / torch.max(torch.abs(A))
        B = B / torch.max(torch.abs(B))

        sites = {(0, 0): A, (1, 0): B}
        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= " \
                         + str(args.ipeps_init_type) + " is not supported")

    print(state)

    # 3) select the "energy" function
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f = model.energy_2x1_1x2
    else:
        raise ValueError("Invalid tiling: " + str(args.tiling) + " Supported options: " \
                         + "BIPARTITE, 2SITE")

    # 4) define convergence criterion
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history = []
            e_curr = energy_f(state, env)
            obs_values, obs_labels = model.eval_obs(state, env)
            history.append([e_curr.item()] + obs_values)
            print(", ".join([f"{len(history)}", f"{e_curr}"] + [f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0] - history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    # 5.1) compute observables from initial environment
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)

    # 5.2) run CTMRG
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{e_curr0}"] + [f"{v}" for v in obs_values0]))
    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)


    # compute horizontal and vertical two-point correlations
    corrSS = model.eval_corrf((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(1,0)] r " + " ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS = model.eval_corrf((0, 0), (0, 1), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(0,1)] r " + " ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))


    # compute horizontal dimer-dimer correlations
    corrDD_H= model.eval_corrf_DD_H((0,0), (1,0), state, ctm_env_init,\
        args.corrf_r)
    print("\n\nDD_H[(0,0),(1,0)] r " + " ".join([label for label in corrDD_H.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrDD_H[label][i]}" for label in corrDD_H.keys()]))

    # environment diagnostics
    print(f"\n\n")
    for c_loc, c_ten in ctm_env_init.C.items():
        u, s, v = torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    # transfer operator spectrum
    site_dir_list = [((0, 0), (1, 0)), ((0, 0), (0, 1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l = transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i, 0]} {l[i, 1]}")


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()