import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_kagome import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import kagome
import unittest
import numpy as np

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--theta", type=float, default=0.5, help="parametrization between 2- and 3-site terms. theta * pi")
parser.add_argument("--phi", type=float, default=0., help="parametrization between normal and chiral terms. phi * pi")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice")
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=8, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--output_prefix", type=str, default="output", help="filename of output data")
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
                         + "1SITE, BIPARTITE, 2SITE, 4SITE, 8SITE")

    # initialize an ipeps
    if args.instate != None:
        state = read_ipeps_kagome(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim_kagome(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type == 'RANDOM':
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

        state = IPEPS_KAGOME(kagome_sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= " \
                         + str(args.ipeps_init_type) + " is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= kagome.KAGOME(phys_dim=3, j=param_j, k=param_k, h=param_h)

    print(state)

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

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = model.eval_obs(state, ctm_env_init)

    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{e_curr0}"] + [f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # corrSS = model.eval_corrf((0, 0), (1, 0), state, ctm_env_init, 40)
    # print("\n\nSS[(0,0),(1,0)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    # # save spin-spin correlation function
    # filename = "./output/corr_ss_theta_"+str(args.theta)+".csv"
    # corr_ss_data = np.zeros(args.corrf_r)
    # for i in range(args.corrf_r):
    #     corr_ss_data[i] = corrSS["ss"][i]
    # np.savetxt(filename, corr_ss_data, fmt="%.10f", delimiter=',')

    # e_final = energy_f(state, ctm_env_init)
    # obs_values, obs_labels = model.eval_obs(state, ctm_env_init)
    # print(", ".join(["epoch", "energy"] + obs_labels))
    # print(", ".join([f"{-1}", f"{e_final}"] + [f"{v}" for v in obs_values]))

    # # save average magnetization and dimer operator
    # filename = "./output/onsite_obs_theta_"+str(args.theta)+".csv"
    # onsite_obs = np.zeros(2)
    # onsite_obs[0] = obs_values["avg_m"]
    # onsite_obs[1] = obs_values["dimer_op"]
    # np.savetxt(filename, onsite_obs, fmt="%.10f", delimiter=',')

    # corrSS = model.eval_corrf_SS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(1,0)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSS = model.eval_corrf_SS((0, 0), (0, 1), state, ctm_env_init, args.corrf_r)
    # print("\n\nSS[(0,0),(0,1)] r " + " ".join([label for label in corrSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))
    #
    # corrSSSS = model.eval_corrf_SSSS((0, 0), (1, 0), state, ctm_env_init, args.corrf_r)
    # print("\n\nSSSS[(0,0),(1,0)] r " + " ".join([label for label in corrSSSS.keys()]))
    # for i in range(args.corrf_r):
    #     print(f"{i} " + " ".join([f"{corrSSSS[label][i]}" for label in corrSSSS.keys()]))
    #
    # # environment diagnostics
    # print("\n")
    # for c_loc, c_ten in ctm_env_init.C.items():
    #     u, s, v = torch.svd(c_ten, compute_uv=False)
    #     print(f"spectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")
    #
    # transfer operator spectrum
    site_dir_list = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 1), (1, 0)), ((1, 1), (0, 1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l = transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i, 0]} {l[i, 1]}")

    # environment diagnostics
    for c_loc,c_ten in ctm_env_init.C.items():
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"\n\nspectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

    perm_2, h3_l, h3_r = model.get_h()
    idp = torch.eye(3, dtype=state.dtype)
    perm_2_tmp = torch.einsum('ij,ab->iabj', idp, idp)
    h3_l_tmp = torch.einsum('ia,jb,kc->ijkbca', idp, idp, idp)
    h3_r_tmp = torch.einsum('ia,jb,kc->ijkcab', idp, idp, idp)
    print(torch.sum(torch.abs(perm_2_tmp - perm_2)))
    print(torch.sum(torch.abs(h3_l - h3_l_tmp)))
    print(torch.sum(torch.abs(h3_r - h3_r_tmp)))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


