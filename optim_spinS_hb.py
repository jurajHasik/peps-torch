import torch
import argparse
import config as cfg
from ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import hb
from ad_optim import optimize_state

if __name__=='__main__':
    # parse command line args and build necessary configuration objects
    parser= cfg.get_args_parser()
    # additional model-dependent arguments
    parser.add_argument("-spinS", type=int, default=2, help="su(2) spin irrep dimension")
    parser.add_argument("-p1", type=float, default=1., help="nearest-neighbour bilinear coupling")
    parser.add_argument("-p2", type=float, default=0., help="nearest-neighbour biquadratic coupling")
    parser.add_argument("-tiling", default="BIPARTITE", help="tiling of the lattice")
    args = parser.parse_args()
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)

    model = j1j2.J1J2(j1=args.j1, j2=args.j2)
    
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
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    elif args.tiling == "8SITE":
        def lattice_to_site(coord):
            shift_x = coord[0] + 2*(coord[1] // 2)
            vx = shift_x % 4
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
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

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        B = B/torch.max(torch.abs(B))

        sites = {(0,0): A, (1,0): B}
        
        if args.tiling == "4SITE" or args.tiling == "8SITE":
            C= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            D= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            sites[(0,1)]= C/torch.max(torch.abs(C))
            sites[(1,1)] = D/torch.max(torch.abs(D))

        if args.tiling == "8SITE":
            E= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            F= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            G= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            H= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype)
            sites[(2,0)]= E/torch.max(torch.abs(E))
            sites[(3,0)] = F/torch.max(torch.abs(F))
            sites[(2,1)] = G/torch.max(torch.abs(G))
            sites[(3,1)] = H/torch.max(torch.abs(H))

        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f=model.energy_2x2_2site
    elif args.tiling == "4SITE":
        energy_f=model.energy_2x2_4site
    elif args.tiling == "8SITE":
        energy_f=model.energy_2x2_8site
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            e_curr = energy_f(state, env)
            history.append(e_curr.item())

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                return True
        return False

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, history, t_ctm = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_args=cfg.opt_args):
        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy)
        loss = energy_f(state, ctm_env_out)
        
        return loss, ctm_env_out, history, t_ctm

    # optimize
    optimize_state(state, ctm_env, loss_fn, model, args)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, history, t_ctm = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  