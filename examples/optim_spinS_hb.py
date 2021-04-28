import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import hb
from optim.ad_optim import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--spinS", type=int, default=2, help="su(2) spin irrep dimension")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour bilinear coupling")
parser.add_argument("--k1", type=float, default=0., help="nearest-neighbour biquadratic coupling")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = hb.HB(spin_s=args.spinS, j1=args.j1, k1=args.k1)
    
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
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
            state= IPEPS(dict(), lX=2, lY=1)
        elif args.tiling == "4SITE":
            state= IPEPS(dict(), lX=2, lY=2)
        elif args.tiling == "8SITE":
            state= IPEPS(dict(), lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        B = B/torch.max(torch.abs(B))

        sites = {(0,0): A, (1,0): B}
        
        if args.tiling == "4SITE" or args.tiling == "8SITE":
            C= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            D= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            sites[(0,1)]= C/torch.max(torch.abs(C))
            sites[(1,1)] = D/torch.max(torch.abs(D))

        if args.tiling == "8SITE":
            E= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            F= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            G= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            H= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
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
        energy_f=model.energy_2x1_1x2
    elif args.tiling == "4SITE":
        energy_f=model.energy_2x1_1x2
    elif args.tiling == "8SITE":
        energy_f=model.energy_2x1_1x2
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        # possibly re-initialize the environment
        if cfg.opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy)
        loss = energy_f(state, ctm_env_out)
        
        return (loss, ctm_env_out, *ctm_log)

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        epoch= len(opt_context["loss_history"]["loss"]) 
        loss= opt_context["loss_history"]["loss"][-1]
        obs_values, obs_labels = model.eval_obs(state,ctm_env)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3

    # basic tests
    def test_opt_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()