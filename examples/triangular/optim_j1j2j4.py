import context
import copy
import torch
import argparse
import config as cfg
import time
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import spin_triangular
from ipeps.integration_yastn import PepsAD
from ctm.generic.env_yastn import from_yastn_env_generic, from_env_generic_dense_to_yastn, \
    YASTN_ENV_INIT, YASTN_PROJ_METHOD
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps.envs.fixed_pt import refill_env, fp_ctmrg
from yastn.yastn.tn.fpeps._peps import Peps2Layers
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
parser.add_argument("--diag", type=float, default=1, help="diagonal strength")
parser.add_argument("--tiling", default="3SITE", help="tiling of the lattice", \
    choices=["1SITE", "1SITE_NOROT", "2SITE", "2SITE_Y", "3SITE", "4SITE", "4SITE_T"])
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
parser.add_argument("--compressed_rdms", type=int, default=-1, help="use compressed RDMs for 2x3 and 3x2 patches"\
        +" with chi lower that chi x D^2")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--grad_type", type=str, default='default', help="gradient algo", choices=['default','fp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # initialize an ipeps and model
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.tiling == "1SITE":
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["1SITE_NOROT"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["2SITE", "2SITE_Y"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, \
            diag=args.diag)
        if args.tiling=="2SITE":
            def lattice_to_site(coord):
                vx = coord[0] % 2
                vy = coord[1]
                return (vx, 0)
        else:
            def lattice_to_site(coord):
                vx = coord[0]
                vy = coord[1] % 2
                return (0, vy)
    elif args.tiling=="3SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        def lattice_to_site(coord):
            vx = coord[0] % 3
            vy = coord[1]
            return ((vx - vy) % 3, 0)
    elif args.tiling=="4SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, \
            diag=args.diag)
        def lattice_to_site(coord):
            vx = coord[0] % 2
            vy = ( coord[1] + ((coord[0]%4)//2) ) % 2
            return (vx, vy)
    elif args.tiling=="4SITE_T":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, \
            diag=args.diag)
        def lattice_to_site(coord):
            vx = coord[0] % 2
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE, 2SITE, 3SITE, 4SITE, 4SITE_T")

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling in ["1SITE", "1SITE_NOROT"]:
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "2SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=2, lY=1)
        elif args.tiling == "2SITE_Y":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=1, lY=2)
        elif args.tiling == "3SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=3, lY=3)
        elif args.tiling == "4SITE_T":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=2, lY=2)
        elif args.tiling == "4SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        sites = {}
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
        sites[(0,0)]= A/torch.max(torch.abs(A))
        state = IPEPS(sites, lX=1, lY=1)
        if args.tiling in ["2SITE","2SITE_Y","3SITE","4SITE","4SITE_T"]:
            B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
            sites[ (0,1) if args.tiling=="2SITE_Y" else (1,0) ]= B/torch.max(torch.abs(B))
            lX_lY= dict(lX= 1, lY= 2) if args.tiling=="2SITE_Y" else dict(lX= 2, lY= 1)
            state = IPEPS(sites, vertexToSite=lattice_to_site, **lX_lY)
        if args.tiling in ["3SITE","4SITE","4SITE_T"]:
            C = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
            sites[(2,0)]= C/torch.max(torch.abs(C))
            state = IPEPS(sites, vertexToSite=lattice_to_site, lX=3, lY=3)
        if args.tiling in ["4SITE","4SITE_T"]:
            D = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
            del sites[(2,0)]
            sites[(0,1)]= C/torch.max(torch.abs(C))
            sites[(1,1)]= D/torch.max(torch.abs(D))
            lX_lY= dict(lX= 4, lY= 2) if args.tiling=="4SITE" else dict(lX= 2, lY= 2)
            state = IPEPS(sites, vertexToSite=lattice_to_site, **lX_lY)
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

    # 2) select the "energy" function
    energy_f=energy_f=model.energy_per_site
    eval_obs_f= model.eval_obs

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env, compressed=args.compressed_rdms, unroll=args.loop_rdms)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    # another option is ctmrg_conv_specC from ctm.generic.env

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    if args.ctm_conv_crit=="CSPEC":
        def ctmrg_conv_f(state,env,history,*aargs,**kwargs):
            conv, history= ctmrg_conv_specC(state,env,history,*aargs, **kwargs)
            if cfg.global_args.cuda_mem_profile:
                torch.cuda.memory._dump_snapshot(f"{args.out_prefix}_CTM-{len(history['conv_crit'])}_CUDAMEM.pickle")
            return conv, history
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    # 3) Initial env evaluation: Fast, possible starting point for optimization
    i0_ctm_args= copy.deepcopy(cfg.ctm_args)
    i0_ctm_args.projector_svd_method, i0_ctm_args.projector_rsvd_niter = "RSVD", 4
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f, ctm_args= i0_ctm_args)
    loss0= energy_f(state, ctm_env, compressed=args.compressed_rdms, unroll=args.loop_rdms)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn_default(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build state with normalized tensors
        state_n= state
        # sites_n= {}
        # for c in state.sites.keys():
        #     with torch.no_grad():
        #         _scale= state.sites[c].abs().max()
        #     sites_n[c]= state.sites[c]/_scale
        # state_n= IPEPS(sites_n, vertexToSite=lattice_to_site, lX=state.lX, lY=state.lY)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            if ctm_env_in is None:
                ctm_env_in= ENV(args.chi, state_n)
            init_env(state_n, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state_n, ctm_env_in, \
             conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        t_loss0= time.perf_counter()
        loss = energy_f(state_n, ctm_env_out, compressed=args.compressed_rdms,\
            unroll=args.loop_rdms)
        t_loss1= time.perf_counter()
        return (loss, ctm_env_out, *ctm_log, t_loss1-t_loss0)


    def loss_fn_fp(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # 2. convert to YASTN's iPEPS
        state_yastn= PepsAD.from_pt(state)

        # 3. proceed with YASTN's CTMRG implementation
        # 3.1 possibly re-initialize the environment
        if opt_args.opt_ctm_reinit or ctm_env_in is None:
            env_leg = yastn.Leg(state_yastn.config, s=1, D=(1,))
            ctm_env_in = EnvCTM(state_yastn, init=YASTN_ENV_INIT[ctm_args.ctm_env_init_type], leg=env_leg)
        else:
            ctm_env_in.psi = Peps2Layers(state_yastn) if state_yastn.has_physical() else state_yastn

        # 3.1.1 post-init CTM steps 
        options_svd_pre_init= {
            "policy": YASTN_PROJ_METHOD["RSVD"],
                "D_total": cfg.main_args.chi, 'D_block': cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
                "eps_multiplet": ctm_args.projector_eps_multiplet, "niter": ctm_args.projector_rsvd_niter,
            }
        with torch.no_grad():
            sweep, max_dsv, max_D, converge= ctm_env_in.ctmrg_(
                method="2site",
                max_sweeps=math.ceil(args.chi/(args.bond_dim**2)),
                opts_svd=options_svd_pre_init,
                corner_tol=ctm_args.projector_svd_reltol
            )
        log.log(logging.INFO, f"WARM-UP: Number of ctm steps: {sweep:d}, t_warm_up: N/As")

        # 3.2 setup and run CTMRG
        options_svd={
            "policy": YASTN_PROJ_METHOD[ctm_args.projector_svd_method],
            "D_total": cfg.main_args.chi, "D_block" : cfg.main_args.chi,
            "tol": ctm_args.projector_svd_reltol,
            "eps_multiplet": ctm_args.projector_eps_multiplet,
            'verbosity': ctm_args.verbosity_projectors
        }

        ctm_env_out, env_ts_slices, env_ts = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd= {'opts_svd': options_svd, 'corner_tol': ctm_args.ctm_conv_tol, 'max_sweeps': ctm_args.ctm_max_iter, \
                'method': "2site", 'use_qr': False,
                'checkpoint_move': 'reentrant' if ctm_args.fwd_checkpoint_move==True else ctm_args.fwd_checkpoint_move,
                },
            ctm_opts_fp= {'opts_svd': {'policy': 'fullrank'}, 'verbosity': 3,})
        refill_env(ctm_env_out, env_ts, env_ts_slices)

        # 3.3 convert environment to peps-torch format
        env_pt= from_yastn_env_generic(ctm_env_out, vertexToSite=state.vertexToSite)

        # 3.4 evaluate loss
        if cfg.global_args.cuda_mem_profile:
            torch.cuda.memory._dump_snapshot(f"{args.out_prefix}_preloss_CUDAMEM.pickle")
        t_loss0= time.perf_counter()
        loss= energy_f(state, env_pt, compressed=args.compressed_rdms, unroll=args.loop_rdms)
        t_loss1= time.perf_counter()
        if cfg.global_args.cuda_mem_profile:
            torch.cuda.memory._dump_snapshot(f"{args.out_prefix}_postloss_CUDAMEM.pickle")

        return (loss, ctm_env_out, [], None, t_loss1-t_loss0)


    @torch.no_grad()
    def obs_fn(state, _ctm_env, opt_context):
        if isinstance(_ctm_env, EnvCTM):
            ctm_env= from_yastn_env_generic(_ctm_env, vertexToSite=state.vertexToSite)
        else:
            ctm_env= _ctm_env

        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = eval_obs_f(state,ctm_env)
            _flag_antivar= False

            # test ENV sensitivity
            # for J2 or Jchi or J4 > 0, energy computation is expensive
            if args.test_env_sensitivity:
                loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
                loc_ctm_args.ctm_max_iter= 1
                ctm_env_out1= ctm_env.extend(ctm_env.chi+10)
                ctm_env_out1, *ctm_log= ctmrg.run(state, ctm_env_out1, \
                    conv_check=ctmrg_conv_f, ctm_args=loc_ctm_args)
                loss1= energy_f(state, ctm_env_out1, compressed=args.compressed_rdms, \
                    unroll=args.loop_rdms)
                delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                    if len(opt_context['loss_history']['loss'])>1 else float('NaN')
                # if we are not linesearching, this can always happen
                # not "line_search" in opt_context.keys()
                _flag_antivar= (loss1-loss)>0 and \
                    (loss1-loss)*opt_context["opt_args"].env_sens_scale>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
                + ([f"{loss1-loss}"] if args.test_env_sensitivity else []) ))
            log.info(f"env_sensitivity: {loss1-loss} loss_diff: "\
                +f"{delta_loss} ENV_ANTIVAR {_flag_antivar}" if args.test_env_sensitivity else ""\
                +" Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

            if _flag_antivar: raise EnvError("ENV_ANTIVAR")

            # with torch.no_grad():
            #     if args.top_freq>0 and epoch%args.top_freq==0:
            #         coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
            #         for c,d in coord_dir_pairs:
            #             # transfer operator spectrum
            #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
            #             l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
            #             print("TOP "+json.dumps(_to_json(l)))

    def post_proc(state, ctm_env, opt_context):
        with torch.no_grad():
            for c in state.sites.keys():
                _tmp= state.sites[c]/state.sites[c].abs().max()
                state.sites[c].copy_(_tmp)
            # if "STATUS" in opt_context and opt_context["STATUS"]=="ENV_ANTIVAR":
            #     state_g= IPEPS_WEIGHTED(state=state).gauge().absorb_weights()
            #     for c in state.sites.keys():
            #         state.sites[c].copy_(state_g.sites[c])

    # optimize
    # enable memory history, which will
    # add tracebacks and event history to snapshots
    torch.cuda.memory._record_memory_history(
        enabled=None, context=None, stacks='all', max_entries=9223372036854775807, device=None
        )

    state.normalize_()
    loss_fn= loss_fn_fp if args.grad_type=='fp' else loss_fn_default
    if args.grad_type=='fp':
        ctm_env= from_env_generic_dense_to_yastn(ctm_env, state)
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn, post_proc=post_proc)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state,ctm_env,compressed=args.compressed_rdms,unroll=args.loop_rdms)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{loss0}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()