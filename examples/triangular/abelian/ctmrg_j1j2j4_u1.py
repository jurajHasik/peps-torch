import context
import argparse
import copy
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_abelian import *
from ipeps.ipeps_abelian_c4v import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian.spin_triangular import J1J2J4_NOSYM
from models.spin_triangular import J1J2J4
from ctm.generic_abelian import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--diag", type=float, default=1., help="diagonal strength")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j4", type=float, default=0., help="plaquette coupling")
parser.add_argument("--jchi", type=float, default=0., help="scalar chirality")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["BIPARTITE", "1SITE_BP", "2SITE"])
parser.add_argument("--pg", default="NEEL_TRIANGULAR", help="point-group symmetries", \
    choices=["NONE", "NEEL_TRIANGULAR"])
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--corrf_DD", action='store_true', help="compute horizontal dimer-dimer correlation function")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument('--top_t', nargs="+", type=int, default=[-2,0,2], help="TM charge sectors")
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='np':
        from yastn.yastn.backend import backend_np as backend
    elif args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings_full= yastn.make_config(backend=backend, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    model= J1J2J4_NOSYM(settings_full, j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, \
            diag=args.diag)
    energy_f= model.energy_per_site
    eval_obs_f= model.eval_obs

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.tiling in ["BIPARTITE", "1SITE_BP"]:
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling in ["2SITE"]:
        def lattice_to_site(coord):
            return (coord[0] % 2, 0)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE")

    if args.instate!=None:
        if args.tiling in ["BIPARTITE","2SITE"]:
            state= read_ipeps(args.instate, settings, vertexToSite=lattice_to_site)
        if args.tiling == "1SITE_BP":
            state= read_ipeps_c4v(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    # TODO checkpointing
    elif args.opt_resume is not None:
        if args.tiling in ["BIPARTITE"]:
            state= IPEPS_ABELIAN(settings, dict(), lX=2, lY=2, vertexToSite=lattice_to_site)
        if args.tiling in ["2SITE"]:
            state= IPEPS_ABELIAN(settings, dict(), lX=2, lY=1, vertexToSite=lattice_to_site)
        if args.tiling == "1SITE_BP":
            state= IPEPS_ABELIAN_C4V(settings, irrep=args.pg)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= J1J2J4_NOSYM(settings_full, j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, \
            diag=args.diag)

    print(state)
    def generate_BP(state_1s_c4v):
        # 1 1      -1 -1                                        1           -1
        # 1 a 1 -> -1  a 1 ; a . flip_charges() -> -isigma^y = -1  -isigma^y a -1 
        #   1          1                                                    -1
        # create BP_rot op
        rot_op= yastn.Tensor(config=settings, s=[1,1], n=0,
            t=((1,-1),(1,-1)), D=((1,1),(1,1)) )
        rot_op.set_block((1,-1), (1,1), val=-np.ones((1,1)) )
        rot_op.set_block((-1,1), (1,1), val=np.ones((1,1)) )

        _tmp_sym= state_1s_c4v.symmetrize()

        b= rot_op.tensordot(_tmp_sym.site().flip_signature(),([1],[0]))
        b= b.flip_charges(axes=(0,3,4))

        return IPEPS_ABELIAN(settings, {(0,0): _tmp_sym.site().flip_charges(axes=(0,1,2)),\
            (1,0): b}, lattice_to_site, lX=2, lY=2)

    if args.tiling=="1SITE_BP":
        state= generate_BP(state)
        print(state)

    # 2) define convergence criterion for ctmrg
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env)
        obs_values, obs_labels = eval_obs_f(state,env)
        history.append([e_curr.item()]+obs_values)
        print(", ".join([f"{len(history)}"]+[f"{e_curr}"]*2+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            return True, history
        return False, history

    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        e_curr= energy_f(state, env)
        obs_values, obs_labels = eval_obs_f(state,env)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}",\
            f"{e_curr}"]+[f"{v}" for v in obs_values]))
        return _conv_check, history

    # alternatively use ctmrg_conv_specC from ctm.generinc.env
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    # 3) evaluate observables for initial environment
    loss= energy_f(state, ctm_env)
    obs_values, obs_labels= eval_obs_f(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)

    # 5) compute final observables and timings
    loss= energy_f(state, ctm_env)
    obs_values, obs_labels= eval_obs_f(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # ----- S(0).S(r) -----
    site_dir_list=[((0,0), (1,0)), ((0,0), (0,1)), ((1,0), (1,0)), ((1,0), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env, args.corrf_r, rl_0=None)
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ----- S(0).Id(r) -----
    for sdp in site_dir_list:
        corrSId= model.eval_corrf_SId(*sdp, state, ctm_env, 0, rl_0=None)
        print(f"\n\nSId[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSId.keys()]))
        for i in range(len(next(iter(corrSId.values())))):
            print(f"{i} "+" ".join([f"{corrSId[label][i]}" for label in corrSId.keys()]))

    # ----- horizontal dimer-dimer (S(0).S(x))(S(rx).S(rx+x)) -----
    # TODO conversion to dense
    state_dense= state.to_dense()
    ctm_env_dense= ctm_env.to_dense(state)
    model_dd= J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, diag=args.diag)
    if args.corrf_DD:
        for sdp in [((0,0), (1,0))]:
            corrDD= model_dd.eval_corrf_DD_H(*sdp, state_dense, ctm_env_dense, args.corrf_r)
            print(f"\n\nDD[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrDD.keys()]))
            for i in range(args.corrf_r):
                print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()])) 

    # environment diagnostics
    for c_loc,c_ten in ctm_env.C.items(): 
        u,s,v= c_ten.svd(([0],[1]))
        log.info(f"\n\nspectrum C[{c_loc}]")
        for charges in s.get_blocks_charge():
            log.info(f"{charges} {s[charges]}")

    # transfer operator spectrum 1-site-width channel
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}] {tuple(args.top_t)}")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env, edge_t=tuple(args.top_t))
        for i in range(l.shape[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()