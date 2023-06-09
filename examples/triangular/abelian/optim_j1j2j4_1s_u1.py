import context
import argparse
import torch
import copy
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_abelian import *
from ipeps.ipeps_abelian_c4v import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian.spin_triangular import J1J2J4_NOSYM
from models.abelian.spin_triangular import J1J2J4_1SITEQ_NOSYM
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
#from ctm.generic import transferops
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
parser.add_argument("--q", type=float, default=1., help="pitch vector")
parser.add_argument("--tiling", default="1SITE_BP", help="tiling of the lattice", \
    choices=["1SITE_BP"])
parser.add_argument("--pg", default="NEEL_TRIANGULAR", help="point-group symmetries", \
    choices=["NONE", "NEEL_TRIANGULAR"])
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    args.pg = None if args.pg=="NONE" else args.pg
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='torch':
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
    if args.instate!=None:
        state= read_ipeps_c4v(args.instate, settings)
        if state.irrep != args.pg:
            state= state.symmetrize(args.pg)
        state= state.add_noise(args.instate_noise)
    # TODO checkpointing
    elif args.opt_resume is not None:
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

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state, env).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    @torch.no_grad()
    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        return ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)

    # alternatively use ctmrg_conv_specC from ctm.generinc.env
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    def generate_BP(state_1s_c4v):
        # 1 1      -1 -1                                        1           -1
        # 1 a 1 -> -1  a 1 ; a . flip_charges() -> -isigma^y = -1  -isigma^y a -1 
        #   1          1                                                    -1
        # create BP_rot op
        def vertexToSite(r):
            x = (r[0] + abs(r[0]) * 2) % 2
            y = abs(r[1])
            return ((x + y) % 2, 0)

        rot_op= yastn.Tensor(config=settings, s=[1,1], n=0,
            t=((1,-1),(1,-1)), D=((1,1),(1,1)) )
        rot_op.set_block((1,-1), (1,1), val=-np.ones((1,1)) )
        rot_op.set_block((-1,1), (1,1), val=np.ones((1,1)) )

        _tmp_sym= state_1s_c4v.symmetrize(irrep=args.pg)

        b= rot_op.tensordot(_tmp_sym.site().flip_signature(),([1],[0]))
        b= b.flip_charges(axes=(0,3,4))

        return IPEPS_ABELIAN(settings, {(0,0): _tmp_sym.site().flip_charges(axes=(0,1,2)),\
            (1,0): b}, vertexToSite, lX=2, lY=2)

    state_bp= generate_BP(state)
    ctm_env= ENV_ABELIAN(args.chi, state=state_bp, init=True)

    ctm_env, *ctm_log= ctmrg.run(state_bp, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state_bp, ctm_env)
    obs_values, obs_labels = eval_obs_f(state_bp,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))


    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        state_bp= generate_BP(state)

        # build double-layer open on-site tensors
        state_bp.sync_precomputed()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_bp, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state_bp, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss= energy_f(state_bp, ctm_env_out)

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state_bp= generate_BP(state)

        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS "+", ".join([f"{epoch}",f"{loss}"]))
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
            obs_values, obs_labels = eval_obs_f(state_bp,ctm_env)

            # test ENV sensitivity
            loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
            loc_ctm_args.ctm_max_iter= 1
            ctm_env_out1= ctm_env.clone()
            ctm_env_out1.chi= ctm_env.chi+10
            ctm_env_out1, *ctm_log= ctmrg.run(state_bp, ctm_env_out1, \
                conv_check=ctmrg_conv_f, ctm_args=loc_ctm_args)
            loss1= energy_f(state_bp, ctm_env_out1)
            delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                if len(opt_context['loss_history']['loss'])>1 else float('NaN')
            # if we are not linesearching, this can always happen
            # not "line_search" in opt_context.keys()
            if args.test_env_sensitivity:
                _flag_antivar= (loss1-loss)>0 and \
                    (loss1-loss)*opt_context["opt_args"].env_sens_scale>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
                + [f"{loss1-loss}"]))
            log.info(f"env_sensitivity: {loss1-loss} loss_diff: {delta_loss}"\
                +" Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

        # with torch.no_grad():
        #     if (not opt_context["line_search"]) and args.top_freq>0 \
        #         and epoch%args.top_freq==0:
        #         coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
        #         for c,d in coord_dir_pairs:
        #             # transfer operator spectrum
        #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
        #             l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
        #             print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v(outputstatefile, settings)
    state_bp= generate_BP(state)
    ctm_env = ENV_ABELIAN(args.chi, state=state_bp, init=True)
    ctm_env, *ctm_log = ctmrg.run(state_bp, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state_bp,ctm_env)
    obs_values, obs_labels = eval_obs_f(state_bp,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()