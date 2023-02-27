import context
import torch
import argparse
import copy
import config as cfg
from ipeps.ipeps import *
from ipeps.ipeps_1s_Q import *
from ipeps.ipeps_trgl_pg import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import spin_triangular
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
parser.add_argument("--diag", type=float, default=1., help="diagonal strength")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j4", type=float, default=0., help="plaquette coupling")
parser.add_argument("--q", type=float, default=1., help="pitch vector")
parser.add_argument("--tiling", default="1STRIV", help="tiling of the lattice", \
    choices=["1SITEQ","1STRIV","1SPG"])
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # initialize an ipeps and model
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    #
    # 2) select the "energy" function 
    if args.tiling in ["1STRIV","1SPG"]:
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4)
        lattice_to_site=None
        energy_f=model.energy_1x3
        eval_obs_f= model.eval_obs
        read_state_f= read_ipeps_trgl_1s_ttphys_pg
        if args.tiling in ["1SPG"]:
            read_state_f= write_ipeps_trgl_1s_tbt_pg
    elif args.tiling in ["1SITEQ"]:
        model= spin_triangular.J1J2J4_1SITEQ(j1=args.j1, j2=args.j2, j4=args.j4, diag=args.diag,\
            q=(1./args.q,1./args.q))
        energy_f=model.energy_1x3
        eval_obs_f= model.eval_obs
        read_state_f= read_ipeps_1s_q
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITEQ, 1STRIV, 1SPG")

    if args.instate!=None:
        state = read_state_f(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = state.extend_bond_dim(args.bond_dim)
        if args.tiling in ["1STRIV","1SPG"]:
            state= state.add_noise(args.instate_noise)
        else:
            state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TRIVALENT()
        elif args.tiling in ["1SITEQ"]:
            state= IPEPS_1S_Q()
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        if args.tiling in ["1SITEQ"]:
            sites = {}
            A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
            sites[(0,0)]= A/torch.max(torch.abs(A))
            state = IPEPS_1S_Q(sites, q=(1./args.q,1./args.q))
        elif args.tiling in ["1STRIV"]:
            ansatz_pgs= IPEPS_TRGL_1S_TTPHYS_PG.PG_A1
            t_aux= torch.rand([bond_dim]*3,\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            t_phys= torch.rand([bond_dim]*3 + [model.phys_dim],\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            state = IPEPS_TRGL_1S_TTPHYS_PG(t_aux=t_aux, t_phys=t_phys, pgs=ansatz_pgs,\
                pg_symmetrize=True, peps_args=cfg.peps_args, global_args=cfg.global_args)
        elif args.tiling in ["1SPG"]:
            ansatz_pgs= IPEPS_TRGL_1S_TBT_PG.PG_A1_A
            t_aux= torch.rand([bond_dim]*3,\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            t_phys= torch.rand([bond_dim]*2 + [model.phys_dim],\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            state = IPEPS_TRGL_1S_TBT_PG(t_aux=t_aux, t_phys=t_phys, pgs=ansatz_pgs,\
                pg_symmetrize=True, peps_args=cfg.peps_args, global_args=cfg.global_args)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
            +" the model")
        if args.tiling in ["1SITEQ"]:
            model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4, diag=args.diag,\
                q=(1./args.q,1./args.q))
        else:    
            model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4)

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    # alternatively use ctmrg_conv_specC from ctm.generinc.env
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state, ctm_env)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build state with normalized tensors
        if args.tiling in ["1STRIV", "1SPG"]:
            state_sym= to_PG_symmetric(state)
        else:
            state_sym= state
            # state_sym.sites= state.build_onsite_tensors()
        
        # for c in state.sites.keys():
        #     with torch.no_grad():
        #         _scale= state.sites[c].abs().max()
        #     sites_n[c]= state.sites[c]/_scale
        # state_n= IPEPS(sites_n, vertexToSite=lattice_to_site, lX=state.lX, lY=state.lY)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state_sym, ctm_env_in, \
             conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        loss = energy_f(state_sym, ctm_env_out)

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state_sym= state
        if args.tiling in ["1SPG"]:
            # symmetrization and implicit rebuild of on-site tensors
            state_sym= to_PG_symmetric(state)

        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = eval_obs_f(state_sym,ctm_env)
            
            # test ENV sensitivity
            if args.test_env_sensitivity:
                loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
                loc_ctm_args.ctm_max_iter= 1
                ctm_env_out1= ctm_env.extend(ctm_env.chi+10)
                ctm_env_out1, *ctm_log= ctmrg.run(state_sym, ctm_env_out1, \
                    conv_check=ctmrg_conv_f, ctm_args=loc_ctm_args)
                loss1= energy_f(state_sym, ctm_env_out1)
                delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                    if len(opt_context['loss_history']['loss'])>1 else float('NaN')
                # if we are not linesearching, this can always happen
                # not "line_search" in opt_context.keys()
                _flag_antivar= (loss1-loss)>0 and (loss1-loss)>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
                + ([f"{loss1-loss}"] if args.test_env_sensitivity else []) ))
            
            log_info_string= f"env_sensitivity: {loss1-loss} loss_diff: "\
                +f"{delta_loss}" if args.test_env_sensitivity else ""
            if args.tiling in ["1STRIV","1SPG"]:
                log_info_string += " Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.elem_tensors.items()])
            else:
                log_info_string += " Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()])
            log.info(log_info_string)

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
            for c in state.elem_tensors.keys():
                _tmp= state.elem_tensors[c]/state.elem_tensors[c].abs().max()
                state.elem_tensors[c].copy_(_tmp)

    # optimize
    # state_g= IPEPS_WEIGHTED(state=state).gauge()
    # state= state_g.absorb_weights()
    # import pdb; pdb.set_trace()
    state.normalize_()
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)#, post_proc=post_proc)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_state_f(outputstatefile)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state,ctm_env)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{loss0}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()