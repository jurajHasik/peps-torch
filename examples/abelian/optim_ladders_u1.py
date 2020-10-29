# import context
import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
import yamps.tensor as TA
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import coupledLadders
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod_abelian import optimize_state
#from optim.ad_optim_lbfgs_mod import optimize_state
#from ctm.generic import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--alpha", type=float, default=0., help="inter-ladder coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    # TODO(?) choose symmetry group
    if args.ipeps_init_type=='TEST_FULL' or (args.instate!=None and False):
        settings= settings_full
        print("HERE")
    elif args.ipeps_init_type=='TEST_U1' or (args.instate!=None and True):
        settings= settings_U1
    settings.back.set_num_threads(args.omp_cores)
    settings.back.random_seed(args.seed)

    model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.instate!=None:
        state= read_ipeps(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    # TODO checkpointing    
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN(settings_U1, dict(), lX=2, lY=2)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='TEST_FULL':
        a= TA.Tensor(settings=settings_full, s=IPEPS_ABELIAN._REF_S_DIRS)
        tmp_block_a= np.zeros((2,2,2,2,2))
        tmp_block_a[0,0,0,0,0]= -1.000635518923222
        tmp_block_a[1,0,0,0,1]= -0.421284989637812
        tmp_block_a[1,0,0,1,0]= -0.421284989637812
        tmp_block_a[1,0,1,0,0]= -0.421284989637812
        tmp_block_a[1,1,0,0,0]= -0.421284989637812
        a.set_block(val=tmp_block_a)

        b= TA.Tensor(settings=settings_full, s=IPEPS_ABELIAN._REF_S_DIRS)
        tmp_block_b= np.zeros((2,2,2,2,2))
        tmp_block_b[1,0,0,0,0]= 1.000635518923222
        tmp_block_b[0,0,0,0,1]= -0.421284989637812
        tmp_block_b[0,0,0,1,0]= -0.421284989637812
        tmp_block_b[0,0,1,0,0]= -0.421284989637812
        tmp_block_b[0,1,0,0,0]= -0.421284989637812
        b.set_block(val=tmp_block_b)

        c= TA.Tensor(settings=settings_full, s=IPEPS_ABELIAN._REF_S_DIRS)
        c.set_block(val=tmp_block_b.copy())
        d= TA.Tensor(settings=settings_full, s=IPEPS_ABELIAN._REF_S_DIRS)
        d.set_block(val=tmp_block_a.copy())

        sites = {(0,0): a, (1,0): b, (0,1): c, (1,1): d}
        state = IPEPS_ABELIAN(settings_full, sites, lX=2, lY=2)
    elif args.ipeps_init_type=='TEST_U1':
        # a = TA.Tensor(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0)
        a= TA.zeros(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0,
                        t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)) )
        tmp1= -1.000635518923222*np.ones((1,1,1,1,1))
        tmp2= -0.421284989637812*np.ones((1,1,1,1,1))
        a.set_block((0,0,0,0,0), (1,1,1,1,1), val=tmp1)
        a.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp2)
        a.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp2)

        # b = TA.Tensor(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0)
        b= TA.zeros(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0,
                        t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)) )
        b.set_block((0,0,0,0,0), (1,1,1,1,1), val=-tmp1)
        b.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp2)
        b.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp2)

        # c = TA.Tensor(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0)
        c= TA.zeros(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0,
                        t=((0, 1), (0, -1), (0, -1), (0,1), (0,1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)) )
        c.set_block((0,0,0,0,0), (1,1,1,1,1), val=-tmp1.copy())
        c.set_block((1,-1,0,0,0), (1,1,1,1,1), val=tmp2.copy())
        c.set_block((1,0,-1,0,0), (1,1,1,1,1), val=tmp2.copy())
        c.set_block((1,0,0,1,0), (1,1,1,1,1), val=tmp2.copy())
        c.set_block((1,0,0,0,1), (1,1,1,1,1), val=tmp2.copy())

        # d = TA.Tensor(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0)
        d= TA.zeros(settings=settings_U1, s=IPEPS_ABELIAN._REF_S_DIRS, n=0,
                        t=((0, -1), (0, 1), (0, 1), (0,-1), (0,-1)),
                        D=((1, 1), (1,1), (1,1), (1,1), (1,1)) )
        d.set_block((0,0,0,0,0), (1,1,1,1,1), val=tmp1.copy())
        d.set_block((-1,1,0,0,0), (1,1,1,1,1), val=tmp2.copy())
        d.set_block((-1,0,1,0,0), (1,1,1,1,1), val=tmp2.copy())
        d.set_block((-1,0,0,-1,0), (1,1,1,1,1), val=tmp2.copy())
        d.set_block((-1,0,0,0,-1), (1,1,1,1,1), val=tmp2.copy())

        sites = {(0,0): a, (1,0): b, (0,1): c, (1,1): d}
        state = IPEPS_ABELIAN(settings_U1, sites, lX=2, lY=2)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha)

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x1_1x2(state, env)
        history.append(e_curr.to_number().item())
        print(history)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss = model.energy_2x1_1x2(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss.to_number().item()}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss= model.energy_2x1_1x2(state, ctm_env_out)
        loss= loss.to_number()

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state,ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

            # with torch.no_grad():
            #     if args.top_freq>0 and epoch%args.top_freq==0:
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
    state= read_ipeps(outputstatefile, settings)
    ctm_env = ENV_ABELIAN(args.chi, state=state, init=True)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = model.energy_2x1_1x2(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy.to_number().item()}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()