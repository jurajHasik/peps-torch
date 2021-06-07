import argparse
import numpy as np
import torch
import config as cfg
import yast
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import coupledLadders
from optim.su_abelian import run_seq_2s
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--alpha", type=float, default=0., help="inter-ladder coupling")
parser.add_argument("--bz_stag", type=float, default=0., help="staggered magnetic field")
parser.add_argument("--symmetry", default="None", help="symmetry structure", choices=["None","U1"])
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
# Alternative 
# --bond_dim, default=1, help="maximal bond dimension within SU")
# ADAPTIVE policy computes real (CTM) energy every SU step
#          and adjusts the time by SU_adaptive_slowdown_factor step if the
#          energy increases
parser.add_argument("--SU_policy", type=str, default="ADAPTIVE", help="SU policy", choices=["ADAPTIVE","REGULAR"])
parser.add_argument("--SU_init_step", type=float, default=0.1, help="intial SU (imaginary) time step")
parser.add_argument("--SU_ctm_obs_freq", type=int, default=0)
parser.add_argument("--SU_adaptive_slowdown_factor", type=float, default=0.5)
parser.add_argument("--SU_stop_cond", type=float, default=1.0e-6)
parser.add_argument("--SU_min_energy_diff", type=float, default=1.0e-8)
# --opt_max_iter, type=int, default=1000, help="maximal number of SU steps"
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config() 
    # TODO(?) choose symmetry group and override default dtype
    if not args.symmetry or args.symmetry=="None":
        settings= settings_full
        settings_full.dtype= settings.dtype= cfg.global_args.dtype
        model= coupledLadders.COUPLEDLADDERS_NOSYM(settings,alpha=args.alpha,Bz_val=args.bz_stag)
    elif args.symmetry=="U1":
        settings= settings_U1
        settings_full.dtype= settings.dtype= cfg.global_args.dtype
        model= coupledLadders.COUPLEDLADDERS_U1(settings,alpha=args.alpha,Bz_val=args.bz_stag)
        # model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha)
    # override default device specified in settings
    default_device= 'cpu' if not hasattr(settings, 'device') else settings.device
    if not cfg.global_args.device == default_device:
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print("Setting backend device: "+settings.device)
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

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
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha,Bz_val=args.bz_stag)

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x1_1x2(state, env).item()
        history.append(e_curr)

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    # 3) compute initial observables
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    loss= model.energy_2x1_1x2(state, ctm_env).to_number()
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","beta","time_step","energy"]+obs_labels))
    print(", ".join([f"{-1}","0","0",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in):
        # possibly re-initialize the environment
        if cfg.opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log = ctmrg.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)

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
        epoch= len(opt_context["loss_history"]["loss"]) 
        loss= opt_context["loss_history"]["loss"][-1]
        beta= opt_context["loss_history"]["beta"]
        ts= opt_context["loss_history"]["time_step"][-1]        
        obs_values, obs_labels = model.eval_obs(state,ctm_env)
        print(", ".join([f"{epoch}",f"{beta}",f"{ts}",f"{loss}"]+[f"{v}" for v in obs_values]))

        # with torch.no_grad():
        #     if args.top_freq>0 and epoch%args.top_freq==0:
        #         coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
        #         for c,d in coord_dir_pairs:
        #             # transfer operator spectrum
        #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
        #             l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
        #             print("TOP "+json.dumps(_to_json(l)))

    def generate_weights(state):
        #   
        #       w0         w2
        # w4--(0,0)--w5--(1,0)--[w4]
        #       w1         w3
        # w6--(0,1)--w7--(1,1)--[w6]
        #      [w0]       [w2]
        def neg_(dxy): return (-dxy[0],-dxy[1])
        def add_(coord,dxy): return (coord[0]+dxy[0],coord[1]+dxy[1])
        dxy_w_to_ind= dict({(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4})
        weights=dict()
        for coord in state.sites.keys():
            for dxy,ind in dxy_w_to_ind.items():
                # generate weight_id and reverse weight_id
                # (coord,dxy) identifies the same weight as (coord+dxy,-dxy) 
                w_id= (coord, dxy)
                w_rid= (state.vertexToSite(add_(coord,dxy)), neg_(dxy))

                if not w_id in weights.keys() and not w_rid in weights.keys():
                    W= yast.match_legs( tensors=[state.site(w_id[0]), state.site(w_rid[0])],\
                        legs=[dxy_w_to_ind[w_id[1]], dxy_w_to_ind[w_rid[1]]], isdiag=True )
                    weights[w_id]= W
                    weights[w_rid]= W
        return weights

    state_w= get_weighted_ipeps(state, generate_weights(state))

    # 4) select gate sequence
    # gen_gate_seq= model.gen_gate_seq_2S_2ndOrder       #  separate S.S and Sz.Id gates
    gen_gate_seq= model.gen_gate_seq_2S_SS_hz_2ndOrder # combined S.S + Sz.Id - Id.Sz gates
    # 5) enter simple update imaginary time evolution
    outputstatefile= args.out_prefix+"_state.json"
    opt_context= {"loss_history": {"loss":[loss], "beta": 0, \
        "time_step": [args.SU_init_step]}}
    su_opts={"weight_inv_cutoff": 1.0e-14, "max_D_total": args.bond_dim, \
        "log_level": 0}
    gate_seq_SS= gen_gate_seq(args.SU_init_step)
    for tstep in range(args.opt_max_iter):
        state_w= run_seq_2s(state_w, gate_seq_SS, su_opts)
        opt_context["loss_history"]["beta"] += opt_context["loss_history"]["time_step"][-1]

        if args.SU_policy=="ADAPTIVE" or (args.SU_ctm_obs_freq>0 \
            and tstep%args.SU_ctm_obs_freq==0):
            _tmp_state= state_w.absorb_weights()
            try:
                loss, ctm_env, *ctm_log= loss_fn(_tmp_state, ctm_env)
                opt_context["loss_history"]["loss"].append(loss)
                obs_fn(_tmp_state, ctm_env, opt_context)
            except Exception as err:
                _tmp_state.write_to_file(args.out_prefix+"_ERR_state.json")
                raise err

        # check CTM energy and depending on policy adjust time step
        _energy_criterion=-1.0
        if len(opt_context["loss_history"]["loss"])>2:
            _energy_criterion= opt_context["loss_history"]["loss"][-1] - \
                opt_context["loss_history"]["loss"][-2]

        if args.SU_policy=="ADAPTIVE" and _energy_criterion > -args.SU_min_energy_diff:
            # # rollback to previous state
            opt_context["loss_history"]["beta"] -= opt_context["loss_history"]["time_step"][-1]
            state= read_ipeps(outputstatefile, settings)
            state_w= get_weighted_ipeps(state, generate_weights(state))

            # generate new gate sequence
            new_ts= args.SU_adaptive_slowdown_factor * opt_context["loss_history"]["time_step"][-1] 
            opt_context["loss_history"]["time_step"].append(new_ts)
            gate_seq_SS= gen_gate_seq(new_ts)
            
        elif _energy_criterion < -args.SU_min_energy_diff:
            # store new state
            _tmp_state= state_w.absorb_weights()
            _tmp_state.write_to_file(outputstatefile)

        if args.SU_policy=="ADAPTIVE" and \
            opt_context["loss_history"]["time_step"][-1]< args.SU_stop_cond:
            break

    # absorb weights and generate a regular (symmetric) iPEPS
    state= state_w.absorb_weights()
    state.write_to_file(outputstatefile)

    # compute final observables
    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)
    loss, ctm_env, *ctm_log= loss_fn(state, ctm_env)
    opt_context["loss_history"]["loss"].append(loss)
    obs_fn(state, ctm_env, opt_context)

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
