
########################################################################################################

import torch
import numpy as np


import os, sys
sys.path.append('/tuph/t30/bigspace/ge74suj/PEPS_torch/PEPS_torch/')
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from optim.ad_optim_lbfgs_mod import optimize_state
import logging
from models.toric_code import TORICCODE_1x1
log = logging.getLogger(__name__)


########################################################################################################

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--jv", type=float, default=1.0000, help="vertex terms")
parser.add_argument("--jp", type=float, default=1.0000, help="plaquette terms")
parser.add_argument("--hx", type=float, default=0, help="horizontal magnetic field")
parser.add_argument("--hz", type=float, default=0, help="vertical magnetic field")

args, unknown_args = parser.parse_known_args()
########################################################################################################
def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    model = TORICCODE_1x1(jv=args.jv,
                              jp=args.jp,
                              hx=args.hx,
                              hz=args.hz,
                              global_args=cfg.global_args
                              )
    if args.instate != None:
        state = read_ipeps(args.instate, vertexToSite=None)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        state = IPEPS(dict(), lX=1, lY=1)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type == 'RANDOM':
        bond_dim = args.bond_dim
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), \
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        # normalization of initial random tensors
        A = A / torch.max(torch.abs(A))
        sites = {(0, 0): A}
        state = IPEPS(sites)
    elif args.ipeps_init_type == 'TC_fixed_point':

        sites = {(0, 0): model.FPT_TC().contiguous()}
        state = IPEPS(sites)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(noise=0.05)
        state.normalize_()
        print('D=', state.get_aux_bond_dims())
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= " \
                         + str(args.ipeps_init_type) + " is not supported")

    print(state)
    @torch.no_grad()
    def ctmrg_conv_f(state, env, history,ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            C=env.C
            T=env.T
            norm_C=[]
            norm_T=[]
            for keys,value in C.items():
                norm_C.append((torch.linalg.norm(value)).item())
            for keys,value in T.items():
                norm_T.append((torch.linalg.norm(value)).item())

            history.append(norm_C+norm_T)
            if len(history) > 1:
                error=abs(sum(history[-1])-sum(history[-2]))
                print('CTMRG_step=',len(history),'CTMRG_error=',error,'chi=',(C[((0, 0), (-1, -1))]).shape[0],'D^2=',(T[((0, 0), (0, -1))]).shape[1])

            if len(history) > 1 and error < ctm_args.ctm_conv_tol:
                return True, history
        return False, history
    ########################################################################################################

    #args.out_prefix = 'data/TC_non_sym_hz_' + str(args.hz) + '_hx_' +  f"{0.6:.2f}"  + '_bond_dim_' + str(args.bond_dim) + '_chi_' + str(args.chi)
    energy_f = model.energy
    eval_obs_f = model.eval_obs
    eval_cth_f = model.eval_corrlen
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0 = energy_f(state, ctm_env)
    obs_values, obs_labels = eval_obs_f(state, ctm_env)
    print(", ".join(["epoch", "energy"] + obs_labels))
    print(", ".join([f"{-1}", f"{loss0:.16f}"] + [f"{v:.16f}" for v in obs_values]), flush=True)





    print("initial CTMRG optimization", flush=True)






    ########################################################################################################
    def loss_fn(state, ctm_env_in, opt_context):

                ctm_args= opt_context["ctm_args"]
                opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
                if opt_args.opt_ctm_reinit:
                    init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
                ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
                loss = energy_f(state, ctm_env_out)

                return (loss, ctm_env_out, *ctm_log)



    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
            with torch.no_grad():
                epoch= len(opt_context["loss_history"]["loss"])
                loss= opt_context["loss_history"]["loss"][-1]
                obs_values, obs_labels = eval_obs_f(state ,ctm_env)
                try:
                    print(", ".join([f"ipeps epoch: {epoch:6d}",
                             f"loss: {loss:.6f}",
                             f"obs: " + ", ".join([f"{v:.6f}" for v in obs_values])
                             ]), flush=True)
                except OSError:
                    pass

                log.info({"epoch": epoch, "loss": loss, "obs": obs_values})

            with torch.no_grad():
                if args.top_freq >0 and epoch%args.top_freq==0:
                    obs_cths = eval_cth_f(state, ctm_env)
                # lengths = [value.item() for value in obs_cths['lengths'].values()]
                    lengths = [v.item() for sublist in obs_cths['lengths'].values() for v in sublist]
                    print(f"corr len: " + ", ".join([f"{v:.6f}" for v in lengths]), flush=True)
                    log.info({"correlation_lengths": lengths})

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)


    ########################################################################################################

    print("- " *80)
    print("final measurements", flush=True)
    log.info({"- " *80})
    log.info({"final_measurements"})

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)

    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0 = energy_f(state ,ctm_env)
    obs_values, obs_labels = eval_obs_f(state ,ctm_env)

    print(", ".join([f"ipeps epoch: {args.opt_max_iter:6d}",
                     f"loss: {loss0:.6f}",
                     f"grad: {loss0:.6f}",
                     f"obs: " + ", ".join([f"{v:.6f}" for v in obs_values])
                     ]), flush=True)
    log.info({"epoch": args.opt_max_iter, "loss": loss0, "obs": obs_values})

    obs_cths = eval_cth_f(state, ctm_env)
    lengths = [v.item() for sublist in obs_cths['lengths'].values() for v in sublist]
    print(f"corr len: " + ", ".join([f"{v:.6f}" for v in lengths]), flush=True)
    log.info({"correlation_lengths": lengths})
    print('updape ini state')

if __name__=='__main__':
    args.hx = 0.3
    args.hz = 0.2
    args.bond_dim = 2
    args.chi = 40
    args.opt_max_iter = 3
    args.top_freq = 10      #frequency of the top spectra
    args.opt_max_iter=100    #number of iterations
    args.ipeps_init_type = 'TC_fixed_point'
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

