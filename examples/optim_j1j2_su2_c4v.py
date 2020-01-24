import context
import copy
import torch
import argparse
import config as cfg
from su2sym.ipeps_su2 import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import j1j2
from optim.ad_optim_su2 import optimize_state
import su2sym.sym_ten_parser as tenSU2
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
args, unknown = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)

    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_su2(args.instate, vertexToSite=None)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"

        abd= args.bond_dim
        cbd= max(state.get_aux_bond_dims())
        if abd > cbd and abd in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors(2,abd ,"A_1",\
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)

            A= torch.zeros(len(su2sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)

            # get uuid lists
            uuid_orig=[t[0]["meta"]["name"].replace("T_","T_"+((abd-cbd)//2)*"0") for t in state.su2_tensors]
            uuid_new=[t[0]["meta"]["name"] for t in su2sym_t]
            print(f"{uuid_orig}")
            print(f"{uuid_new}")
            coeffs_orig=next(iter(state.coeffs.values()))
            for i,uuid in enumerate(uuid_orig):
                A[uuid_new.index(uuid)]=coeffs_orig[i]

            coeffs = {(0,0): A}
            state = IPEPS_SU2SYM(su2_tensors=su2sym_t, coeffs=coeffs)

        add_random_noise(state, args.instate_noise)
        state.coeffs[(0,0)]= state.coeffs[(0,0)]/torch.max(torch.abs(state.coeffs[(0,0)]))
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))

        A= torch.rand(len(su2sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))

        coeffs = {(0,0): A}

        state = IPEPS_SU2SYM(su2_tensors=su2sym_t, coeffs=coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    # initialize on-site tensors
    state.sites= state.build_onsite_tensors()
    print(state)

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history["log"]=[]
            rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
            dist= float('inf')
            if len(history["log"]) > 1:
                dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
            history["rdm"]=rdm2x1
            history["log"].append(dist)
        return dist<ctm_args.ctm_conv_tol

    if cfg.ctm_args.ctm_logging:
        env_log= args.out_prefix+"_env_log.json"
    else:
        env_log=None
    ctm_env = ENV_C4V(args.chi, state, log=env_log)
    init_env(state, ctm_env)
    ctm_env, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env, \
        conv_check=ctmrg_conv_energy)

    loss= energy_f(state, ctm_env, force_cpu=True)
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels)+", ctm-steps")
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values])+f", {len(history)}")

    def loss_fn(state, ctm_env_in, opt_args=cfg.opt_args):
        # build on-site tensors from su2sym components
        state.sites= state.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_energy)
        loss0 = energy_f(state, ctm_env_out, force_cpu=True)
        
        loc_ctm_args= copy.deepcopy(cfg.ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=True)

        loss=(loss0+loss1)/2

        return loss, ctm_env_out, history, t_ctm, t_obs

    # optimize
    optimize_state(state, ctm_env, loss_fn, model, args)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_su2(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env,force_cpu=True)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values])+f", {len(history)}")

if __name__=='__main__':
    main()