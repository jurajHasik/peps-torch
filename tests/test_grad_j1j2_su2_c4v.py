import context
import time
import copy
import torch
import argparse
import config as cfg
from su2sym.ipeps_su2 import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from models import j1j2
import su2sym.sym_ten_parser as tenSU2
import logging
log = logging.getLogger(__name__)

if __name__=='__main__':
    # parse command line args and build necessary configuration objects
    parser= cfg.get_args_parser()
    # additional model-dependent arguments
    parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
    parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
    args = parser.parse_args()
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)

    model = j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
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
            state = IPEPS_SU2SYM(su2sym_t, coeffs)
            state.add_noise(args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors(f"D{args.bond_dim}.txt",2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))

        A= torch.rand(len(su2sym_t), dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))
        coeffs = {(0,0): A}
        state = IPEPS_SU2SYM(su2sym_t, coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    # initialize on-site tensors
    state.sites= state.build_onsite_tensors()
    print(state)
    
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)

    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=dict({"log": []})
            rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
            dist= float('inf')
            if len(history["log"]) > 1:
                dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
            history["rdm"]=rdm2x1
            history["log"].append(dist)
            if dist<ctm_args.ctm_conv_tol:
                log.info({"history_length": len(history['log']), "history": history['log'],
                    "final_multiplets": compute_multiplets(ctm_env)})
                return True, history
            elif len(history['log']) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history['log']), "history": history['log'],
                    "final_multiplets": compute_multiplets(ctm_env)})
                return False, history
        return False, history

    def loss_fn(state, ctm_env_in, opt_args=cfg.opt_args):
        # build on-site tensors from su2sym components
        state.sites= state.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f)
        loss0 = energy_f(state, ctm_env_out, force_cpu=True)
        
        loc_ctm_args= copy.deepcopy(cfg.ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        loss1 = energy_f(state, ctm_env_out, force_cpu=True)

        loss= torch.max(loss0,loss1)

        return loss, ctm_env_out, history, t_ctm, t_obs

    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    t_data = dict({"loss": [1.0e+16], "min_loss": 1.0e+16})
    current_env=[ctm_env]
    
    #@profile
    def closure(uuid):
        for el in parameters: 
            if el.grad is not None: el.grad.zero_()

        # 0) pre-process state: normalize on-site tensors by largest elements
        # with torch.no_grad():
        #     for c in state.coeffs.keys():
        #         state.coeffs[c]*= 1.0/torch.max(torch.abs(state.coeffs[c]))

        # 1) evaluate loss and the gradient
        loss, ctm_env, history, t_ctm, t_obs = loss_fn(state, current_env[0])

        # 4) log CTM metrics for debugging
        if cfg.opt_args.opt_logging:
            log_entry=dict({"id": uuid, \
                "t_ctm": t_ctm, "t_obs": t_obs})
            log.info(json.dumps(log_entry))

        # We evaluate observables inside closure as it is the only place with environment
        # consistent with the state
        # 2) compute observables if we moved into new epoch
        obs_values, obs_labels = model.eval_obs(state,ctm_env)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]
            +[f"{len(history['log'])}"]))

        if loss.requires_grad:
            t0= time.perf_counter()
            loss.backward()
            t1= time.perf_counter()

            # 5) log grad metrics
            if cfg.opt_args.opt_logging:
                log_entry=dict({"id": uuid+"_GRAD", "ctm_iter": len(history), "t_grad": t1-t0, 
                    "grad": [p.grad.tolist() for p in parameters],
                    "coeffs": [p.tolist() for p in parameters] })
                log.info(json.dumps(log_entry))

            # 6) detach current environment from autograd graph
            lst_C = list(ctm_env.C.values())
            lst_T = list(ctm_env.T.values())
            # current_env[0] = ctm_env
            for el in lst_T + lst_C: el.detach_()

        return loss

    epoch= -1
    loss0= energy_f(state, ctm_env, force_cpu=True)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels+["ctm_steps"]))
    print(", ".join([f"{epoch}",f"{loss0}"]+[f"{v}" for v in obs_values]+["0"]))
    loss0= closure("FULL_CTM")
    loss0.detach_()
    

    print("----- NO ENV REINIT -------------------------------------------------")
    eps=1e-4
    cfg.ctm_args.projector_svd_method= "SYMARP"
    cfg.opt_args.opt_ctm_reinit= False
    fd_grad=[]
    with torch.no_grad():
        for k in state.coeffs.keys():
            for i in range(state.coeffs[k].size()[0]):
                A_orig= state.coeffs[k].clone()
                e_i= torch.zeros(A_orig.size()[0],dtype=A_orig.dtype,device=A_orig.device)
                e_i[i]= eps
                state.coeffs[k]+=e_i
                loss= closure(f"FD_PARTIAL_CTM_{i}")
                print(f"{A_orig} {state.coeffs[k]}")
                print(f"{i} {loss} {(loss-loss0)/eps}")
                fd_grad.append(float(loss-loss0)/eps)
                state.coeffs[k]=A_orig
    print(f"FD_PARTIAL_CTM_GRAD {fd_grad}")
    if cfg.opt_args.opt_logging:
        log_entry=dict({"id": "FD_PARTIAL_CTM_GRAD",  
            "grad": [fd_grad],
            "coeffs": [state.coeffs[k].tolist() for k in state.coeffs.keys()] })
        log.info(json.dumps(log_entry))

    print("----- AD NO ENV REINIT ----------------------------------------------")
    parameters = [state.coeffs[k] for k in state.coeffs.keys()]
    for A in parameters: A.requires_grad_(True)
    cfg.ctm_args.projector_svd_method= "DEFAULT"
    cfg.opt_args.opt_ctm_reinit= False
    cfg.ctm_args.ctm_conv_tol= -1.
    for epoch in range(args.opt_max_iter):
        cfg.ctm_args.ctm_max_iter= max(1,epoch*5)
        closure(f"AD_PARTIAL_CTM_{epoch}")
