import copy
import warnings
import time
import json
import logging
log = logging.getLogger(__name__)
import torch
import config as cfg
from optim import lbfgs_modified
from .ad_optim_lbfgs_mod import store_checkpoint

def optimize_state(state, ctm_env_init, loss_fn_vtnr, loss_fn_grad, 
    obs_fn=None, post_proc=None,
    main_args=cfg.main_args, opt_args=cfg.opt_args, ctm_args=cfg.ctm_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_init: initial environment corresponding to ``state``
    :param loss_fn: loss function
    :param model: model with definition of observables
    :param local_args: parsed command line arguments
    :param opt_args: optimization configuration
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,CTMARGS,OPTARGS,GLOBALARGS)->torch.tensor
    :type model: TODO Model base class
    :type local_args: argparse.Namespace
    :type opt_args: OPTARGS
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using 
    :class:`optim.lbfgs_modified.LBFGS_MOD` optimizer.
    The main parameters influencing the optimization process are given in :class:`config.OPTARGS`.
    Calls to functions ``loss_fn``, ``obs_fn``, and ``post_proc`` pass the current configuration
    as dictionary ``{"ctm_args":ctm_args, "opt_args":opt_args}``

    The optimizer saves the best energy state into file ``main_args.out_prefix+"_state.json"``
    and checkpoints the optimization at every step to ``main_args.out_prefix+"_state.json"``.
    """
    verbosity = opt_args.verbosity_opt_epoch
    checkpoint_file = main_args.out_prefix+"_checkpoint.p"
    outputstatefile= main_args.out_prefix+"_state.json"
    
    t_data = dict({"loss": [], "min_loss": 1.0e+16, "loss_ls": [], "min_loss_ls": 1.0e+16})
    current_env= [ctm_env_init]
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args, "loss_history": t_data,\
        "vtnr_success": True, "vtnr_timeout_counter": 0, "vtnr_timeout": False})
    epoch=0

    # generators 
    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    # isometries
    old_isometries= [None]*len(state.isometries)
    p_isometries= [None]*len(state.isometries)

    optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, \
        lr=opt_args.lr, tolerance_grad=opt_args.tolerance_grad, \
        tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
        line_search_eps=opt_args.line_search_tol)

    # load and/or modify optimizer state from checkpoint
    if main_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {main_args.opt_resume}")
        if not str(global_args.device)==str(state.device):
            warnings.warn(f"Device mismatch: state.device {state.device}"\
                +f" global_args.device {global_args.device}",RuntimeWarning)
        checkpoint = torch.load(main_args.opt_resume,map_location=state.device)
        epoch0 = checkpoint["epoch"]
        loss0 = checkpoint["loss"]
        cp_state_dict= checkpoint["optimizer_state_dict"]
        cp_opt_params= cp_state_dict["param_groups"][0]
        cp_opt_history= cp_state_dict["state"][cp_opt_params["params"][0]]
        if main_args.opt_resume_override_params:
            cp_opt_params["lr"] = opt_args.lr
            cp_opt_params["max_iter"] = opt_args.max_iter_per_epoch
            cp_opt_params["tolerance_grad"] = opt_args.tolerance_grad
            cp_opt_params["tolerance_change"] = opt_args.tolerance_change
            cp_opt_params["line_search_fn"] = opt_args.line_search
            cp_opt_params["line_search_eps"] = opt_args.line_search_tol
            # resize stored old_dirs, old_stps, ro, al to new history size
            cp_history_size= cp_opt_params["history_size"]
            cp_opt_params["history_size"] = opt_args.history_size
            if opt_args.history_size < cp_history_size:
                if len(cp_opt_history["old_dirs"]) > opt_args.history_size: 
                    assert len(cp_opt_history["old_dirs"])==len(cp_opt_history["old_stps"])\
                        ==len(cp_opt_history["ro"]), "Inconsistent L-BFGS history"
                    cp_opt_history["old_dirs"]= cp_opt_history["old_dirs"][-opt_args.history_size:]
                    cp_opt_history["old_stps"]= cp_opt_history["old_stps"][-opt_args.history_size:]
                    cp_opt_history["ro"]= cp_opt_history["ro"][-opt_args.history_size:]
            cp_al_filtered= list(filter(None,cp_opt_history["al"]))
            if len(cp_al_filtered) > opt_args.history_size:
                cp_opt_history["al"]= cp_al_filtered[-opt_args.history_size:]
            else:
                cp_opt_history["al"]= cp_al_filtered + [None for i in range(opt_args.history_size-len(cp_al_filtered))]
        cp_state_dict["param_groups"][0]= cp_opt_params
        cp_state_dict["state"][cp_opt_params["params"][0]]= cp_opt_history
        optimizer.load_state_dict(cp_state_dict)
        print(f"checkpoint.loss = {loss0}")

    #@profile
    def closure_vtnr(linesearching=False):
        context["line_search"]= linesearching
        for i in range(len(p_isometries)):
            p_isometries[i]= state.isometries[i]
        for A in p_isometries:
            if not A.requires_grad: A.requires_grad_(True)
            if not A.grad is None:
                A.grad= 0 * A.grad

        # 0) evaluate loss
        loss, ctm_env, history, t_ctm, t_check = loss_fn_vtnr(state, current_env[0], context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()

        # 6) detach current environment from autograd graph
        ctm_env.detach_()
        current_env[0]= ctm_env
        # current_env[0]= ctm_env.detach().clone()

        # 1) record loss and store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > t_data["loss"][-1]:
            t_data["min_loss"]= t_data["loss"][-1]
            state.write_to_file(outputstatefile, normalize=True)
            context["vtnr_success"]=True
        else:
            context["vtnr_success"]=False
        print(f"closure_vtnr vtnr_success {context['vtnr_success']}")

        # 2) log CTM metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "loss": t_data["loss"][-1], "t_ctm": t_ctm, \
                    "t_check": t_check})

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, current_env[0], context)

        # 5) log grad metrics
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch})
            log_entry["t_grad"]=t_grad1-t_grad0
            # log just l2 and l\infty norm of the full grad
            # log_entry["grad_mag"]= [p.grad.norm().item() for p in parameters]
            flat_grad= torch.cat(tuple(p.grad.view(-1) for p in p_isometries))
            log_entry["grad_mag"]= [flat_grad.norm().item(), flat_grad.norm(p=float('inf')).item()]
            if opt_args.opt_log_grad: log_entry["grad"]= [p.grad.tolist() for p in p_isometries]
            log.info(json.dumps(log_entry))

        return loss
    
    def closure(linesearching=False):
        context["line_search"]= linesearching
        loc_opt_args= copy.deepcopy(opt_args)
        if not context["vtnr_success"]:
            # loc_opt_args.opt_ctm_reinit= False
            pass
        # _context= dict({"ctm_args":ctm_args, "opt_args":loc_opt_args, "loss_history": t_data,
        #     "line_search": linesearching})

        # 0) evaluate loss
        optimizer.zero_grad()
        loss, ctm_env, history, t_ctm, t_check = loss_fn_grad(state, current_env[0], context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()
        # rescale ? NOTE: specific to TTN ansatz
        # for i,p in enumerate(parameters):
        #     p.grad*= 1./(2**(len(parameters)-1-i))

        # 6) detach current environment from autograd graph
        ctm_env.detach_()
        current_env[0]= ctm_env

        # 1) record loss and store current state if the loss improves
        if linesearching:
            t_data["loss_ls"].append(loss.item())
            if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
                t_data["min_loss_ls"]= t_data["loss_ls"][-1]
        else:  
            t_data["loss"].append(loss.item())
            if t_data["min_loss"] > t_data["loss"][-1]:
                t_data["min_loss"]= t_data["loss"][-1]
                state.write_to_file(outputstatefile, normalize=True)

        # 2) log CTM metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "loss": t_data["loss"][-1], "t_ctm": t_ctm, \
                    "t_check": t_check})
            if linesearching:
                log_entry["LS"]=len(t_data["loss_ls"])
                log_entry["loss"]=t_data["loss_ls"]
            log.info(json.dumps(log_entry))

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, current_env[0], context)

        # 5) log grad metrics
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch})
            if linesearching: log_entry["LS"]=len(t_data["loss_ls"])
            else: 
                log_entry["t_grad"]=t_grad1-t_grad0
                # log just l2 and l\infty norm of the full grad
                # log_entry["grad_mag"]= [p.grad.norm().item() for p in parameters]
                flat_grad= torch.cat(tuple(p.grad.view(-1) for p in parameters))
                log_entry["grad_mag"]= [flat_grad.norm().item(), flat_grad.norm(p=float('inf')).item()]
                log_entry["grad_mags"]= [p.grad.norm().item() for p in parameters]
                if opt_args.opt_log_grad: log_entry["grad"]= [p.grad.tolist() for p in parameters]
            log.info(json.dumps(log_entry))        

        return loss

    # closure for derivative-free line search. This closure
    # is to be called within torch.no_grad context
    @torch.no_grad()
    def closure_linesearch(linesearching):
        context["line_search"]=linesearching

        # 1) evaluate loss
        loc_opt_args= copy.deepcopy(opt_args)
        loc_opt_args.opt_ctm_reinit= opt_args.line_search_ctm_reinit
        loc_ctm_args= copy.deepcopy(ctm_args)

        if opt_args.line_search_svd_method != 'DEFAULT':
            loc_ctm_args.projector_svd_method= opt_args.line_search_svd_method
        ls_context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args, "loss_history": t_data,
            "line_search": linesearching})
        
        loss, ctm_env, history, t_ctm, t_check = loss_fn_grad(state, current_env[0],\
            ls_context)
        current_env[0]= ctm_env

        # 2) store current state if the loss improves
        t_data["loss_ls"].append(loss.item())
        if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
            t_data["min_loss_ls"]= t_data["loss_ls"][-1]

        # 3) log metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "LS": len(t_data["loss_ls"]), \
                "loss": t_data["loss_ls"], "t_ctm": t_ctm, "t_check": t_check})
            log.info(json.dumps(log_entry))

        # 4) compute desired observables
        if obs_fn is not None:
            obs_fn(state, current_env[0], ls_context)

        return loss

    def step_vtnr(closure_vtnr):
        # evaluate loss and obtain the environments of isometries as gradients
        loss0= closure_vtnr()

        if not context["vtnr_success"]:
            # revert back isometries
            with torch.no_grad():
                for i in range(len(p_isometries)):
                    p_isometries[i].copy_(old_isometries[i])
                    # p_isometries[i]= old_isometries[i]
        else:
            # force update on generators
            with torch.no_grad():
                state.right_inverse_()

            new_iso= []
            for p in p_isometries:
                U,S,Vh= torch.linalg.svd(p.grad.view(p.size(0)*p.size(1),p.size(2)),\
                    full_matrices=False)
                new_iso.append( (U@Vh).view(p.size()) )
            
            with torch.no_grad():
                for i in range(len(p_isometries)):
                    old_isometries[i]= p_isometries[i].detach().clone()
                    if p_isometries[i].size(0)*p_isometries[i].size(1)>p_isometries[i].size(2):
                        p_isometries[i].copy_(new_iso[i])

    for epoch in range(main_args.opt_max_iter):
        # checkpoint the optimizer
        # checkpointing before step, guarantees the correspondence between the wavefunction
        # and the last computed value of loss t_data["loss"][-1]
        if epoch>0:
            store_checkpoint(checkpoint_file, state, optimizer, epoch, t_data["loss"][-1])

        # 1) attempt VTNR step
        if context["vtnr_timeout_counter"]==0: 
            context["vtnr_timeout"]=False
        
        if not context["vtnr_timeout"]:
            step_vtnr(closure_vtnr)
        else:
            context["vtnr_timeout_counter"]+= -1

        # 2) if VTNR failed, timeout VTNR for X steps
        if not context["vtnr_success"] and not context["vtnr_timeout"]:
            context["vtnr_timeout"]= True
            context["vtnr_timeout_counter"]= 10

            # if len(t_data["loss"])>3 and t_data["loss"][-2]<t_data["loss"][-3]:
            # # reset L-BFGS only after at least one successful VTNR step
            #     print(f"L-BFGS reset")
            #     optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, \
            #         lr=opt_args.lr, tolerance_grad=opt_args.tolerance_grad, \
            #         tolerance_change=opt_args.tolerance_change, \
            #         history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
            #         line_search_eps=opt_args.line_search_tol)

        # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # the ``state`` on-site tensors have been modified by gradient. 
        if not context["vtnr_success"]:
            optimizer.step_2c(closure, closure_linesearch)
        
            # reset line search history
            t_data["loss_ls"]=[]
            t_data["min_loss_ls"]=1.0e+16

        # if post_proc is not None:
        #     post_proc(state, current_env[0], context)

        # terminate condition
        if len(t_data["loss"])>1 and \
            abs(t_data["loss"][-1]-t_data["loss"][-2])<opt_args.tolerance_change:
            break

    # optimization is over, store the last checkpoint if at least a single step was made
    if len(t_data["loss"])>0:
        store_checkpoint(checkpoint_file, state, optimizer, \
            main_args.opt_max_iter, t_data["loss"][-1])
