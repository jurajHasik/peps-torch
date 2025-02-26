import copy
import warnings
import time
import json
import logging
log = logging.getLogger(__name__)
import torch
from optim import lbfgs_modified
import config as cfg

def store_checkpoint(checkpoint_file, state, optimizer, current_epoch, current_loss,\
    verbosity=0):
    r"""
    :param checkpoint_file: target file
    :param state: ipeps wavefunction
    :param optimizer: Optimizer
    :param current_epoch: current epoch
    :param current_loss: current value of a loss function
    :param verbosity: verbosity
    :type checkpoint_file: str or Path
    :type state: IPEPS
    :type optimizer: torch.optim.Optimizer
    :type current_epoch: int
    :type current_loss: float
    :type verbosity: int

    Store the current state of the optimization in ``checkpoint_file``.
    """
    torch.save({
            'epoch': current_epoch,
            'loss': current_loss,
            'parameters': state.get_checkpoint(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file)
    if verbosity>0:
        print(checkpoint_file)

def optimize_state(state, ctm_env_init, loss_fn, obs_fn=None, post_proc=None,
    main_args=cfg.main_args, opt_args=cfg.opt_args,ctm_args=cfg.ctm_args, 
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
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args, "loss_history": t_data})
    epoch=0

    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.tolerance_grad, tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
        line_search_eps=opt_args.line_search_tol)

    # load and/or modify optimizer state from checkpoint
    if main_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {main_args.opt_resume}")
        if not str(global_args.device)==str(state.device):
            warnings.warn(f"Device mismatch: state.device {state.device}"\
                +f" global_args.device {global_args.device}",RuntimeWarning)
        checkpoint = torch.load(main_args.opt_resume,map_location=state.device, weights_only=False)
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
    def closure(linesearching=False):
        context["line_search"]= linesearching
        
        # 0) evaluate loss
        optimizer.zero_grad()
        loss, ctm_env, history, *timings = loss_fn(state, current_env[0], context)
        if len(timings)==2: # legacy
            t_ctm, t_check, t_loss= *timings, None
        elif len(timings)==3: # 
            t_ctm, t_check, t_loss= timings

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()

        # 6) detach current environment from autograd graph
        ctm_env.detach_()
        current_env[0]= ctm_env
        # current_env[0]= ctm_env.detach().clone()

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
                    "t_check": t_check, "t_loss": t_loss})
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
        
        loss, ctm_env, history, *timings = loss_fn(state, current_env[0],\
            ls_context)
        if len(timings)==2: # legacy
            t_ctm, t_check, t_loss= *timings, None
        elif len(timings)==3: # 
            t_ctm, t_check, t_loss= timings
        current_env[0]= ctm_env

        # 2) store current state if the loss improves
        t_data["loss_ls"].append(loss.item())
        if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
            t_data["min_loss_ls"]= t_data["loss_ls"][-1]

        # 3) log metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "LS": len(t_data["loss_ls"]), \
                "loss": t_data["loss_ls"], "t_ctm": t_ctm, "t_check": t_check, "t_loss": t_loss})
            log.info(json.dumps(log_entry))

        # 4) compute desired observables
        if obs_fn is not None:
            obs_fn(state, current_env[0], context)

        return loss

    for epoch in range(main_args.opt_max_iter):
        # checkpoint the optimizer
        # checkpointing before step, guarantees the correspondence between the wavefunction
        # and the last computed value of loss t_data["loss"][-1]
        if epoch>0:
            store_checkpoint(checkpoint_file, state, optimizer, epoch, t_data["loss"][-1])

        # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # the ``state`` on-site tensors have been modified by gradient. 
        optimizer.step_2c(closure, closure_linesearch)
        
        # reset line search history
        t_data["loss_ls"]=[]
        t_data["min_loss_ls"]=1.0e+16

        if post_proc is not None:
            post_proc(state, current_env[0], context)

        # terminate condition
        if len(t_data["loss"])>1 and \
            abs(t_data["loss"][-1]-t_data["loss"][-2])<opt_args.tolerance_change:
            break

        if (opt_args.line_search not in ["default", None]) and \
            "STATUS" in context and context["STATUS"]=="ENV_ANTIVAR":
            raise RuntimeError("Over-optimized environment, see log file for "\
                +" env_sensitivity and loss_diff.")


    # optimization is over, store the last checkpoint if at least a single step was made
    if len(t_data["loss"])>0:
        store_checkpoint(checkpoint_file, state, optimizer, \
            main_args.opt_max_iter, t_data["loss"][-1])
