import copy
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

def optimize_state(state, ctm_env_init, loss_fn, grad_fn,
    obs_fn=None, post_proc=None,
    main_args=cfg.main_args, opt_args=cfg.opt_args,ctm_args=cfg.ctm_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_init: initial environment corresponding to ``state``
    :param loss_fn: loss function
    :param model: model with definition of observables
    :param main_args: parsed command line arguments
    :param opt_args: optimization configuration
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,CTMARGS,OPTARGS,GLOBALARGS)->torch.tensor
    :type model: TODO Model base class
    :type main_args: argparse.Namespace
    :type opt_args: OPTARGS
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using LBFGS optimizer.
    The main parameters influencing the optimization process are given in :py:class:`config.OPTARGS`.
    """
    verbosity = opt_args.verbosity_opt_epoch
    checkpoint_file = main_args.out_prefix+"_checkpoint.p"   
    outputstatefile= main_args.out_prefix+"_state.json"
    t_data = dict({"loss": [], "min_loss": 1.0e+16, "loss_ls": [], "min_loss_ls": 1.0e+16})
    current_env=[ctm_env_init]
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args, "loss_history": t_data})
    epoch= 0

    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, \
        lr=opt_args.lr, tolerance_grad=opt_args.tolerance_grad, \
        tolerance_change=opt_args.tolerance_change, history_size=opt_args.history_size, \
        line_search_fn=opt_args.line_search, line_search_eps=opt_args.line_search_tol)

    # load and/or modify optimizer state from checkpoint
    if main_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {main_args.opt_resume}")
        checkpoint = torch.load(main_args.opt_resume, weights_only=False)
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
            # resize stored old_dirs, old_stps, ro, al to new history size
            cp_history_size= cp_opt_params["history_size"]
            cp_opt_params["history_size"] = opt_args.history_size
            if opt_args.history_size < cp_history_size:
                if len(cp_opt_history["old_dirs"]) > opt_args.history_size: 
                    cp_opt_history["old_dirs"]= cp_opt_history["old_dirs"][-opt_args.history_size:]
                    cp_opt_history["old_stps"]= cp_opt_history["old_stps"][-opt_args.history_size:]
            cp_ro_filtered= list(filter(None,cp_opt_history["ro"]))
            cp_al_filtered= list(filter(None,cp_opt_history["al"]))
            if len(cp_ro_filtered) > opt_args.history_size:
                cp_opt_history["ro"]= cp_ro_filtered[-opt_args.history_size:]
                cp_opt_history["al"]= cp_al_filtered[-opt_args.history_size:]
            else:
                cp_opt_history["ro"]= cp_ro_filtered + [None for i in range(opt_args.history_size-len(cp_ro_filtered))]
                cp_opt_history["al"]= cp_al_filtered + [None for i in range(opt_args.history_size-len(cp_ro_filtered))]
        cp_state_dict["param_groups"][0]= cp_opt_params
        cp_state_dict["state"][cp_opt_params["params"][0]]= cp_opt_history
        optimizer.load_state_dict(cp_state_dict)
        print(f"checkpoint.loss = {loss0}")

    #@profile
    def closure(linesearching=False):
        context["line_search"]=linesearching

        # 0) evaluate loss
        optimizer.zero_grad()
        with torch.no_grad():
            loss, ctm_env, history, timings= loss_fn(state, current_env[0], context)

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
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": history["final_multiplets"]})
            log_entry=dict({"id": epoch, "loss": t_data["loss"][-1], "timings": timings})
            if linesearching:
                log_entry["LS"]=len(t_data["loss_ls"])
                log_entry["loss"]=t_data["loss_ls"]
            log.info(json.dumps(log_entry))

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, ctm_env, context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        with torch.no_grad():
            grad= grad_fn(state, ctm_env, context, loss)
            for k in state.coeffs.keys():
                state.coeffs[k].grad= grad[k]
        t_grad1= time.perf_counter()

        # 5) log grad metrics
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "t_grad": t_grad1-t_grad0 })
            if linesearching: log_entry["LS"]=len(t_data["loss_ls"])
            log.info(json.dumps(log_entry))

        # 6) detach current environment from autograd graph
        current_env[0] = ctm_env.detach().clone()

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
        # TODO check if we are optimizing C4v symmetric ansatz
        if opt_args.line_search_svd_method != 'DEFAULT':
            loc_ctm_args.projector_svd_method= opt_args.line_search_svd_method
        loc_context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args, \
            "loss_history": t_data, "line_search": True})
        loss, ctm_env, history, timings = loss_fn(state, current_env[0],\
            loc_context)

        # 2) store current state if the loss improves
        t_data["loss_ls"].append(loss.item())
        if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
            t_data["min_loss_ls"]= t_data["loss_ls"][-1]

        # 5) log CTM metrics for debugging
        if opt_args.opt_logging:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": history["final_multiplets"]})
            log_entry=dict({"id": epoch, "LS": len(t_data["loss_ls"]), \
                "loss": t_data["loss_ls"], "timings": timings})
            log.info(json.dumps(log_entry))

        # 4) compute desired observables
        if obs_fn is not None:
            obs_fn(state, ctm_env, context)

        current_env[0]= ctm_env
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

    # optimization is over, store the last checkpoint
    store_checkpoint(checkpoint_file, state, optimizer, \
        main_args.opt_max_iter, t_data["loss"][-1])