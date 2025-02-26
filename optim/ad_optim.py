import time
import json
import logging
log = logging.getLogger(__name__)
import torch
#from memory_profiler import profile
import config as cfg

def store_checkpoint(checkpoint_file, state, optimizer, current_epoch, current_loss,\
    verbosity=0):
    r"""
    :param checkpoint_file: target file
    :param state: ipeps wavefunction
    :param optimizer: Optimizer
    :param current_epoch: current epoch
    :param current_loss: current value of a loss function
    :type checkpoint_file: str or Path
    :type state: IPEPS
    :type optimizer: torch.optim.Optimizer
    :type current_epoch: int
    :type current_loss: float

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
    main_args=cfg.main_args, opt_args=cfg.opt_args, ctm_args=cfg.ctm_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_init: initial environment of ``state``
    :param loss_fn: loss function
    :param obs_fn: optional function to evaluate observables
    :param post_proc: optional function for post-processing the state and environment  
    :param main_args: main configuration
    :param opt_args: optimization configuration
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,dict)->torch.tensor
    :type obs_fn: function(IPEPS,ENV,dict)->None
    :type post_proc: function(IPEPS,ENV,dict)->None
    :type main_args: MAINARGS
    :type opt_args: OPTARGS
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using 
    `LBFGS optimizer <https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS>`_.
    The main parameters influencing the optimization process are given in :class:`config.OPTARGS`.
    Calls to functions ``loss_fn``, ``obs_fn``, and ``post_proc`` pass the current configuration
    as dictionary ``{"ctm_args":ctm_args, "opt_args":opt_args}``.
    """
    verbosity = opt_args.verbosity_opt_epoch
    checkpoint_file = main_args.out_prefix+"_checkpoint.p"   
    outputstatefile= main_args.out_prefix+"_state.json"
    t_data = dict({"loss": [], "min_loss": float('inf')})
    current_env=[ctm_env_init]
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args, "loss_history": t_data})
    epoch= 0

    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    optimizer = torch.optim.LBFGS(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.tolerance_grad, tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size)

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
    def closure():
        # 0) evaluate loss
        optimizer.zero_grad()
        loss, ctm_env, history, t_ctm, t_check = loss_fn(state, current_env[0], context)

        # 1) store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > t_data["loss"][-1]:
            t_data["min_loss"]= t_data["loss"][-1]
            state.write_to_file(outputstatefile, normalize=True)

        # 2) log CTM metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "loss": t_data["loss"][-1], "t_ctm": t_ctm, \
                    "t_check": t_check})
            log.info(json.dumps(log_entry))

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, ctm_env, context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()

        # 5) log grad metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "t_grad": t_grad1-t_grad0})
            log_entry["grad_mag"]= [p.grad.norm().item() for p in parameters]
            if opt_args.opt_log_grad: log_entry["grad"]= [p.grad.tolist() for p in parameters]
            log.info(json.dumps(log_entry))

        # 6) detach current environment from autograd graph
        current_env[0] = ctm_env.detach().clone()

        return loss
    
    for epoch in range(main_args.opt_max_iter):
        # checkpoint the optimizer
        # checkpointing before step, guarantees the correspondence between the wavefunction
        # and the last computed value of loss t_data["loss"][-1]
        if epoch>0:
            store_checkpoint(checkpoint_file, state, optimizer, epoch, t_data["loss"][-1])

        # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # the ``state`` on-site tensors have been modified by gradient. 
        optimizer.step(closure)
        
        if post_proc is not None:
            post_proc(state, current_env[0], context)

        # terminate condition
        if len(t_data["loss"])>1 and \
            abs(t_data["loss"][-1]-t_data["loss"][-2])<opt_args.tolerance_change:
            break

    # optimization is over, store the last checkpoint
    store_checkpoint(checkpoint_file, state, optimizer, \
        main_args.opt_max_iter, t_data["loss"][-1])
