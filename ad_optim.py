import time
import json
import torch
from pytorch_memlab import mem_reporter
from memory_profiler import profile
import config as cfg
from ipeps import write_ipeps

def store_checkpoint(checkpoint_file, parameters, optimizer, current_epoch, current_loss,\
    verbosity=0):
    r"""
    :param checkpoint_file: target file
    :param parameters: wavefunction parameters
    :param optimizer: Optimizer
    :param current_epoch: current epoch
    :param current_loss: current value of a loss function
    :param verbosity: verbosity
    :type checkpoint_file: str or Path
    :type parameters: list[torch.tensor]
    :type optimizer: torch.optim.Optimizer
    :type current_epoch: int
    :type current_loss: float
    :type verbosity: int

    Store the current state of the optimization in ``checkpoint_file``.
    """
    torch.save({
            'epoch': current_epoch,
            'loss': current_loss,
            'parameters': parameters,
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file)
    if verbosity>0:
        print(checkpoint_file)

# A = torch.rand((phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
# A = 2 * (A - 0.5)
# A.requires_grad_(True)
# def zero_fn(coord): return (0,0)
# sites = {(0,0): A}
# state1 = ipeps.IPEPS(None, sites, zero_fn)

#     def loss_fn(state, model, ctm_args):
#         ctm_env = ENV(env_args,state1)
#         ctm_env = ctmrg.run(state, env, ctm_args=ctm_args, global_args=global_args)
#         energy = model.energy_1x1c4v(state1, ctm_env)
#         return energy

def optimize_state(state, ctm_env_init, loss_fn, model, local_args, opt_args=cfg.opt_args,\
    ctm_args=cfg.ctm_args, global_args=cfg.global_args):
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

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using LBFGS optimizer.
    The main parameters influencing the optimization process are given in :py:class:`args.OPTARGS`.
    """
    parameters = list(state.sites.values())
    for A in parameters: A.requires_grad_(True)

    outputstatefile= local_args.out_prefix+"_state.json"
    outputlogfile= open(local_args.out_prefix+"_log.json",mode="w",buffering=1)
    t_data = dict({"loss": [1.0e+16], "min_loss": 1.0e+16})
    eval_counter=[0]
    prev_epoch=[-1]
    current_env=[ctm_env_init]
    
    @profile
    def closure():
        for el in parameters: 
            if el.grad is not None: el.grad.zero_()

        # 0) pre-process state: normalize on-site tensors by largest elements
        for coord,site in state.sites.items():
            site = site/torch.max(torch.abs(site))


        # 1) evaluate loss and the gradient
        loss, ctm_env, history, t_ctm = loss_fn(state, current_env[0], opt_args=opt_args)
        t0= time.perf_counter()
        loss.backward(retain_graph=False)
        t1= time.perf_counter()

        # We evaluate observables inside closure as it is the only place with environment
        # consistent with the state
        if prev_epoch[0]!=epoch:
            # 2) compute observables if we moved into new epoch
            obs_values, obs_labels = model.eval_obs(state,ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

            # 3) store current state if the loss improves
            t_data["loss"].append(loss.item())
            if t_data["min_loss"] > t_data["loss"][-1]:
                t_data["min_loss"]= t_data["loss"][-1]
                write_ipeps(state, outputstatefile, normalize=True)

        # 4) log additional metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": len(t_data["loss"])-1, \
                "t_grad": t1-t0, "t_ctm": t_ctm, \
                "ctm_history_len": len(history), "ctm_history": history})
            outputlogfile.write(json.dumps(log_entry)+'\n')

        # 5) detach current environment from autograd graph
        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        current_env[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()
        del ctm_env

        eval_counter[0]+=1
        prev_epoch[0]=epoch
        return loss

    verbosity = opt_args.verbosity_opt_epoch
    outputstatefile = local_args.out_prefix+"_state.json"
    checkpoint_file = local_args.out_prefix+"_checkpoint.p"
    optimizer = torch.optim.LBFGS(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.tolerance_grad, tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size)
    epoch0 = 0
    loss0 = 0

    if local_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {local_args.opt_resume}")
        checkpoint = torch.load(local_args.opt_resume)
        init_parameters = checkpoint["parameters"]
        epoch0 = checkpoint["epoch"]
        loss0 = checkpoint["loss"]
        for i in range(len(parameters)):
            parameters[i].data = init_parameters[i].data
        cp_state_dict= checkpoint["optimizer_state_dict"]
        cp_opt_params= cp_state_dict["param_groups"][0]
        cp_opt_history= cp_state_dict["state"][cp_opt_params["params"][0]]
        if local_args.opt_resume_override_params:
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

    for epoch in range(local_args.opt_max_iter):
        # checkpoint the optimizer
        # checkpointing before step, guarantees the correspondence between the wavefunction
        # and the last computed value of loss t_data["loss"][-1]
        store_checkpoint(checkpoint_file, parameters, optimizer, epoch0+epoch, t_data["loss"][-1])
        
        # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # the ``state`` on-site tensors have been modified by gradient. 
        loss = optimizer.step(closure)
