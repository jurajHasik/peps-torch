import time
import json
import logging
log = logging.getLogger(__name__)
import torch
#from memory_profiler import profile
from optim import lbfgs_modified
import config as cfg

def itevol_plaquette_step(state1, loss_fn, post_proc=None, main_args=cfg.main_args,
    opt_args=cfg.opt_args,ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_init: initial environment corresponding to ``state``
    :param loss_fn: loss function
    :param main_args: parsed command line arguments
    :param opt_args: optimization configuration
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,CTMARGS,OPTARGS,GLOBALARGS)->torch.tensor
    :type main_args: MAINARGS
    :type opt_args: OPTARGS
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using LBFGS optimizer.
    The main parameters influencing the optimization process are given in :py:class:`config.OPTARGS`.
    """
    verbosity = opt_args.verbosity_opt_epoch
    outputstatefile= main_args.out_prefix+"_state.json"
    t_data = dict({"loss": [], "min_loss": 1.0e+16, "loss_ls": [], "min_loss_ls": 1.0e+16,
        "grad_max": []})
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args, "loss_history": t_data})
    _current_diag= dict({"n1": 0, "overlap1U0": 0})

    parameters= state1.get_parameters()
    for A in parameters: A.requires_grad_(True)

    optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.itevol_tolerance_grad, tolerance_change=opt_args.itevol_tolerance_change, \
        history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
        line_search_eps=opt_args.tol_line_search)

    #@profile
    def closure():
        context["line_search"]=False

        # 0) evaluate loss
        optimizer.zero_grad()
        loss, g, n1, overlap1U0= loss_fn(state1, context)
        _current_diag["n1"]= n1.item()
        _current_diag["overlap1U0"]= overlap1U0.item()

        # 1) store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > t_data["loss"][-1]:
            t_data["min_loss"]= t_data["loss"][-1]

        # 2) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()
        t_data["grad_max"].append(max([p.grad.abs().max() for p in parameters]).item())

        # dims= g.size()
        # for i in range(dims[0]*(dims[1]**4)):
        #     print(f"{state1.site().grad.view(dims[0]*(dims[1]**4))[i]} {g.view(dims[0]*(dims[1]**4))[i]}")
        # print(f"g_ad-g_exact diff {(state1.site().grad-2.*g).norm()}")

        return loss
    
    # closure for derivative-free line search. This closure
    # is to be called within torch.no_grad context
    def closure_linesearch():
        context["line_search"]=True

        # 1) evaluate loss
        loss, g, n1, overlap1U0= loss_fn(state1, context)
        _current_diag["n1"]= n1.item()
        _current_diag["overlap1U0"]= overlap1U0.item()

        # 2) store current state if the loss improves
        t_data["loss_ls"].append(loss.item())
        if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
            t_data["min_loss_ls"]= t_data["loss_ls"][-1]

        return loss
        
    def log_progress(epoch,final=False):
        if (epoch>1 and epoch%(opt_args.itevol_max_iter/100)==0) or final: 
            log.info(f"{epoch} loss {t_data['loss'][-1]} n1 {_current_diag['n1']}"\
                + f" overlap1U0 {_current_diag['overlap1U0']} norm {state1.site().norm()}"\
                + f" {t_data['loss'][-1] - t_data['loss'][-2]} {t_data['grad_max'][-1]}")

    for epoch in range(opt_args.itevol_max_iter):
        optimizer.step_2c(closure, closure_linesearch)

        # externalize optimization termination conditions here
        if len(t_data["loss"])>1 and \
            abs(t_data["loss"][-1] - t_data["loss"][-2]) < opt_args.itevol_tolerance_change:
            break
        if t_data["grad_max"][-1] <= opt_args.itevol_tolerance_grad:
            break
        log_progress(epoch)

        # reset line search history
        t_data["loss_ls"]=[]
        t_data["min_loss_ls"]=1.0e+16

        if post_proc is not None:
            post_proc(state1, context)
    log_progress(epoch,final=True)

    # turn of autograd
    for A in parameters: A.requires_grad_(False)