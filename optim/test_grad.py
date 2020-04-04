import copy
import time
import json
import logging
log = logging.getLogger(__name__)
import torch
import config as cfg

def test_grad_ad(state, ctm_env_in, chis, loss_fn, 
    main_args=cfg.main_args, ctm_args=cfg.ctm_args, opt_args=cfg.opt_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_in: initial environment
    :param chis: list of environment dimensions
    :param loss_fn: loss function
    :param main_args: parsed command line arguments
    :param ctm_args: CTM algorithm configuration
    :param opt_args: optimization configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type chis: list(int)
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,CTMARGS,OPTARGS,GLOBALARGS)->torch.tensor
    :type main_args: MAINARGS
    :type ctm_args: CTMARGS
    :type opt_args: OPTARGS
    :type global_args: GLOBALARGS

    Computes AD gradient of ``loss_fn`` with respect to ``state`` for selected 
    environment bond dimensions``chis``. The first choice ``chis[0]`` is taken
    as a reference for post-analysis.
    The main parameters influencing the CTM process are given in :py:class:`config.CTMARGS`.
    """
    past_env=[ctm_env_in]
    grad_ad= dict()
    context= dict({"ctm_args":ctm_args, "opt_args":opt_args})
    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    def _gather_flat_grad(_params):
        views = []
        for p in _params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    for i,chi in enumerate(chis):
        # clean gradients
        for A in parameters: A.grad= None
        # pdb.set_trace()

        # extend environment
        current_env= ctm_env_in.extend(chi, ctm_args=ctm_args, global_args=global_args)

        # compute loss
        loss, ctm_env, history, t_ctm, t_check = loss_fn(state, current_env, context)

        # 2) log CTM metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": f"test_grad_ad-{i}", "chi": chi, "t_ctm": t_ctm, \
                "t_check": t_check}) 
            log.info(json.dumps(log_entry))

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()

        # 4a) flatten and record grad
        grad_ad[chi]= _gather_flat_grad(parameters)

        # 5) log grad metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": f"test_grad_ad-{i}", "chi": chi, "t_grad": t_grad1-t_grad0})
            log.info(json.dumps(log_entry))

        # 6) detach current environment from autograd graph
        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        past_env[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()

    # analyze grad
    ref_g= grad_ad[chis[0]]
    print(f"test_grad_ad g0 X= {chis[0]} |ref_g|= {torch.norm(ref_g)}")

    # assume single param tensor
    for c in chis[1:]:
        g= grad_ad[c]
        overlap= g.dot(ref_g)/(torch.norm(g)*torch.norm(ref_g))
        print(f"test_grad_ad g X={c} |g|= {torch.norm(g)} |g-g0|= {torch.norm(g-ref_g)}"\
            +f" g.g0= {g.dot(ref_g)} angle= {overlap}")

    return grad_ad

def test_grad_fd(state, ctm_env_in, chis, loss_fn, 
    main_args=cfg.main_args, ctm_args=cfg.ctm_args, opt_args=cfg.opt_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_in: initial environment
    :param chis: list of environment dimensions
    :param loss_fn: loss function
    :param main_args: parsed command line arguments
    :param ctm_args: CTM algorithm configuration
    :param opt_args: optimization configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type chis: list(int)
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,CTMARGS,OPTARGS,GLOBALARGS)->torch.tensor
    :type main_args: MAINARGS
    :type ctm_args: CTMARGS
    :type opt_args: OPTARGS
    :type global_args: GLOBALARGS

    Computes finite-difference gradient of ``loss_fn`` with respect to ``state`` for selected 
    environment bond dimensions``chis``. The first choice ``chis[0]`` is taken
    as a reference for post-analysis.
    The main parameters influencing the CTM process are given in :py:class:`config.CTMARGS`,    
    while parameters influencing finite-difference gradients are given 
    in :py:class:`config.OPTARGS`.
    """

    grad_fd= dict()
    past_env= [ctm_env_in]
    loc_opt_args= copy.deepcopy(opt_args)
    loc_opt_args.opt_ctm_reinit= opt_args.line_search_ctm_reinit
    loc_ctm_args= copy.deepcopy(ctm_args)
    if opt_args.line_search_svd_method != 'DEFAULT':
        loc_ctm_args.projector_svd_method= opt_args.line_search_svd_method
    context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args})

    def eval_grad_fd(state, ctm_env_in, loss_fn):
        fd_grad=[]
        with torch.no_grad():
            loss0, ctm_env0, history, t_ctm, t_check = loss_fn(state, ctm_env_in,\
                context)

            for t in state.get_parameters():
                fd_grad.append(torch.zeros(t.numel(),dtype=t.dtype,device=t.device))
                t_orig= t.clone()
                for i in range(t.numel()):
                    e_i= torch.zeros(t.numel(),dtype=t.dtype,device=t.device)
                    e_i[i]= opt_args.fd_eps
                    t+=e_i.view(t.size())
                    loss1, ctm_env1, history, t_ctm, t_check = loss_fn(state, ctm_env0,\
                        context)
                    fd_grad[-1][i]= (float(loss1-loss0)/opt_args.fd_eps)
                    log.info(f"FD_GRAD {i} loss1 {loss1} grad_i {fd_grad[-1][i]}"\
                        +f" t_ctm {t_ctm} t_check {t_check}")
                    t.copy_(t_orig)
                fd_grad[-1]= fd_grad[-1].view(t.size())
        log.info(f"FD_GRAD grad {fd_grad}")

        return fd_grad, ctm_env0

    def _flatten_grad(_grad):
        views = []
        for g in _grad:
            view = g.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    for i,chi in enumerate(chis):
        # extend environment
        current_env= ctm_env_in.extend(chi, ctm_args=ctm_args, global_args=global_args)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        grad, current_env= eval_grad_fd(state, current_env, loss_fn)
        t_grad1= time.perf_counter()

        # 4a) flatten and record grad
        grad_fd[chi]= _flatten_grad(grad)

        # 5) log grad metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": f"test_grad_fd-{i}", "chi": chi, "t_grad": t_grad1-t_grad0})
            log.info(json.dumps(log_entry))

        past_env[0]= current_env

    # analyze grad
    ref_g= grad_fd[chis[0]]
    print(f"test_grad_fd g0 X= {chis[0]} |ref_g|= {torch.norm(ref_g)}")

    # assume single param tensor
    for c in chis[1:]:
        g= grad_fd[c]
        overlap= g.dot(ref_g)/(torch.norm(g)*torch.norm(ref_g))
        print(f"test_grad_fd g X={c} |g|= {torch.norm(g)} |g-g0|= {torch.norm(g-ref_g)}"\
            +f" g.g0= {g.dot(ref_g)} angle= {overlap}")

    return grad_fd