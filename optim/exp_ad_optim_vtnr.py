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
        "step_history": [], \
        "vtnr_state": "INIT", "vtnr_timeout_counter": 0, "vtnr_timeout": False})
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
            if len(context["step_history"])>1 and context["step_history"][-1]!="vtnr":
                # no vtnr update was done yet, just re-evaluation of energy from previous
                # grad step
                context["vtnr_state"]="RESUME"
            else:
                context["vtnr_state"]="SUCCESS"
        else:
            context["vtnr_state"]="FAIL"

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

    def closure_sweep(linesearching=False):
        context["line_search"]= linesearching
        for i in range(len(p_isometries)):
            p_isometries[i]= state.isometries[i]
        for A in p_isometries:
            if not A.requires_grad: A.requires_grad_(True)
            if not A.grad is None:
                A.grad= 0 * A.grad
                
        # 0) evaluate loss
        with torch.no_grad():
            loss, ctm_env, history, t_ctm, t_check = loss_fn_vtnr(state, current_env[0], context)

        # 6) detach current environment from autograd graph
        ctm_env.detach_()
        current_env[0]= ctm_env

        # 1) record loss and store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > t_data["loss"][-1]:
            t_data["min_loss"]= t_data["loss"][-1]
            # import pdb; pdb.set_trace()
            state.write_to_file(outputstatefile, normalize=True)
        #     if len(context["step_history"])>1 and context["step_history"][-1]!="vtnr":
        #         # no vtnr update was done yet, just re-evaluation of energy from previous
        #         # grad step
        #         context["vtnr_state"]="RESUME"
        #     else:
        #         context["vtnr_state"]="SUCCESS"
        # else:
        #     context["vtnr_state"]="FAIL"

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
            # log_entry["t_grad"]=t_grad1-t_grad0
            # log just l2 and l\infty norm of the full grad
            # log_entry["grad_mag"]= [p.grad.norm().item() for p in parameters]
            # flat_grad= torch.cat(tuple(p.grad.view(-1) for p in p_isometries))
            # log_entry["grad_mag"]= [flat_grad.norm().item(), flat_grad.norm(p=float('inf')).item()]
            # if opt_args.opt_log_grad: log_entry["grad"]= [p.grad.tolist() for p in p_isometries]
            log.info(json.dumps(log_entry))

        return loss
    
    def closure(linesearching=False):
        context["line_search"]= linesearching

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

        if context["vtnr_state"]=="FAIL":
            # revert back isometries
            with torch.no_grad():
                for i in range(len(p_isometries)):
                    p_isometries[i].copy_(old_isometries[i])
                    # p_isometries[i]= old_isometries[i]
        if context["vtnr_state"]=="SUCCESS":
            with torch.no_grad():
                state.right_inverse_()
        if context["vtnr_state"]=="SUCCESS" or context["vtnr_state"]=="RESUME":
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

        closure_sweep()
        from ctm.one_site_c4v.rdm_c4v import ddA_rdm1x1
        E= ddA_rdm1x1(state.to_fused_ipeps_c4v(), current_env[0])
        E= E.view([state.site().size(0),state.site().size(1)]+[state.site().size(2)]*4)
        _tmp_Es= down_sweep(state,E)
        _tmp_Es= _tmp_Es[1:]+[E]
        up_sweep(state,_tmp_Es)

        # # 1) attempt VTNR step
        # if context["vtnr_timeout_counter"]==0: 
        #     context["vtnr_state"]="INIT"
        
        # if not context["vtnr_state"]=="TIMEOUT":
        #     step_vtnr(closure_vtnr)
        #     context["step_history"].append('vtnr')
        # else:
        #     context["vtnr_timeout_counter"]+= -1

        # # 2) if VTNR failed, timeout VTNR for X steps
        # if context["vtnr_state"]=="FAIL":
        #     context["vtnr_state"]= "TIMEOUT"
        #     context["vtnr_timeout_counter"]= opt_args.vtnr_timeout

        #     # if there was at least one successful step of VTNR, reset L-BFGS
        #     if len(context["step_history"])>3 and context["step_history"][-3:]==["vtnr","vtnr","vtnr"]:
        #         optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, \
        #             lr=opt_args.lr, tolerance_grad=opt_args.tolerance_grad, \
        #             tolerance_change=opt_args.tolerance_change, \
        #             history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
        #             line_search_eps=opt_args.line_search_tol)

        # # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # # the ``state`` on-site tensors have been modified by gradient. 
        # if context["vtnr_state"]=="TIMEOUT":
        #     optimizer.step_2c(closure, closure_linesearch)
        #     context["step_history"].append('grad')
        
        #     # reset line search history
        #     t_data["loss_ls"]=[]
        #     t_data["min_loss_ls"]=1.0e+16

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

def down_sweep(state,E_TM):
    # taking E_TM and T_Mm1 and W compute env of W
    # perform SVD and built updated W

    # 0) precompute
    As= [state.seed_site]
    for i in range(len(state.isometries)):
        A= As[-1].clone()
        U= state.isometries[i]
        #
        #              a
        #            |/
        #     /--tmp_A--\
        # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
        #     \x   b   y/
        #
        tmp_A_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
        #
        #             a   u
        #              \ /
        #               U
        #             |/
        #      x--tmp_A--y
        #            /|
        #        b\ /
        #          U
        #         /
        #        d
        #
        tmp_A_ud= torch.einsum('amu,bnd,spmxny->spauxbdy',U,U,A)
        As.append(torch.einsum('skalxbry,kpauxbdy->spuldr',tmp_A_lr,tmp_A_ud).contiguous())
    
    E= E_TM
    intermediate_E= [None]*len(state.isometries)
    for i in range(len(state.isometries)-1,-1,-1):
        U= state.isometries[i]
        A= As[i]
        #`1) compute env of (up) isometry
        
        #            |/
        #     /--tmp_A--\
        # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
        #     \x       y/
        tmp_A_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
        #   
        #                --0(p)  0(p)---------
        #               /                   __|__
        #   2(m-up)----|E|--3(left)  3(l)--|     |--2(a-up)
        #   4(n-down)--|_|--5(right) 6(r)--|tmp_A|--5(b-down)
        #               |                  |     |--4(x-left)
        #               |                  |_____|--7(y-right)
        #               \--1(s)               1(k) 
        #
        tmp_E= torch.einsum('psmlnr,pkalxbry->ksmaxnby',E,tmp_A_lr)
        #   
        #                           0(k)
        #                           |
        #                           |
        #              --3(a-up)----|---6(b-down)--
        #   2(m-up)--               0                --5(n-down)
        #              --2(u-up)----A---5(d-down)--
        #                           1
        #                           1(s)
        tmp_E= torch.einsum('ksmaxnby,ksuxdy->maunbd',tmp_E,A)
        # 
        # a--|\--m
        # u--|/
        tmp_E= torch.einsum('maunbd,bdn->aum',tmp_E,U).contiguous()
        D_tmp_E= tmp_E.size()

        # compute new isometry
        U,S,Vh= torch.linalg.svd(tmp_E.view(D_tmp_E[0]*D_tmp_E[1],D_tmp_E[2]),\
                    full_matrices=False)
        state.isometries[i]= (U@Vh).view(D_tmp_E)
        # new_iso.append( (U@Vh).view(D_tmp_E) )

        U= state.isometries[i]
        #
        #              a
        #            |/
        #     /--tmp_A--\
        # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
        #     \x   b   y/
        #
        tmp_A_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
        tmp_E= torch.einsum('psmlnr,pkalxbry->ksmaxnby',E,tmp_A_lr)
        intermediate_E[i]= torch.einsum('psmaxnby,aum,bdn->psuxdy',tmp_E,U,U)
        E= intermediate_E[i]
        # NOTE: the first (i=0) environment is the environment of the seed_site itself
    return intermediate_E

def up_sweep(state, intermediate_E):
    A= state.seed_site.clone()
    for i in range(len(state.isometries)):
        U= state.isometries[i]
        E= intermediate_E[i]
        #            |/
        #     /--tmp_A--\
        # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
        #     \x       y/
        tmp_A_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
        #   
        #                --0(p)  0(p)---------
        #               /                   __|__
        #   2(m-up)----|E|--3(left)  3(l)--|     |--2(a-up)
        #   4(n-down)--|_|--5(right) 6(r)--|tmp_A|--5(b-down)
        #               |                  |     |--4(x-left)
        #               |                  |_____|--7(y-right)
        #               \--1(s)               1(k) 
        #
        tmp_E= torch.einsum('psmlnr,pkalxbry->ksmaxnby',E,tmp_A_lr)
        #   
        #                           0(k)
        #                           |
        #                           |
        #              --3(a-up)----|---6(b-down)--
        #   2(m-up)--               0                --5(n-down)
        #              --2(u-up)----A---5(d-down)--
        #                           1
        #                           1(s)
        tmp_E= torch.einsum('ksmaxnby,ksuxdy->maunbd',tmp_E,A)
        # 
        # a--|\--m
        # u--|/
        tmp_E= torch.einsum('maunbd,bdn->aum',tmp_E,U).contiguous()
        D_tmp_E= tmp_E.size()

        # compute new isometry
        U,S,Vh= torch.linalg.svd(tmp_E.view(D_tmp_E[0]*D_tmp_E[1],D_tmp_E[2]),\
                    full_matrices=False)
        state.isometries[i]= (U@Vh).view(D_tmp_E)

        U= state.isometries[i]
        #
        #              a
        #            |/
        #     /--tmp_A--\
        # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
        #     \x   b   y/
        #
        tmp_B_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
        #
        #             a   u
        #              \ /
        #               U
        #             |/
        #      x--tmp_A--y
        #            /|
        #        b\ /
        #          U
        #         /
        #        d
        #
        tmp_B_ud= torch.einsum('amu,bnd,spmxny->spauxbdy',U,U,A)
        A= torch.einsum('skalxbry,kpauxbdy->spuldr',tmp_B_lr,tmp_B_ud).contiguous()