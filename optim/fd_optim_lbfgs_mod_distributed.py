import os
import copy
import time
import json
import logging
log = logging.getLogger(__name__)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
#from memory_profiler import profile
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

    optimizer = lbfgs_modified.LBFGS_MOD(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.tolerance_grad, tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size, line_search_fn=opt_args.line_search, \
        line_search_eps=opt_args.line_search_tol)

    # load and/or modify optimizer state from checkpoint
    if main_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {main_args.opt_resume}")
        checkpoint = torch.load(main_args.opt_resume)
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

    # check if distributed available
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")

    N=100
    mv_result= torch.zeros(N)
    mv_result.share_memory_()
    def manager_code(rank,size,queue):
        print(f"rank {rank} starting MANAGER with N {N}")
        a= torch.zeros(N,N) 
        c= torch.zeros(N)
        dummy_tensor= torch.zeros(1)
        dummy_tensor[0]= -1

        sender, row, numsent = 0, 0, 0 
        # recv does not expose information about tag
        tmp_send= torch.zeros(1+N)
        tmp_result= torch.zeros(2)
        status= -1

        # (arbitrary) initialization of a
        for i in range(N):
            for j in range(N):
                a[i,j]= j

        for i in range(1,min( size, N )):
            # tensor (Tensor) – Tensor to send.
            # dst (int) – Destination rank.
            # group (ProcessGroup, optional) – The process group to work on
            # tag (int, optional) – Tag to match send with remote recv
            tmp_send[0]= i-1
            tmp_send[1:]= a[i-1,:]
            print(f"rank {rank} sending row {i-1} to rank {i}")
            dist.send( tensor=tmp_send, dst=i, tag=i)
            numsent+=1
        
        # receive dot products back from workers
        print(f"rank {rank} entering receive stage")
        for i in range(N):
            # tensor (Tensor) – Tensor to fill with received data.
            # src (int, optional) – Source rank. Will receive from any process if unspecified.
            # group (ProcessGroup, optional) – The process group to work on
            # tag (int, optional) – Tag to match recv with remote send
            sender= dist.recv( tmp_result )
            if sender<0:
                raise RuntimeError(f"Error in recv, unexpected sender {sender}")
            print(f"rank {rank} received result from {sender}")
            row= int(tmp_result[0])
            c[row]= tmp_result[1]
            # send another row back to this worker if there is one
            if numsent < N:
                tmp_send[0]= numsent
                tmp_send[1:]= a[numsent,:]
                print(f"rank {rank} sending row {numsent} to rank {sender}")
                dist.send( tensor=tmp_send, dst=sender )
                print(c)
                numsent+=1
            else: 
                # no more work
                dist.send( dummy_tensor, sender )
        print("FINAL")
        print(c)
        for i in range(N): mv_result[i]= c[i]

    def worker_code(rank,size):
        print(f"rank {rank} starting WORKER with N {N}")
        b= torch.zeros(N)
        c= torch.zeros(1+N)
        row= 0
        tmp_result= torch.zeros(2)
        status= -1

        for i in range(N): 
            # (arbitrary) b initialization
            b[i]= 1.0

        if rank <= N:
            print(f"rank {rank} open for receiving")
            dist.recv( tensor=c, src=0, tag=rank )
            status= int(c[0])
            print(f"rank {rank} received row {status}")
            
            while status >= 0:
                row= status
                tmp_result[0]= row
                tmp_result[1]= 0.
                for i in range(N):
                    tmp_result[1] += c[i+1] * b[i]
                print(f"rank {rank} sending result of row {row}")
                dist.send( tensor=tmp_result, dst=0 )
                dist.recv( tensor=c, src=0 )
                status= int(c[0])
                print(f"rank {rank} receiving new column {status}")
            print(f"rank {rank} received terminating signal")

    def init_process(fn, rank, size, *args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, *args)

    queue= mp.Queue()
    size= 3
    processes= []
    for rank in range(size):
        if rank==0:
            p = mp.Process(target=init_process, args=(manager_code, rank, size, queue))
        else:
            p = mp.Process(target=init_process, args=(worker_code, rank, size))
        # p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(mv_result)

    def grad_fd(loss0):
        loc_opt_args= copy.deepcopy(opt_args)
        loc_opt_args.opt_ctm_reinit= opt_args.fd_ctm_reinit
        loc_ctm_args= copy.deepcopy(ctm_args)
        # TODO check if we are optimizing C4v symmetric ansatz
        if opt_args.line_search_svd_method != 'DEFAULT':
            loc_ctm_args.projector_svd_method= opt_args.line_search_svd_method
        loc_context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args, 
            "loss_history": t_data, "line_search": False})

        # compute components of the grad
        fd_grad=dict()
        with torch.no_grad():
            for k in state.coeffs.keys():
                A_orig= state.coeffs[k].clone()
                assert len(A_orig.size())==1, "coefficient tensor is not 1D"
                fd_grad[k]= torch.zeros(A_orig.size(),dtype=A_orig.dtype,device=A_orig.device)
                for i in range(state.coeffs[k].size()[0]):                  
                    e_i= torch.zeros(A_orig.size()[0],dtype=A_orig.dtype,device=A_orig.device)
                    e_i[i]= opt_args.fd_eps
                    state.coeffs[k]+=e_i
                    loss1, ctm_env, history, timings = loss_fn(state, current_env[0],\
                        loc_context)
                    fd_grad[k][i]=(float(loss1-loss0)/opt_args.fd_eps)
                    log.info(f"FD_GRAD {i} loss1 {loss1} grad_i {fd_grad[k][i]}"\
                        +f" t_ctm {timings['t_ctm']} t_check {timings['t_obs']}"\
                        +f" t_energy {timings['t_energy']}")
                    state.coeffs[k].data.copy_(A_orig)
        log.info(f"FD_GRAD grad {fd_grad}")

        return fd_grad

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
            log_entry=dict({"id": epoch, "loss": t_data["loss"][-1], "t_ctm": timings['t_ctm'], \
                    "t_check": timings['t_obs'], "t_energy": timings['t_energy']})
            if linesearching:
                log_entry["LS"]=len(t_data["loss_ls"])
                log_entry["loss"]=t_data["loss_ls"]
            log.info(json.dumps(log_entry))

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, ctm_env, context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        grad= grad_fd(loss)
        for k in state.coeffs.keys():
            state.coeffs[k].grad= grad[k]
        t_grad1= time.perf_counter()

        # 5) log grad metrics
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "t_grad": t_grad1-t_grad0 })
            if linesearching: log_entry["LS"]=len(t_data["loss_ls"])
            log.info(json.dumps(log_entry))

        # 6) detach current environment from autograd graph
        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        current_env[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()

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
        loc_context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args, "loss_history": t_data,
            "line_search": True})
        loss, ctm_env, history, timings = loss_fn(state, current_env[0],\
            loc_context)

        # 2) store current state if the loss improves
        t_data["loss_ls"].append(loss.item())
        if t_data["min_loss_ls"] > t_data["loss_ls"][-1]:
            t_data["min_loss_ls"]= t_data["loss_ls"][-1]

        # 5) log CTM metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "LS": len(t_data["loss_ls"]), \
                "loss": t_data["loss_ls"], "t_ctm": timings['t_ctm'], \
                "t_check": timings['t_obs'], "t_energy": timings['t_energy']})
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