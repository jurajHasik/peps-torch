import os
import copy
import torch
import argparse
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_abelian_c4v_lc import *
from models.abelian import j1j2
from ctm.one_site_c4v_abelian.env_c4v_abelian import *
from ctm.one_site_c4v_abelian import ctmrg_c4v
from ctm.one_site_c4v_abelian.rdm_c4v import rdm2x1
import torch.multiprocessing as mp
import torch.distributed as dist
from optim.fd_optim_lbfgs_mod_distributed import optimize_state
import time
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--workers", type=int, default=0, help="number of worker processes")
parser.add_argument("--force_cpu", action="store_true", help="force energy and observable evaluation on CPU")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--top_freq", type=int, default=-1, help="frequency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--obs_freq", type=int, default=-1, help="frequency of computing observables"
    + " during CTM convergence")
parser.add_argument("--symmetry", default=None, help="symmetry structure", choices=["NONE","U1"])
parser.add_argument("--yast_backend", type=str, default='torch', 
    help="YAST backend", choices=['torch','torch_cpp'])
args, unknown_args = parser.parse_known_args()

@torch.no_grad()
def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
    if not history:
        # first element counts the number of iterations executed, rest hold current distance
        history_count= 0
        history_log= torch.full((ctm_args.ctm_max_iter,), 1.0e+16, \
            dtype=next(iter(state.coeffs.values())).dtype, device=state.device)
        previous_rdm= torch.zeros(1, dtype=next(iter(state.coeffs.values())).dtype, \
            device=state.device)
        history= (history_count, history_log, previous_rdm)
    history_count= history[0]

    rdm= rdm2x1(state, env, force_cpu=ctm_args.conv_check_cpu, \
        verbosity=cfg.ctm_args.verbosity_rdm)
    if history_count > 0:
        history[1][history_count]= (rdm - history[2]).norm().item()
    history_count += 1
    history= (history_count, history[1], rdm)

    if history[1][history_count-1]<ctm_args.ctm_conv_tol:
        history= dict(log= history[1][:history_count], \
            final_multiplets=env.compute_multiplets()[1])
        # log.info({"history_length": len(history['log']), "history": history['log'],
        #         "final_multiplets": history["final_multiplets"]})
        return True, history
    elif history_count >= ctm_args.ctm_max_iter:
        history= dict(log= history[1][:history_count], \
            final_multiplets=env.compute_multiplets()[1])
        # log.info({"history_length": len(history['log']), "history": history['log'],
        #         "final_multiplets": history["final_multiplets"]})
        return False, history
    return False, history

def loss_functional(energy_f, state, ctm_env, opt_context):
    ctm_args= opt_context["ctm_args"]
    opt_args= opt_context["opt_args"]

    # build on-site tensors from su2sym components
    state.sites[(0,0)]= state.build_onsite_tensors()
    state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')()

    # possibly re-initialize the environment
    if opt_args.opt_ctm_reinit:
        init_env(state, ctm_env)

    # 1) compute environment by CTMRG
    t0_ctm_main= time.perf_counter()
    ctm_env, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env, \
        conv_check=ctmrg_conv_f, ctm_args=ctm_args)
    t1_ctm_main= time.perf_counter()
    t0_energy= time.perf_counter()
    loss0 = energy_f(state, ctm_env, force_cpu=args.force_cpu)
    t1_energy= time.perf_counter()
    
    loc_ctm_args= copy.deepcopy(ctm_args)
    loc_ctm_args.ctm_max_iter= 1
    ctm_env, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env, \
        ctm_args=loc_ctm_args)
    t2_energy= time.perf_counter()
    loss1 = energy_f(state, ctm_env, force_cpu=args.force_cpu)
    t3_energy= time.perf_counter()

    timings= dict({"t_ctm_main": t1_ctm_main-t0_ctm_main, "t_ctm": t_ctm,\
        "t_obs": t_obs, "t_energy": (t1_energy-t0_energy)+(t3_energy-t2_energy)})
    #loss=(loss0+loss1)/2
    loss= torch.max(loss0,loss1)

    return loss, ctm_env, history, timings

def grad_fd_component(loss_fn, state, ctm_env, opt_args, ctm_args, loss0, \
    coeff_key_id, grad_id, log_file):
    loc_opt_args= copy.deepcopy(opt_args)
    loc_opt_args.opt_ctm_reinit= opt_args.fd_ctm_reinit
    loc_ctm_args= copy.deepcopy(ctm_args)
    # TODO check if we are optimizing C4v symmetric ansatz
    if opt_args.line_search_svd_method != 'DEFAULT':
        loc_ctm_args.projector_svd_method= opt_args.line_search_svd_method
    t_data = dict({"loss": [], "min_loss": 1.0e+16, "loss_ls": [], "min_loss_ls": 1.0e+16})
    loc_context= dict({"ctm_args":loc_ctm_args, "opt_args":loc_opt_args, 
        "loss_history": t_data, "line_search": False})

    # compute single component of the grad
    coeff_key= list(state.coeffs.keys())[coeff_key_id]
    with torch.no_grad():
        A_orig= state.coeffs[coeff_key].clone()
        assert len(A_orig.size())==1, "coefficient tensor is not 1D"
        
        # 0) add perturbation
        e_i= torch.zeros(A_orig.size()[0],dtype=A_orig.dtype,device=A_orig.device)
        e_i[grad_id]= opt_args.fd_eps
        state.coeffs[coeff_key]+=e_i

        # 1) re-evaluate the energy and compute gradient
        loss1, ctm_env, history, timings = loss_fn(state, ctm_env,\
            loc_context)
        grad_val=(loss1.cpu()-loss0)/opt_args.fd_eps
        log_file.write(f"history_length: {len(history['log'])}, history: {history['log']},"\
                +f" final_multiplets: {history['final_multiplets']}\n")
        log_file.write(f"FD_GRAD id {coeff_key} {grad_id} loss1 {loss1} grad_val {grad_val}"\
            +f" timings {timings}\n")

        # 2) reset perturbation
        state.coeffs[coeff_key].data.copy_(A_orig)
    return grad_val

def manager_code(rank,size,state,ctm_env,pipes_to_workers,tasks,full_grad,loss0):
    if size<2: raise RuntimeError(f"MANAGER {rank} requires at least one WORKER - size {size}")
    # assume full_grad has all tensors in shared memory
    # 1.1) tasks= [(coeff_key_id)x(local_grad_id)]
    log.info(f"MANAGER {rank} starting for {len(tasks)} tasks "\
        +f"- CPU {mp.cpu_count()} this process ")

    _terminate_grad= torch.full((2,), -1, dtype=torch.long, device='cpu')
    _terminate_worker= torch.full((2,), -2, dtype=torch.long, device='cpu')
    if len(tasks)<1:
        # terminate workers
        for i in range(1, size):
            dist.send( _terminate_worker, dst=i, tag=i )
        return 0

    C_loc_cpu_meta, C_loc_cpu_raw1d = ctm_env.get_C().compress_to_1d()
    T_loc_cpu_meta, T_loc_cpu_raw1d = ctm_env.get_T().compress_to_1d()
    coeff_loc= state.coeffs[(0,0)].cpu()
    loss0_t= torch.zeros(1, dtype=C_loc_cpu_raw1d.dtype, device='cpu')
    loss0_t[0]= loss0

    # 2) Assuming every worker has the necessary tensors to perform 
    #    energy evaluation, issue the first set of gradient components to
    #    compute
    numsent= 0
    grad_id= torch.zeros(2, dtype=torch.long, device='cpu')
    for i in range(0, min(size-1, len(tasks))):
        # 0) sent initial grad_id, invoking new grad computation
        grad_id[0]= tasks[numsent][0]
        grad_id[1]= tasks[numsent][1]
        log.info(f"MANAGER {rank} sending id {grad_id} to WORKER {i+1}")
        dist.send( tensor=grad_id, dst=i+1, tag=i+1 )
        numsent+=1

        # 1) send energy to workers
        log.info(f"MANAGER {rank} sending loss0 to rank {i+1} tag {3000*size+i+1}")
        dist.send( tensor=loss0_t, dst=i+1, tag=3000*size+i+1 )
        dist.send( tensor=coeff_loc, dst=i+1, tag=4000*size+i+1 )

        # 1) send current evironment to workers
        log.info(f"MANAGER {rank} sending C to rank {i+1} tag {1000*size+i+1}")
        # dist.send( tensor=C_loc_cpu, dst=i+1, tag=1000*size+i+1 )
        pipes_to_workers[i].send((C_loc_cpu_raw1d, C_loc_cpu_meta))
        log.info(f"MANAGER {rank} sending T to rank {i+1} tag {2000*size+i+1}")
        # dist.send( tensor=T_loc_cpu, dst=i+1, tag=2000*size+i+1 )
        pipes_to_workers[i].send((T_loc_cpu_raw1d, T_loc_cpu_meta))

    # 3) receive gradient components back from workers
    log.info(f"MANAGER {rank} entering receiving stage")
    tmp_result= torch.zeros(3, dtype=next(iter(state.coeffs.values())).dtype, device='cpu')
    for i in range(len(tasks)):
        # tensor (Tensor) – Tensor to fill with received data.
        # src (int, optional) – Source rank. Will receive from any process if unspecified.
        # group (ProcessGroup, optional) – The process group to work on
        # tag (int, optional) – Tag to match recv with remote send
        sender= dist.recv( tmp_result )
        if sender<0:
            raise RuntimeError(f"Error in recv, unexpected sender {sender}")
        log.info(f"MANAGER {rank} received result from {sender}")
        recv_coeff_id= int(tmp_result[0])
        recv_grad_id= int(tmp_result[1])
        recv_grad_res= tmp_result[2]
        
        full_grad[list(full_grad.keys())[recv_coeff_id]][recv_grad_id]= recv_grad_res
        # 3.1) send another gradient component back to sender (if there still is 
        #      a component to compute)
        if numsent < len(tasks):
            grad_id[0]= tasks[numsent][0]
            grad_id[1]= tasks[numsent][1]
            log.info(f"MANAGER {rank} sending id {grad_id} to WORKER {sender}")
            dist.send( tensor=grad_id, dst=sender )
            numsent+=1
        else: 
            # no more work
            dist.send( _terminate_grad, sender )

def worker_code(rank,size,gpu_id,pipe):
    print(f"WORKER - {rank} in group of {size} - CPU"\
        +f" {len(os.sched_getaffinity(0))}/{mp.cpu_count()}"\
        +f" torch.get_num_threads {torch.get_num_threads()}")
    cfg.configure(args)
    loc_log= open(f"{cfg.main_args.out_prefix}.w{rank}.log","w",1)

    if not args.symmetry or args.symmetry=="NONE":
        settings= settings_full
    elif args.symmetry=="U1":
        settings= settings_U1

    # set GPU device to be used. Assuming valid gpu_id, move state and env on the device
    cuda_dev= None
    if gpu_id>=0:
        cuda_dev= torch.device(f"cuda:{gpu_id}")
        cfg.global_args.device= cuda_dev
        settings.device = cfg.global_args.device
        settings_full.device = cfg.global_args.device
        print(f"WORKER {rank} device {cfg.global_args.device}")

    settings.back.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg
    settings.back.set_num_threads(args.omp_cores)
    settings.back.random_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    def loss_fn(state, ctm_env_in, opt_context):
        return loss_functional(energy_f, state, ctm_env_in, opt_context)

    # get a local copy of initial state
    state_json_str= pipe.recv()
    state= deserialize_from_json(state_json_str, settings)
    #log.info(
    loc_log.write(f"WORKER {rank} initial state {state.coeffs}\n")

    # initialize environment
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)

    # main worker loop
    optimization_done=False
    loss0= torch.zeros(1, dtype=next(iter(state.coeffs.values())).dtype, device='cpu')
    grad_id= torch.zeros(2, dtype=torch.long, device='cpu')
    while not optimization_done:
        #log.info(
        loc_log.write(f"WORKER {rank} open for new dist_grad - waiting for ENV\n")
        
        # 1) receive initial gradient component - new dist grad region starting
        dist.recv( tensor=grad_id, src=0, tag=rank )
        if (grad_id[0]==-2 and grad_id[1]==-2): 
            loc_log.write(f"WORKER {rank} received final terminating signal\n")
            break
        #log.info(
        loc_log.write(f"WORKER {rank} received grad_id {grad_id}\n")

        #log.info(
        loc_log.write(f"WORKER {rank} waiting for loss0 tag {3000*size + rank}\n")
        dist.recv( tensor=loss0, src=0, tag=3000*size + rank )
        #log.info(
        loc_log.write(f"WORKER {rank} received loss0 {loss0.item()} tag {3000*size + rank}\n")

        # receive current state
        coeff_loc= torch.zeros_like(state.coeffs[(0,0)], device='cpu')
        dist.recv( tensor=coeff_loc, src=0, tag=4000*size + rank )

        # receive current (converged) env
        # C_loc_cpu= torch.zeros_like(ctm_env.get_C(),device='cpu')
        # T_loc_cpu= torch.zeros_like(ctm_env.get_T(),device='cpu')
        #log.info(
        loc_log.write(f"WORKER {rank} wating for C,T tags {1000*size + rank},{2000*size + rank}\n")
        # dist.recv( tensor=C_loc_cpu, src=0, tag=1000*size + rank )
        # dist.recv( tensor=T_loc_cpu, src=0, tag=2000*size + rank )
        C_r1d, C_meta= pipe.recv()
        T_r1d, T_meta= pipe.recv()
        C_loc_cpu= yastn.decompress_from_1d(C_r1d,settings=settings,d=C_meta)
        T_loc_cpu= yastn.decompress_from_1d(T_r1d,settings=settings,d=T_meta)
        if gpu_id>=0:
            state.coeffs[(0,0)]= coeff_loc.to(cuda_dev)
            ctm_env.C[ctm_env.keyC]= C_loc_cpu.to(cuda_dev)
            ctm_env.T[ctm_env.keyT]= T_loc_cpu.to(cuda_dev)
        #log.info(
        loc_log.write(f"WORKER {rank} received C, T and moved to {cuda_dev}\n")
        
        # 2) while there is a valid gradient component to compute
        #    execute energy computation
        tmp_result= torch.zeros(3, dtype=next(iter(state.coeffs.values())).dtype, device='cpu')
        while grad_id[0] >= 0 and grad_id[1] >= 0:
            coeff_key_id= grad_id[0]
            loc_grad_id= grad_id[1]

            # 3) gradient evaluation forcing cuda_dev as default GPU
            with torch.cuda.device(cuda_dev):
                # clone the env
                env_clone= ctm_env.clone()
                grad_val= grad_fd_component(loss_fn, state, env_clone, \
                    cfg.opt_args, cfg.ctm_args, loss0, coeff_key_id, loc_grad_id, loc_log)

            tmp_result[0]= coeff_key_id
            tmp_result[1]= loc_grad_id
            tmp_result[2]= grad_val
            #log.info(
            loc_log.write(f"WORKER {rank} sending result for grad_id {grad_id}\n")
            dist.send( tensor=tmp_result, dst=0 )
            dist.recv( tensor=grad_id, src=0 )
            #log.info(
            loc_log.write(f"WORKER {rank} received grad_id {grad_id}\n")
        #log.info(
        loc_log.write(f"WORKER {rank} received terminating signal for grad_fn\n")
        # 3) current dist grad region done - waiting for next region
        # optimization_done= (grad_id[0]==-2 and grad_id[1]==-2)
        # #log.info(
        # loc_log.write(f"WORKER {rank} received final terminating signal\n")

def main(rank, size, pipes_to_workers):
    print(f"MASTER - rank {rank} in group of {size}")

    # 0) configure
    cfg.configure(args)
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings_full= yastn.make_config(backend=backend, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    settings= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)
    settings.back.ad_decomp_reg= cfg.ctm_args.ad_decomp_reg

    # 1) set model
    model= j1j2.J1J2_C4V_BIPARTITE_NOSYM(settings_full, j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # 2) initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v_lc(args.instate, settings)
        state= state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/state.sites[(0,0)].norm(p='inf')()
    elif args.opt_resume is not None:
        state= IPEPS_ABELIAN_C4V_LC(settings, None, None, None)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    # 3) distribute current state to all workers (if there are any) 
    if len(pipes_to_workers)>0 and size>1:
        state_json_str= state.serialize_to_json()
        for p in pipes_to_workers:
            p.send(state_json_str)

    # 4) initialize environment
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)
    print(ctm_env)

    # 5) compute initial observables
    loss0 = energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values])) 

    def loss_fn(state, ctm_env_in, opt_context):
        return loss_functional(energy_f, state, ctm_env_in, opt_context)

    def grad_fn(state, ctm_env, opt_context, loss0):
        # 1) prepare list of tasks
        tasks=[]
        for i,k in enumerate(list(state.coeffs.keys())): 
            for j in range(state.coeffs[k].numel()):
                tasks.append((i,j))

        # dictionary holding gradients of coeffs
        full_grad= dict()
        for k in state.coeffs.keys():
            full_grad[k]= torch.zeros(state.coeffs[k].numel(), dtype=state.coeffs[k].dtype, \
                device='cpu')

        # invoke the manager to start sending out component computation jobs
        status= manager_code(rank,size,state,ctm_env,pipes_to_workers,tasks,full_grad,loss0)
        
        # move full_grad to default device
        for k in state.coeffs.keys(): full_grad[k]= full_grad[k].to(state.device)
        log_grad_str= [ [full_grad[k][i].item() for i in range(full_grad[k].numel())] for k in full_grad.keys()]
        print(f"FULL_GRAD {log_grad_str}")

        return full_grad

    # def _to_json(l):
    #     re=[l[i,0].item() for i in range(l.size()[0])]
    #     im=[l[i,1].item() for i in range(l.size()[0])]
    #     return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

        # if (not opt_context["line_search"]) and args.top_freq>0 and epoch%args.top_freq==0:
        #     coord_dir_pairs=[((0,0), (1,0))]
        #     for c,d in coord_dir_pairs:
        #         # transfer operator spectrum
        #         print(f"TOP spectrum(T)[{c},{d}] ",end="")
        #         l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
        #         print("TOP "+json.dumps(_to_json(l)))

    # 5) optimize - enter main loop
    optimize_state(state, ctm_env, loss_fn, grad_fn, obs_fn=obs_fn)

    # 6) inform workers that computation has been finished
    status= manager_code(rank,size,None,None,pipes_to_workers,[],None,None)

    # 7) compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v_lc(outputstatefiles, settings)
    ctm_env= ENV_C4V_ABELIAN(args.chi, state=state, init=True)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env,force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

def init_process(rank, size, fn, *args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, *args)

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")

    # check if distributed available
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    world_size=args.workers+1

    torch.__config__.parallel_info()

    # check if and how many GPU's are available and assign them equally among workers
    gpu_ids= [-1]*world_size
    if torch.cuda.is_available():
        num_gpus= torch.cuda.device_count()
        gpu_ids= [-1]+[(i-1)%num_gpus for i in range(1,world_size)]
        print(f"GPUs available {num_gpus} GPU assignment {gpu_ids}")

    mp.set_start_method('spawn')
    processes=[]
    pipe_master_to_worker=[None]*max(0,world_size-1)
    pipe_worker_to_master=[None]*max(0,world_size-1)
    for rank in range(1,world_size):
        m, w= mp.Pipe()
        pipe_master_to_worker[rank-1]= m
        pipe_worker_to_master[rank-1]= w
    for rank in range(world_size):
        if rank==0:
            p= mp.Process(target= init_process, args=(rank, world_size, main, \
                pipe_master_to_worker))
        else:
            p= mp.Process(target= init_process, args=(rank, world_size, worker_code, \
                gpu_ids[rank], pipe_worker_to_master[rank-1]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# TODO tests