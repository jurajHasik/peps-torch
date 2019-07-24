import torch
from args import *
from env import *
from ipeps import write_ipeps
import matplotlib.pyplot as plt
from IPython import embed

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


def optimize_state(state, ctm_env_init, loss_fn, model, local_args, opt_args=OPTARGS(), ctm_args=CTMARGS(), global_args = GLOBALARGS()):
    """
    expects a loss function that operates
    loss_fn(state, ctm_env_init, ctm_args = ctm_args, global_args = GLOBALARGS())
    """
    parameters = list(state.sites.values())
    for A in parameters: A.requires_grad_(True)

    outputstatefile = local_args.out_prefix+"_state.json"
    t_data = dict({"loss": [1.0e+16], "min_loss": 1.0e+16})
    current_env = [ctm_env_init]
    def closure():
        for el in parameters: 
            if el.grad is not None: el.grad.zero_()

        # 0) pre-process state: normalize on-site tensors by largest elements
        for coord,site in state.sites.items():
            site = site/torch.max(torch.abs(site))

        loss, ctm_env = loss_fn(state, current_env[0], ctm_args=ctm_args, opt_args=opt_args, global_args=global_args)
        loss.backward()

        # 2) detach current environment from autograd graph
        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        current_env[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()

        # 3) store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > t_data["loss"][-1]:
            t_data["min_loss"]= t_data["loss"][-1]
            write_ipeps(state, outputstatefile, normalize=True)

        return loss

    verbosity = opt_args.verbosity_opt_epoch
    outputstatefile = local_args.out_prefix+"_state.json"
    checkpoint_file = local_args.out_prefix+"_checkpoint.p"
    optimizer = torch.optim.LBFGS(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr)
    epoch0 = 0
    loss0 = 0

    print(f"resume = {opt_args.resume}")

    if opt_args.resume is not None:
        print("resuming from check point")
        checkpoint = torch.load(opt_args.resume)
        init_parameters = checkpoint["parameters"]
        epoch0 = checkpoint["epoch"]
        loss0 = checkpoint["loss"]
        for i in range(len(parameters)):
            parameters[i].data = init_parameters[i].data 
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"checkpoint.loss = {loss0}")

    print(f"new_loss0 = {closure().item()}")

    for epoch in range(local_args.opt_max_iter):
 
        loss = optimizer.step(closure)
        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t")        
        # compute and print observables
        if verbosity>0:
            obs_values, obs_labels = model.eval_obs(state,current_env[0])
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

        # store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["loss"][-2] > t_data["loss"][-1]:
            write_ipeps(state, outputstatefile, normalize=True)
    

    loss, _ctm_env = loss_fn(state, current_env[0], ctm_args=ctm_args, opt_args=opt_args, global_args=global_args)
    torch.save({
            'epoch': epoch0 + local_args.opt_max_iter,
            'loss': loss.item(),
            'parameters': parameters,
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file)
    print(checkpoint_file)


    # plt.plot(t_data["loss"][20:])
    # plt.show()
