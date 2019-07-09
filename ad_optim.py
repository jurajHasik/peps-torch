import torch
from args import *
from IPython import embed
from env import *
from ipeps import write_ipeps

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


def optimize_state(state, ctm_env_init, loss_fn, opt_args=OPTARGS(), ctm_args=CTMARGS(), global_args = GLOBALARGS(), verbosity=0):
    """
    expects a loss function that operates
    loss_fn(state, ctm_env_init, ctm_args = ctm_args, global_args = GLOBALARGS())
    """
    parameters = list(state.sites.values())
    for A in parameters: A.requires_grad_(True)

    previous_ctm = [ctm_env_init]
    def closure():
        for el in parameters: 
            if el.grad is not None: el.grad.zero_()

        # 0) pre-process state: normalize on-site tensors by largest elements
        for coord,site in state.sites.items():
            site = site/torch.max(torch.abs(site))

        loss, ctm_env = loss_fn(state, previous_ctm[0], ctm_args=ctm_args, opt_args=opt_args, global_args=global_args)
        loss.backward()

        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        previous_ctm[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()

        return loss

    outputstatefile = args.out_prefix+"_state.json"
    t_data = dict({"loss": [1.0e+16]})
    optimizer = torch.optim.LBFGS(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr)    
    for epoch in range(args.opt_max_iter):
        loss = optimizer.step(closure)
        if verbosity>0: print(f"epoch= {epoch}, loss= {loss}")

        t_data["loss"].append(loss.item())

        # store current state if the loss improves
        if t_data["loss"][-2] > t_data["loss"][-1]:
            write_ipeps(state, outputstatefile)