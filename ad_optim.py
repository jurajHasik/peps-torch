import torch
from args import *
from IPython import embed
from env import *

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

    peremantent_ctm = [ctm_env_init]
    def closure():
        for el in parameters: 
            if el.grad is not None: el.grad.zero_()
        loss, ctm_env = loss_fn(state, peremantent_ctm[0], ctm_args=ctm_args, global_args=global_args)
        # loss, ctm_env = loss_fn(state, ctm_env_init, ctm_args=ctm_args, global_args=global_args)
        loss.backward()
        lst_C = list(ctm_env.C.values())
        lst_T = list(ctm_env.T.values())
        peremantent_ctm[0] = ctm_env
        for el in lst_T + lst_C: el.detach_()
        #if verbosity>0: print(f"closure loss = {loss}")
        return loss

    optimizer = torch.optim.LBFGS(parameters, lr=opt_args.lr)    
    for epoch in range(opt_args.max_iter):
        optimizer.step(closure)
        loss = closure()
        if verbosity>0: print(f"loss = {loss}")
