
import torch
from args import *
import env
from env import ENV
import ipeps
import ctmrg
import rdm
from env import *
from models import akltS2, coupledLadders
from ad_optim import optimize_state
from IPython import embed

if __name__=='__main__':

    torch.set_printoptions(precision=7)

    model = akltS2.AKLTS2()
    gs_state = ipeps.read_ipeps("test-input/AKLT-S2_1x1.in")

    gs_ctm_env = ENV(args.chi, gs_state)
    init_env(gs_state, gs_ctm_env)
    gs_energy = model.energy_1x1c4v(gs_state, gs_ctm_env)
    print("(exact GS) E(2x1+1x2): "+str(gs_energy))

    # initialize an ipeps ( one site shift invariant)
    bond_dim = 2

    A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
    A = 2 * (A - 0.5)
    def zero_fn(coord): return (0,0)
    sites = {(0,0): A}

    state = ipeps.IPEPS(sites, zero_fn)
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    def loss_fn(state, ctm_env_init, ctm_args = CTMARGS(), global_args = GLOBALARGS()):
        ctm_env = ctmrg.run(state, ctm_env_init, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_1x1(state, ctm_env)
        return loss, ctm_env

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, verbosity=1)

    ctm_env = ENV(args.chi, state)
    init_env(ipeps, ctm_env)
    ctm_env = ctmrg.run(state, ctm_env)
    opt_energy = model.energy_1x1(state,ctm_env)

    # energy = model.energy_1x1c4v(state,ctm_env)
    # print("E(1x1c4v): "+str(energy))

    print(f"(exact GS.) E(2x1+1x2): {gs_energy}")
    print(f"(opt state) E(2x1+1x2): {opt_energy}")
    