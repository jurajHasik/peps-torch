
import torch
from args import *
import env
from env import ENV
import ipeps
import ctmrg
import rdm
from env import *
from models import akltS2, coupledLadders, hb
from ad_optim import optimize_state
from IPython import embed

if __name__=='__main__':

    torch.set_printoptions(precision=7)

    model = hb.HB()
    # model = akltS2.AKLTS2()
    # gs_state = ipeps.read_ipeps("test-input/AKLT-S2_1x1.in")

    # gs_ctm_env = ENV(args.chi, gs_state)
    # init_env(gs_state, gs_ctm_env)
    # gs_energy = model.energy_1x1c4v(gs_state, gs_ctm_env)
    # print("(exact GS) E(2x1+1x2): "+str(gs_energy))

    # initialize an ipeps
    bond_dim = args.bond_dim

    A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
    B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
   
    # define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    def zero_fn(coord): 
        vx = (coord[0] + abs(coord[0]) * 2) % 2
        vy = abs(coord[1])
        return ((vx + vy) % 2, 0)
    
    sites = {(0,0): A, (1,0): B}

    state = ipeps.IPEPS(sites, zero_fn)
    
    # test tiling of square lattice
    lx, ly = 6, 6
    for y in range(-ly//2,ly//2+1):
        if y == -ly//2:
            for x in range(-lx//2,lx//2):
                print(str(x)+" ", end="")
            print("")
        print(str(y)+" ", end="")
        for x in range(-lx//2,lx//2):
            print(str(state.vertexToSite((x,y)))+" ", end="")
        print("")

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    def loss_fn(state, ctm_env_init, ctm_args = CTMARGS(), global_args = GLOBALARGS()):
        ctm_env = ctmrg.run(state, ctm_env_init, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_2x1_1x2(state, ctm_env)
        return loss, ctm_env

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, verbosity=1)

    ctm_env = ENV(args.chi, state)
    init_env(ipeps, ctm_env)
    ctm_env = ctmrg.run(state, ctm_env)
    opt_energy = model.energy_2x1_1x2(state,ctm_env)
    
    print(f"(opt state) E(2x1+1x2): {opt_energy}")
    