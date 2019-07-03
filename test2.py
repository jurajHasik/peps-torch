
import torch
from args import args
import env
from env import ENV
import ipeps
from ipeps import IPEPS
import ctmrg
import rdm
from models import akltS2, coupledLadders

if __name__=='__main__':

    state = ipeps.read_ipeps(args.instate)

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

    torch.set_printoptions(precision=7)
    ctm_env = ENV(args.chi,state)

    # x) Initialize env tensors C,T
    env.init_env(state, ctm_env)
    # if verbosity>0:
    env.print_env(ctm_env, 1)

    ctm_env = ctmrg.run(state,ctm_env)

    # model = akltS2.AKLTS2()
    # energy = model.energy_1x1c4v(state,ctm_env)
    # print("E(1x1c4v): "+str(energy))
    