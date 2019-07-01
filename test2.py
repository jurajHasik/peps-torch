
import torch
from args import args
import env
from env import ENV
import ipeps
import ctmrg
import rdm

if __name__=='__main__':

    state = ipeps.read_ipeps(None, args.instate)

    # test tiling of square lattice with 3x2 unit cell
    #   0 1 2
    # 0 A B C
    # 1 D E F
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
    ctm_env = ENV(args,state)
    ctm_env = ctmrg.run(args,state,ctm_env)

    rdm1x1_00 = rdm.rdm1x1((0,0),state,ctm_env)
    print("RDM_1x1(0,0)")
    print(rdm1x1_00)

    rdm2x2_00 = rdm.rdm2x2((0,0),state,ctm_env)
    print("RDM_2x2(0,0)")
    print(rdm2x2_00)

    state = ipeps.read_ipeps(None, args.instate)

    # test tiling of square lattice with 3x2 unit cell
    #   0 1 2
    # 0 A B C
    # 1 D E F
    # lx, ly = 6, 6
    # for y in range(-ly//2,ly//2+1):
    #     if y == -ly//2:
    #         for x in range(-lx//2,lx//2):
    #             print(str(x)+" ", end="")
    #         print("")
    #     print(str(y)+" ", end="")
    #     for x in range(-lx//2,lx//2):
    #         print(str(state.vertexToSite((x,y)))+" ", end="")
    #     print("")
