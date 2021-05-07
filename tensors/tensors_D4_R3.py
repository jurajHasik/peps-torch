""" Rank-3 tensor from Mathieu's Mathematica code.
    First index is the one to contract """

import torch

# S=0

T_2_B_1 = torch.zeros((4,2,2), dtype=torch.float64)
T_2_B_1[3,0,1]=-2**(-1./2.)
T_2_B_1[3,1,0]=2**(-1./2.)

# S=1

T_2_A_1 = torch.zeros((4,2,2), dtype=torch.float64)
T_2_A_1[0,0,0]=1
T_2_A_1[1,0,1]=2**(-1./2.)
T_2_A_1[1,1,0]=2**(-1./2.)
T_2_A_1[2,1,1]=1