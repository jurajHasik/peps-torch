""" Rank-3 tensor from Mathieu's Mathematica code.
    First index is the one to contract """

import torch

# S=0

T_S0 = torch.zeros((4,2,2), dtype=torch.float64)
T_S0[3,0,1]=-2**(-1./2.)
T_S0[3,1,0]=2**(-1./2.)

# S=1

T_S1 = torch.zeros((4,2,2), dtype=torch.float64)
T_S1[0,0,0]=1
T_S1[1,0,1]=2**(-1./2.)
T_S1[1,1,0]=2**(-1./2.)
T_S1[2,1,1]=1