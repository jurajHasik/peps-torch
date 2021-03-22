import torch
import numpy as np
from ipeps.ipeps import IPEPS, write_ipeps, read_ipeps

# SU(3) AKLT-like state with D=3
# Compute the a tensor (kagome-iPESS -> square-iPEPS) and write the correspondig iPEPS object it into a .json file

def fmap(n1,n2,n3):
	return n3+3*n2+9*n1
	
def fmap_inv(s):
	n1 = s//9
	n2 = (s-9*n1)//3
	n3 = s-9*n1-3*n2
	return(n1,n2,n3)
	
	
#______ Trivalent tensors _______
S0 = S1 = S2 = S3 = S4 = torch.zeros(7,7,7)

# S0: {0,3,0}, A2 
S0[3,4,5] = S0[4,5,3] = S0[5,3,4] = -1./np.sqrt(6.)
S0[3,5,4] = S0[4,3,5] = S0[5,4,3] = 1./np.sqrt(6.)

# S1: {3,0,0}, A2
S1[0,1,2] = S1[1,2,0] = S1[2,0,1] = -1./np.sqrt(6.)
S1[0,2,1] = S1[1,0,2] = S1[2,1,0] = 1./np.sqrt(6.)

# S2: {1,1,1}, A2
S2[0,5,6] = S2[5,6,0] = S2[6,0,5] = S2[1,6,4] = S2[4,1,6] = S2[6,4,1] = S2[2,3,6] = S2[3,6,2] = S2[6,2,3] = -1./(3.*np.sqrt(2.))
S2[0,6,5] = S2[5,0,6] = S2[6,5,0] = S2[1,4,6] = S2[4,6,1] = S2[6,1,4] = S2[2,6,3] = S2[3,2,6] = S2[6,3,2] = 1./(3.*np.sqrt(2.))

# S3: {0,0,3}, A1
S3[6,6,6] = 1.

# S4: {1,1,1}, A1
S4[0,5,6] = S4[0,6,5] = S4[5,0,6] = S4[5,6,0] = S4[6,0,5] = S4[6,5,0] = 1./(3.*np.sqrt(2.))
S4[1,4,6] = S4[1,6,4] = S4[6,1,4] = S4[6,4,1] = S4[4,6,1] = S4[4,1,6] = -1./(3.*np.sqrt(2.))
S4[2,3,6] = S4[3,6,2] = S4[6,2,3] = S4[2,6,3] = S4[3,2,6] = S4[6,3,2] = 1./(3.*np.sqrt(2.))
#________________________________


#______ Bivalent tensors ________
L0 = L1 = L2 = torch.zeros(3,7,7)

# L0: {0,2,0}, B
L0[0,3,4] = L0[1,3,5] = L0[2,4,5] = -1./np.sqrt(2.)
L0[0,4,3] = L0[1,5,3] = L0[2,5,4] = 1./np.sqrt(2.)

# L1: {1,0,1}, B
L1[0,0,6] = L1[1,1,6] = L1[2,2,6] = 1./np.sqrt(2.)
L1[0,6,0] = L1[1,6,1] = L1[2,6,2] = -1./np.sqrt(2.)

# L2: {1,0,1}, A
L2[0,0,6] = L2[1,1,6] = L2[2,2,6] = 1./np.sqrt(2.)
L2[0,6,0] = L2[1,6,1] = L2[2,6,2] = 1./np.sqrt(2.)
#________________________________


# save tensors to external files
torch.save(S0, 'SU3_CSL_ipeps/SU3_tensors/S0.pt')
torch.save(S1, 'SU3_CSL_ipeps/SU3_tensors/S1.pt')
torch.save(S2, 'SU3_CSL_ipeps/SU3_tensors/S2.pt')
torch.save(S3, 'SU3_CSL_ipeps/SU3_tensors/S3.pt')
torch.save(S4, 'SU3_CSL_ipeps/SU3_tensors/S4.pt')
torch.save(L0, 'SU3_CSL_ipeps/SU3_tensors/L0.pt')
torch.save(L1, 'SU3_CSL_ipeps/SU3_tensors/L1.pt')
torch.save(L2, 'SU3_CSL_ipeps/SU3_tensors/L2.pt')


# S(lambda1, lambda2, lambda3, lambda4) = S0 + lambda1 * S1 + lambda2 * S2 + 1j * lambda3 * S3 + 1j * lambda4 * S4
# L(mu1, mu2) = L0 + mu1 * L1 + 1j * mu2 * L2
# a(lambdas, mus) = SLSLL
