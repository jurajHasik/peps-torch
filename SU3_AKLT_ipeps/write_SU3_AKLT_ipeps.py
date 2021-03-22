import torch
import numpy as np
from ipeps.ipeps import IPEPS, write_ipeps, read_ipeps

# SU(3) AKLT-like state with D=3
# This script computes the a tensor (kagome-iPESS -> square-iPEPS) and writes the correspondig iPEPS object it into a .json file

# function (n1,n2,n3) --> s that maps the basis of states in the fundamental irrep of SU(3) (states n=0,1,2) for the three sites of the unit cell to a single physical index s=0...26
# NB: the 3 sites are labeled as:
#        1---3
#         \ /
#          2
def fmap(n1,n2,n3):
	return n3+3*n2+9*n1
	
# reverse mapping s --> (n1, n2, n3)
def fmap_inv(s):
	n1 = s//9
	n2 = (s-9*n1)//3
	n3 = s-9*n1-3*n2
	return(n1,n2,n3)
	
# star (trivalent) tensor S
S = torch.zeros(3,3,3)
S[0,1,2] = S[1,2,0] = S[2,0,1] = -1.
S[0,2,1] = S[2,1,0] = S[1,0,2] = 1.
S = S * 1./np.sqrt(6.)

# leg (on-site projector) tensor L
L = torch.zeros(3,3,3)
L[0,0,1] = -1.
L[0,1,0] = 1.
L[1,0,2] = -1.
L[1,2,0] = 1.
L[2,1,2] = -1.
L[2,2,1] = 1.
L = L * 1./np.sqrt(2.)

# a tensor: a^uvw_abc ~ 'SLSLL' (uvw are the d=3 physical indices)
a_temp = torch.einsum('abi,uij,jkl,vkc,wld->uvwabcd', S,L,S,L,L)
# reshape the 3 physical indices (d=3) to a single index (d=27)
a = torch.zeros(27,3,3,3,3)
for si in range(27):
	n1,n2,n3 = fmap_inv(si)
	a[si,:,:,:,:] = a_temp[n1,n2,n3,:,:,:,:]


# create IPEPS object from the tensor a
sites = {(0,0):a}
def vertexToSite(coord): 
	return (0,0)
state = IPEPS(sites,vertexToSite)
# write the ipeps
write_ipeps(state, 'SU3_AKLT_ipeps/SU3_AKLT_ipeps.json')
