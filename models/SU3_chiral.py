import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from math import sqrt
from numpy import exp
import itertools

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

# define matrices that permutate the SU(3) states for a bond (P_12, P_23, P_31) and for a triangle (P_t and its inverse P_t2 = P_t^(-1)
matP_12 = torch.zeros((27,27), dtype=torch.complex128)
matP_23 = torch.zeros((27,27), dtype=torch.complex128)
matP_31 = torch.zeros((27,27), dtype=torch.complex128)
matP_t = torch.zeros((27,27), dtype=torch.complex128)
matP_t2 = torch.zeros((27,27), dtype=torch.complex128)
for s in range(27):
	n1,n2,n3 = fmap_inv(s)
	matP_12[s,fmap(n2,n1,n3)] = 1.
	matP_23[s,fmap(n1,n3,n2)] = 1.
	matP_31[s,fmap(n3,n2,n1)] = 1.
	matP_t[s, fmap(n2,n3,n1)] = 1.
	matP_t2[s, fmap(n3,n1,n2)] = 1.

# define the matrices associated with the observables \lambda_3 and \lambda_8 for the 3 sites
lambda_3 = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]], dtype = torch.complex128)
lambda_8 = 1./sqrt(3.) * torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,-2.]], dtype = torch.complex128)
lambda_3_1 = torch.eye(27, dtype = torch.complex128)
lambda_3_2 = torch.eye(27, dtype = torch.complex128)
lambda_3_3 = torch.eye(27, dtype = torch.complex128)
lambda_8_1 = torch.eye(27, dtype = torch.complex128)
lambda_8_2 = torch.eye(27, dtype = torch.complex128)
lambda_8_3 = torch.eye(27, dtype = torch.complex128)
for s in range(27):
	n1,n2,n3 = fmap_inv(s)
	lambda_3_1[s,s] = lambda_3[n1,n1]
	lambda_3_2[s,s] = lambda_3[n2,n2]
	lambda_3_3[s,s] = lambda_3[n3,n3]
	lambda_8_1[s,s] = lambda_8[n1,n1]
	lambda_8_2[s,s] = lambda_8[n2,n2]
	lambda_8_3[s,s] = lambda_8[n3,n3]


class SU3_CHIRAL():

	def __init__(self, theta=0., global_args=cfg.global_args):
		r"""
		:param theta: rotation angle of the hamiltonian which parametrizes the ratio between real and imaginary permutations
        :param global_args: global configuration
        :type theta: float
        :type global_args: GLOBALARGS
        """
		self.theta=theta
		self.dtype=global_args.dtype
		self.device=global_args.device
		self.phys_dim=27
		# in-cell bond permutation operators:
		self.P12, self.P23, self.P31 = matP_12, matP_23, matP_31
		# in-cell triangle permutation operators:
		self.P123, self.P123m = matP_t, matP_t2
		self.h_triangle = exp(1j*self.theta) * self.P123 + exp(-1j*self.theta) * self.P123m
		
	
	def energy_triangle(self,state,env):
		# Computes the expectation value of the energy per site for a "down"-triangle (defined by the three sites within the unit cell)
		id_matrix = torch.eye(27, dtype=torch.complex128)
		e_dn = rdm.rdm1x1((0,0), state, env, operator=self.h_triangle)
		norm_wf = rdm.rdm1x1((0,0), state, env, operator=id_matrix)
		e_dn = torch.real(e_dn / norm_wf)
		return(2/3*e_dn)
		
