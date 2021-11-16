import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from math import sqrt
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
matP_12 = torch.zeros(27,27).double()
matP_23 = torch.zeros(27,27).double()
matP_31 = torch.zeros(27,27).double()
matP_t = torch.zeros(27,27).double()
matP_t2 = torch.zeros(27,27).double()
for s in range(27):
	n1,n2,n3 = fmap_inv(s)
	matP_12[s,fmap(n2,n1,n3)] = 1.
	matP_23[s,fmap(n1,n3,n2)] = 1.
	matP_31[s,fmap(n3,n2,n1)] = 1.
	matP_t[s, fmap(n2,n3,n1)] = 1.
	matP_t2[s, fmap(n3,n1,n2)] = 1.

# define the matrices associated with the observables \lambda_3 and \lambda_8 for the 3 sites
lambda_3 = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]]).double()
lambda_8 = 1./sqrt(3.) * torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,-2.]]).double()
lambda_3_1 = torch.eye(27).double()
lambda_3_2 = torch.eye(27).double()
lambda_3_3 = torch.eye(27).double()
lambda_8_1 = torch.eye(27).double()
lambda_8_2 = torch.eye(27).double()
lambda_8_3 = torch.eye(27).double()
for s in range(27):
	n1,n2,n3 = fmap_inv(s)
	lambda_3_1[s,s] = lambda_3[n1,n1]
	lambda_3_2[s,s] = lambda_3[n2,n2]
	lambda_3_3[s,s] = lambda_3[n3,n3]
	lambda_8_1[s,s] = lambda_8[n1,n1]
	lambda_8_2[s,s] = lambda_8[n2,n2]
	lambda_8_3[s,s] = lambda_8[n3,n3]


class SU3_AKLT():

	def __init__(self, global_args=cfg.global_args):
		r"""
        :param global_args: global configuration
        :type global_args: GLOBALARGS
        """
		self.dtype=global_args.dtype
		self.device=global_args.device
		self.phys_dim=27
		# in-cell bond permutation operators:
		self.P12, self.P23, self.P31 = matP_12.double(), matP_23.double(), matP_31.double()
		# in-cell triangle permutation operators:
		self.P123, self.P123m = matP_t.double(), matP_t2.double()


	def energy_triangle(self,state,env):
		# Computes the expectation value of the energy per site for a "down"-triangle (defined by the three sites within the unit cell)
		h_triangle = self.P123 + self.P123m
		rho1x1 = rdm.rdm1x1((0,0),state,env)
		energy = torch.trace(rho1x1@h_triangle)
		# Other way to compute the energy without explicitly computing the reduced density matrix: insert directly h_triangle
		# print(2/3*rdm.rdm1x1((0,0),state,env,operator=h_triangle)/rdm.rdm1x1((0,0),state,env,operator=torch.eye(27,dtype=torch.float64)))
		return(2/3*energy)
	
	def energy_triangle_up(self,state,env):
		# Computes the expectation value of the energy for a "up"-triangle.
		# First define the up-triangle reduced density matrix with the following
		# unit cells:      (1,0)      with the index order convention (1,0), (0,1), (1,1)
		#				   /   \
		#			    (0,1)--(1,1)
		# The up-triangle is made of the following 3 sites: (1,0)_2, (0,1)_3, (1,1)_1
		# where (i,j)_k labels the k-th site of the unit cell (i,j).
		# rho3 = rdm.rdm2x2_up_triangle((0,0), state, env)
		
		# define a permutation operator between (1,0)_2, (0,1)_3, (1,1)_1
		P_upm = torch.zeros(3,3,3,3,3,3).double()
		P_up = torch.zeros(3,3,3,3,3,3).double()
		#				   2 3 1 2'3'1' (prime indices are 'ket' indices)
		for n1 in range(3):
			for n2 in range(3):
				for n3 in range(3):
					P_up[n3,n1,n2,n2,n3,n1] = 1.
					P_upm[n1,n2,n3,n2,n3,n1] = 1.		
		# contract rho3 and the operator
		P_op = P_up + P_upm
		norm_wf = rdm.rdm2x2_id((0,0), state, env)
		energy = rdm.rdm2x2_up_triangle((0,0), state, env, operator = P_op)
		return(2/3*energy/norm_wf)
		
		
	def eval_lambdas(self,state,env):
		# computes the expectation value of the SU(3) observables \lambda_3 and \lambda_8 for the three sites of the unit cell
		
		rho1x1 = rdm.rdm1x1((0,0),state,env)
		color3_1 = torch.einsum('ii,ii->',rho1x1,lambda_3_1)
		color3_2 = torch.einsum('ii,ii->',rho1x1,lambda_3_2)
		color3_3 = torch.einsum('ii,ii->',rho1x1,lambda_3_3)
		color8_1 = torch.einsum('ii,ii->',rho1x1,lambda_8_1)
		color8_2 = torch.einsum('ii,ii->',rho1x1,lambda_8_2)
		color8_3 = torch.einsum('ii,ii->',rho1x1,lambda_8_3)
		return((color3_1, color3_2, color3_3), (color8_1, color8_2, color8_3))
		
	def eval_corrf_LL(self, direction, state, env, dist=10):
		# computes the correlation functions for observables \lambda_3 (L3) and \lambda_8 (L8)
		corrf_L3L3 = 0
		corrf_L8L8 = 0
		
		O1 = lambda_3_1
		get_O2 = lambda r: O1
		corrf_L3L3 += corrf.corrf_1sO1sO((0,0), direction, state, env, O1, get_O2, dist)
		
		O1 = lambda_8_1
		get_O2 = lambda r: O1
		corrf_L8L8 += corrf.corrf_1sO1sO((0,0), direction, state, env, O1, get_O2, dist)
		
		return(corrf_L3L3, corrf_L8L8)
		
	def eval_corrf_PP(self, direction, state, env, dist=10):
		# computes the correlation function for P = P_123 + P_123^(-1)
		corrf_PP = 0
		O1 = self.P123 + self.P123m
		get_O2 = lambda r: O1
		corrf_PP += corrf.corrf_1sO1sO((0,0), direction, state, env, O1, get_O2, dist)
		
		e_t = 3./2. * self.energy_triangle(state, env)
		return(corrf_PP - (e_t)**2)
		
