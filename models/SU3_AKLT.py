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
matP_12 = torch.zeros(27,27)
matP_23 = torch.zeros(27,27)
matP_31 = torch.zeros(27,27)
matP_t = torch.zeros(27,27)
matP_t2 = torch.zeros(27,27)
for s in range(27):
	n1,n2,n3 = fmap_inv(s)
	matP_12[s,fmap(n2,n1,n3)] = 1.
	matP_23[s,fmap(n1,n3,n2)] = 1.
	matP_31[s,fmap(n3,n2,n1)] = 1.
	matP_t[s, fmap(n2,n3,n1)] = 1.
	matP_t2[s, fmap(n3,n1,n2)] = 1.


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
		# rescale with 2/3 because we are interested in the energy per site
		return(2/3*energy)
	
	def energy_triangle_up(self,state,env):
		# Computes the expectation value of the energy for a "up"-triangle
		# define 2*2 reduced density matrix with the following
		# unit cells: (0,0)--(1,0)      with the index order convention (0,0), (1,0), (0,1), (1,1)
		#				\   /   \
		#			    (0,1)--(1,1)
		# The down-triangle is made of the following 3 sites: (1,0)_2, (0,1)_3, (1,1)_1
		# where (i,j)_k labels the k-th site of the unit cell (i,j).
		rho2x2 = rdm.rdm2x2((0,0), state, env)
		# trace over the unit cell (0,0) which does not belong to the up triangle
		rho3 = torch.einsum('abcdajkl->bcdjkl',rho2x2)
		
		# define a permutation operator between (1,0)_2, (0,1)_3, (1,1)_1
		P_upm = torch.zeros(27,27,27,27,27,27)
		P_up = torch.zeros(27,27,27,27,27,27)
		#				   2  3  1  2' 3' 1'
		for s1 in range(27):
			for s2 in range(27):
				for s3 in range(27):
					n11,n12,n13 = fmap_inv(s1)
					n21,n22,n23 = fmap_inv(s2)
					n31,n32,n33 = fmap_inv(s3)
					# direct permutation
					dn11,dn12,dn13, dn21,dn22,dn23, dn31,dn32,dn33 = n22,n12,n13, n21,n33,n23, n31,n32,n11
					ds1, ds2, ds3 = fmap(dn11,dn12,dn13), fmap(dn21,dn22,dn23), fmap(dn31,dn32,dn33)
					P_up[ds2,ds3,ds1,s2,s3,s1] = 1.
					# inverse permutation
					in11,in12,in13, in21,in22,in23, in31,in32,in33 = n33,n12,n13, n21,n11,n23, n31,n32,n22
					is1, is2, is3 = fmap(in11,in12,in13), fmap(in21,in22,in23), fmap(in31,in32,in33)
					P_upm[is2,is3,is1,s2,s3,s1] = 1.
		
		# contract rho3 and the operator
		P_op = P_up + P_upm
		energy = torch.einsum('ijkabc,abcijk->',P_op,rho3)
		return(2/3*energy)
		
		
	def eval_color3(self,state,env):
		# computes the expectation value of the SU(3) observable \lambda_3 (i.e. the color 3) for the three sites of the unit cell
		# todo: the same for the observable \lambda_8
		lambda_3 = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]]).double()
		lambda_3_1 = torch.eye(27,27).double()
		lambda_3_2 = torch.eye(27,27).double()
		lambda_3_3 = torch.eye(27,27).double()
		for s in range(27):
			n1,n2,n3 = fmap_inv(s)
			lambda_3_1[s,s] = lambda_3[n1,n1]
			lambda_3_2[s,s] = lambda_3[n2,n2]
			lambda_3_3[s,s] = lambda_3[n3,n3]
		rho1x1 = rdm.rdm1x1((0,0),state,env)
		color3_1 = torch.trace(rho1x1@lambda_3_1)
		color3_2 = torch.trace(rho1x1@lambda_3_2)
		color3_3 = torch.trace(rho1x1@lambda_3_3)
		return(color3_1, color3_2, color3_3)
		
