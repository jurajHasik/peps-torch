import torch
import su2
from env import ENV
import ipeps
import rdm

class AKLTS2():
	def __init__(self, dtype=torch.float64, device='cpu'):
		self.dtype=dtype
		self.device=device
		self.h = self.get_h()
		self.phys_dim = 5

	# build AKLT S=2 Hamiltonian <=> Projector from product of two S=2 DOFs
	# to S=4 DOF H = \sum_{<i,j>} h_ij, where h_ij= ...
	#
	# indices of h correspond to s_i,s_j;s_i',s_j'
	def get_h(self):
		s5 = su2.SU2(5, dtype=self.dtype, device=self.device)
		expr_kron = 'ij,ab->iajb'
		SS = torch.einsum(expr_kron,s5.SZ(),s5.SZ()) + 0.5*(torch.einsum(expr_kron,s5.SP(),s5.SM()) \
			+ torch.einsum(expr_kron,s5.SM(),s5.SP()))
		SS = SS.view(5*5,5*5)
		h = (1./14)*(SS + (7./10.)*SS@SS + (7./45.)*SS@SS@SS + (1./90.)*SS@SS@SS@SS)
		h = h.view(5,5,5,5)
		return h

	# evaluation of energy depends on the nature of underlying
	# ipeps state
	#
	# Ex.1 for 1-site c4v invariant iPEPS there is just a single two-site
	# term which gives the energy-per-site
	#
	# Ex.2 for 1-site invariant iPEPS there are two two-site terms
	# which give the energy-per-site
	#    0       0
	# 1--A--3 1--A--3 
	#    2       2                          A
	#    0       0                          2
	# 1--A--3 1--A--3                       0
	#    2       2    , terms A--3 1--A and A have to be evaluated
	#
	# Ex.3 for 2x2 cluster iPEPS there are eight two-site terms
	#    0       0       0
	# 1--A--3 1--B--3 1--A--3
	#    2       2       2
	#    0       0       0
	# 1--C--3 1--D--3 1--C--3
	#    2       2       2             A--3 1--B      A B C D
	#    0       0                     B--3 1--A      2 2 2 2
	# 1--A--3 1--B--3                  C--3 1--D      0 0 0 0
	#    2       2             , terms D--3 1--C and  C D A B  
	def energy_1x1c4v(self,state,env):
		rdm2x1 = rdm.rdm2x1((0,0), state, env)
		# apply a rotation on physical index of every "odd" site
		# A A => A B
		# A A => B A
		rot_op = su2.get_rot_op(5)
		h_rotated = torch.einsum('jl,ilak,kb->ijab',rot_op,self.h,rot_op)
		energy = torch.einsum('ijab,ijab',rdm2x1,h_rotated)
		return energy		

	def energy_1x1(self,state,env):
		rdm2x1 = rdm.rdm2x1((0,0), state, env)
		rdm1x2 = rdm.rdm1x2((0,0), state, env)
		# apply a rotation on physical index of every "odd" site
		# A A => A B
		# A A => B A
		rot_op = su2.get_rot_op(5)
		h_rotated = torch.einsum('jl,ilak,kb->ijab',rot_op,self.h,rot_op)
		energy = torch.einsum('ijab,ijab',rdm2x1,h_rotated) + torch.einsum('ijab,ijab',rdm1x2,h_rotated)
		return energy		

	def energy_2x2(self,ipeps):
		pass

	# assuming reduced density matrix of 2x2 cluster with indexing of DOFs
	# as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
	def energy_2x2(self,rdm2x2):
		energy = torch.einsum('ijklabkl,ijab',rdm2x2,self.h)
		return energy

	def energy_2x1_1x2(self,rdm2x1,rdm1x2):
		energy = torch.einsum('ijab,ijab',rdm2x1,self.h)\
			+ torch.einsum('ijab,ijab',rdm1x2,self.h)
		return energy
		

