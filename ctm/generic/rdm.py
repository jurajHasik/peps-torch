import torch
#from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV
from tn_interface import contract, einsum
from tn_interface import contiguous, view, permute
from tn_interface import conj
from math import sqrt

def _sym_pos_def_matrix(rdm, sym_pos_def=False, verbosity=0, who="unknown"):
	rdm_asym= 0.5*(rdm-rdm.conj().t())
	rdm= 0.5*(rdm+rdm.conj().t())
	if verbosity>0: 
		log.info(f"{who} norm(rdm_sym) {rdm.norm()} norm(rdm_asym) {rdm_asym.norm()}")
	if sym_pos_def:
		with torch.no_grad():
			D, U= torch.symeig(rdm, eigenvectors=True)
			if D.min() < 0:
				log.info(f"{who} max(diag(rdm)) {D.max()} min(diag(rdm)) {D.min()}")
				D= torch.clamp(D, min=0)
				rdm_posdef= U@torch.diag(D)@U.conj().t()
				rdm.copy_(rdm_posdef)
	rdm = rdm / rdm.diagonal().sum()
	return rdm

def _sym_pos_def_rdm(rdm, sym_pos_def=False, verbosity=0, who=None):
	assert len(rdm.size())%2==0, "invalid rank of RDM"
	nsites= len(rdm.size())//2

	orig_shape= rdm.size()
	rdm= rdm.reshape(torch.prod(torch.as_tensor(rdm.size())[:nsites]),-1)
	
	rdm= _sym_pos_def_matrix(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
	rdm= rdm.reshape(orig_shape)
	return rdm

def rdm1x1(coord, state, env, operator=None, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) for which reduced density matrix is constructed
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param state: 1-site operator to contract with the two physical indices of the rdm
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type operator: torch.tensor
	:type verbosity: int
	:return: 1-site reduced density matrix with indices :math:`s;s'`. If an operator was provided, 
	returns the expectation value of this operator (not normalized by the norm of the wavefunction).
	:rtype: torch.tensor

	Computes 1-site reduced density matrix :math:`\rho_{1x1}` centered on vertex ``coord`` by 
	contracting the following tensor network::

		C--T-----C
		|  |	 |
		T--A^+A--T
		|  |	 |
		C--T-----C

	If no operator was provided, the physical indices `s` and `s'` of on-site tensor :math:`A` 
	at vertex ``coord`` and it's hermitian conjugate :math:`A^\dagger` are left uncontracted. 
	Else, they are contracted with the operator.
	"""
	who= "rdm1x1"
	# C(-1,-1)--1->0
	# 0
	# 0
	# T(-1,0)--2
	# 1
	rdm = contract(env.C[(coord,(-1,-1))],env.T[(coord,(-1,0))],([0],[0]))
	if verbosity>0:
		print("rdm=CT "+str(rdm.size()))
	# C(-1,-1)--0
	# |
	# T(-1,0)--2->1
	# 1
	# 0
	# C(-1,1)--1->2
	rdm = contract(rdm,env.C[(coord,(-1,1))],([1],[0]))
	if verbosity>0:
		print("rdm=CTC "+str(rdm.size()))
	# C(-1,-1)--0
	# |
	# T(-1,0)--1
	# |			    0->2
	# C(-1,1)--2 1--T(0,1)--2->3
	rdm = contract(rdm,env.T[(coord,(0,1))],([2],[1]))
	if verbosity>0:
		print("rdm=CTCT "+str(rdm.size()))
	# TODO - more efficent contraction with uncontracted-double-layer on-site tensor
	#		Possibly reshape indices 1,2 of rdm, which are to be contracted with 
	#		on-site tensor and contract bra,ket in two steps instead of creating 
	#		double layer tensor
	#	 /
	# --A--
	#  /|s
	#  
	# s'|/
	# --A--
	#  /
	#
	dimsA = state.site(coord).size()
	if operator==None:
		a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(coord),conj(state.site(coord))))
		a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))
	else:
		a = contiguous(einsum('mefgh,mn,nabcd->eafbgchd',state.site(coord), operator, conj(state.site(coord))))
		a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
	# C(-1,-1)--0
	# |
	# |			    0->2
	# T(-1,0)--1 1--a--3
	# |			    2\45(s,s')
	# |			    2
	# C(-1,1)-------T(0,1)--3->1
	rdm = contract(rdm,a,([1,2],[1,2]))
	if verbosity>0:
		print("rdm=CTCTa "+str(rdm.size()))
	# C(-1,-1)--0 0--T(0,-1)--2->0
	# |			 	 1
	# |			 	 2
	# T(-1,0)--------a--3->2
	# |			  	 |\45->34(s,s')
	# |			  	 |
	# C(-1,1)--------T(0,1)--1
	rdm = contract(env.T[(coord,(0,-1))],rdm,([0,1],[0,2]))
	if verbosity>0:
		print("rdm=CTCTaT "+str(rdm.size()))
	# C(-1,-1)--T(0,-1)--0 0--C(1,-1)
	# |		    |			  1->0
	# |		    |
	# T(-1,0)---a--2
	# |		    |\34(s,s')
	# |		    |
	# C(-1,1)---T(0,1)--0->1
	rdm = contract(env.C[(coord,(1,-1))],rdm,([0],[0]))
	if verbosity>0:
		print("rdm=CTCTaTC "+str(rdm.size()))
	# C(-1,-1)--T(0,-1)-----C(1,-1)
	# |		    |		  	    0
	# |		    |		 	    0
	# T(-1,0)---a--2 1------T(1,0) 
	# |		    |\34->23(s,s')  2->0
	# |		    |
	# C(-1,1)---T(0,1)--1
	rdm = contract(env.T[(coord,(1,0))],rdm,([0,1],[0,2]))
	if verbosity>0:
		print("rdm=CTCTaTCT "+str(rdm.size()))
	# C(-1,-1)--T(0,-1)--------C(1,-1)
	# |		    |			     |
	# |		    |		 	     |
	# T(-1,0)---a--------------T(1,0) 
	# |		    |\23->12(s,s')   0
	# |		    |			     0
	# C(-1,1)---T(0,1)--1 1----C(1,1)
	rdm = contract(rdm,env.C[(coord,(1,1))],([0,1],[0,1]))
	if verbosity>0:
		print("rdm=CTCTaTCTC "+str(rdm.size()))

	# symmetrize and normalize
	if operator == None:
		rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)	
	return rdm

def rdm2x1(coord, state, env, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) specifies position of 2x1 subsystem
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type verbosity: int
	:return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
	:rtype: torch.tensor

	Computes 2-site reduced density matrix :math:`\rho_{2x1}` of a horizontal 
	2x1 subsystem using following strategy:
	
		1. compute four individual corners 
		2. construct right and left half of the network
		3. contract right and left halt to obtain final reduced density matrix

	::

		C--T------------T------------------C = C2x2_LU(coord)--C2x2(coord+(1,0))
		|  |			|				  |   |			   |  
		T--A^+A(coord)--A^+A(coord+(1,0))--T   C2x1_LD(coord)--C2x1(coord+(1,0))
		|  |			|				  |
		C--T------------T------------------C 

	The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
	at vertices ``coord``, ``coord+(1,0)`` are left uncontracted
	"""
	who="rdm2x1"
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]
	dimsA = state.site(coord).size()
	a = einsum('mefgh,nabcd->eafbgchdmn',state.site(coord),conj(state.site(coord)))
	a = view(contiguous(a),\
		(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# C--10--T1--2
	# 0	 	 1
	C2x2_LU =contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	  1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU =contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	  	  0
	# |	      0
	# T2--3 1 a--3 
	# 2->1	  2\45
	C2x2_LU =contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 012345->120345
	# reshape (12)(03)45->0123
	# C2x2--1
	# |\23
	# 0
	C2x2_LU= permute(C2x2_LU, (1,2,0,3,4,5))
	C2x2_LU= view(contiguous(C2x2_LU), \
		(T2.size(1)*a.size(2),T1.size(2)*a.size(3),dimsA[0],dimsA[0]))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

	#----- building C2x1_LD ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,1))]
	T2 = env.T[(state.vertexToSite(coord),(0,1))]

	# 0	   0->1
	# C--1 1--T2--2
	C2x1_LD=contract(C, T2, ([1],[1]))

	# reshape (01)2->(0)1
	# 0
	# |
	# C2x1--1
	C2x1_LD= view(contiguous(C2x1_LD), (C.size(0)*T2.size(0),T2.size(2)))
	if verbosity>0:
		print("C2X1 LD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,1): "+str(C2x1_LD.size()))

	#----- build left part C2x2_LU--C2x1_LD ------------------------------------
	# C2x2_LU--1 
	# |\23
	# 0
	# 0
	# C2x1_LD--1->0
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
	left_half= contract(C2x1_LD, C2x2_LU, ([0],[0]))

	#----- building C2x2_RU ----------------------------------------------------
	vec = (1,0)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,-1))]
	T1 = env.T[(shitf_coord,(1,0))]
	T2 = env.T[(shitf_coord,(0,-1))]
	dimsA = state.site(shitf_coord).size()
	a= einsum('mefgh,nabcd->eafbgchdmn',state.site(shitf_coord),conj(state.site(shitf_coord)))
	a= view(contiguous(a), \
		(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	 2
	C2x2_RU =contract(C, T1, ([1],[0]))

	# 2<-0--T2--2 0--C
	#    3<-1	     |
	#		   0<-1--T1
	#			  1<-2
	C2x2_RU =contract(C2x2_RU, T2, ([0],[2]))

	# 1<-2--T2------C
	#	    3	    |
	#	 45\0	    |
	# 2<-1--a--3 0--T1
	#	 3<-2	 0<-1
	C2x2_RU =contract(C2x2_RU, a, ([0,3],[3,0]))

	# permute 012334->120345
	# reshape (12)(03)45->0123
	# 0--C2x2
	# 23/|
	#	 1
	C2x2_RU= permute(C2x2_RU, (1,2,0,3,4,5))
	C2x2_RU= view(contiguous(C2x2_RU), \
		(T2.size(0)*a.size(1),T1.size(2)*a.size(2), dimsA[0], dimsA[0]))
	if verbosity>0:
		print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

	#----- building C2x1_RD ----------------------------------------------------
	C = env.C[(shitf_coord,(1,1))]
	T1 = env.T[(shitf_coord,(0,1))]

	#	 1<-0  	 	 0
	# 2<-1--T1--2 1--C
	C2x1_RD =contract(C, T1, ([1],[2]))

	# reshape (01)2->(0)1
	C2x1_RD = view(contiguous(C2x1_RD), (C.size(0)*T1.size(0),T1.size(1)))

	#	 0
	#	 |
	# 1--C2x1
	if verbosity>0:
		print("C2X1 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x1_RD.size()))

	

	#----- build right part C2x2_RU--C2x1_RD -----------------------------------
	# 1<-0--C2x2_RU
	#	    |\23
	#	    1
	#	    0
	# 0<-1--C2x1_RD
	right_half =contract(C2x1_RD, C2x2_RU, ([0],[1]))

	# construct reduced density matrix by contracting left and right halfs
	# C2x2_LU--1 1----C2x2_RU
	# |\23->01		  |\23
	# |			      |	
	# C2x1_LD--0 0----C2x1_RD
	rdm =contract(left_half,right_half,([0,1],[0,1]))

	# permute into order of s0,s1;s0',s1' where primed indices
	# represent "ket"
	# 0123->0213
	# symmetrize and normalize 
	rdm = contiguous(permute(rdm, (0,2,1,3)))
	rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

	return rdm

def rdm1x2(coord, state, env, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) specifies position of 1x2 subsystem
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type verbosity: int
	:return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
	:rtype: torch.tensor

	Computes 2-site reduced density matrix :math:`\rho_{1x2}` of a vertical 
	1x2 subsystem using following strategy:
	
		1. compute four individual corners 
		2. construct upper and lower half of the network
		3. contract upper and lower halt to obtain final reduced density matrix

	::

		C--T------------------C = C2x2_LU(coord)--------C1x2(coord)
		|  |				  |   |					 |
		T--A^+A(coord)--------T   C2x2_LD(coord+(0,1))--C1x2(coord+0,1))
		|  |				  |
		T--A^+A(coord+(0,1))--T
		|  |				  |
		C--T------------------C

	The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
	at vertices ``coord``, ``coord+(0,1)`` are left uncontracted
	"""
	who="rdm1x2"
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]
	dimsA = state.site(coord).size()
	a= einsum('mefgh,nabcd->eafbgchdmn',state.site(coord), conj(state.site(coord)))
	a= view(contiguous(a), \
		(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# C--10--T1--2
	# 0	  1
	C2x2_LU =contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	     1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU =contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	 	  0
	# |	 	  0
	# T2--3 1 a--3 
	# 2->1 	  2\45
	C2x2_LU =contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 012345->120345
	# reshape (12)(03)45->0123
	# C2x2--1
	# |\23
	# 0
	C2x2_LU= permute(C2x2_LU, (1,2,0,3,4,5))
	C2x2_LU= view(contiguous(C2x2_LU), \
		(T2.size(1)*a.size(2),T1.size(2)*a.size(3),dimsA[0],dimsA[0]))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

	#----- building C1x2_RU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(1,0))]

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	 2
	C1x2_RU =contract(C, T1, ([1],[0]))

	# reshape (01)2->(0)1
	# 0--C1x2
	# 23/|
	#	 1
	C1x2_RU= view(contiguous(C1x2_RU), (C.size(0)*T1.size(1),T1.size(2)))
	if verbosity>0:
		print("C1X2 RU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,-1): "+str(C1x2_RU.size()))

	#----- build upper part C2x2_LU--C1x2_RU -----------------------------------
	# C2x2_LU--1 0--C1x2_RU
	# |\23		    |
	# 0->1		    1->0
	upper_half =contract(C1x2_RU, C2x2_LU, ([0],[1]))

	#----- building C2x2_LD ----------------------------------------------------
	vec = (0,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(-1,1))]
	T1 = env.T[(shitf_coord,(-1,0))]
	T2 = env.T[(shitf_coord,(0,1))]
	dimsA = state.site(shitf_coord).size()
	a= einsum('mefgh,nabcd->eafbgchdmn',state.site(shitf_coord),conj(state.site(shitf_coord)))
	a= view(contiguous(a), \
		(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# 0->1
	# T1--2
	# 1
	# 0
	# C--1->0
	C2x2_LD =contract(C, T1, ([0],[1]))

	# 1->0
	# T1--2->1
	# |
	# |	      0->2
	# C--0 1--T2--2->3
	C2x2_LD =contract(C2x2_LD, T2, ([0],[1]))

	# 0	       0->2
	# T1--1 1--a--3
	# |		   2\45
	# |		   2
	# C--------T2--3->1
	C2x2_LD =contract(C2x2_LD, a, ([1,2],[1,2]))

	# permute 012345->021345
	# reshape (02)(13)45->0123
	# 0
	# |/23
	# C2x2--1
	C2x2_LD= permute(C2x2_LD, (0,2,1,3,4,5))
	C2x2_LD= view(contiguous(C2x2_LD), \
		(T1.size(0)*a.size(0),T2.size(2)*a.size(3), dimsA[0], dimsA[0]))
	if verbosity>0:
		print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

	#----- building C2x2_RD ----------------------------------------------------
	C = env.C[(shitf_coord,(1,1))]
	T2 = env.T[(shitf_coord,(1,0))]

	#	    0
	#	 1--T2
	#	    2
	#	    0
	# 2<-1--C
	C1x2_RD =contract(T2, C, ([2],[0]))

	# permute 012->021
	# reshape 0(12)->0(1)
	C1x2_RD = view(contiguous(permute(C1x2_RD,(0,2,1))), \
		(T2.size()[0],C.size()[1]*T2.size()[1]))

	#	 0
	#	 |
	# 1--C1x2
	if verbosity>0:
		print("C1X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C1x2_RD.size()))

	

	#----- build lower part C2x2_LD--C1x2_RD -----------------------------------
	# 0->1		    0
	# |/23		    |
	# C2x2_LD--1 1--C1x2_RD 
	lower_half =contract(C1x2_RD, C2x2_LD, ([1],[1]))

	# construct reduced density matrix by contracting lower and upper halfs
	# C2x2_LU------C1x2_RU
	# |\23->01	   |
	# 1			   0	
	# 1			   0	
	# |/23		   |
	# C2x2_LD------C1x2_RD
	rdm =contract(upper_half,lower_half,([0,1],[0,1]))

	# permute into order of s0,s1;s0',s1' where primed indices
	# represent "ket"
	# 0123->0213
	# symmetrize and normalize
	rdm = contiguous(permute(rdm, (0,2,1,3)))
	rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

	return rdm

def rdm2x2(coord, state, env, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) specifies upper left site of 2x2 subsystem 
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type verbosity: int
	:return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
	:rtype: torch.tensor

	Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem specified
	by the vertex ``coord`` of its upper left corner using strategy:

		1. compute four individual corners
		2. construct upper and lower half of the network
		3. contract upper and lower half to obtain final reduced density matrix

	::

		C--T------------------T------------------C = C2x2_LU(coord)--------C2x2(coord+(1,0))
		|  |				  |				  |   |					 |
		T--A^+A(coord)--------A^+A(coord+(1,0))--T   C2x2_LD(coord+(0,1))--C2x2(coord+(1,1))
		|  |				  |				  |
		T--A^+A(coord+(0,1))--A^+A(coord+(1,1))--T
		|  |				  |				  |
		C--T------------------T------------------C
		
	The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
	at vertices ``coord``, ``coord+(1,0)``, ``coord+(0,1)``, and ``coord+(1,1)`` are 
	left uncontracted and given in the same order::
		
		s0 s1
		s2 s3

	"""
	who= "rdm2x2"
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]
	dimsA = state.site(coord).size()
	a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(coord),conj(state.site(coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# C--10--T1--2
	# 0	  1
	C2x2_LU = contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	     1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU = contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	      0
	# |	      0
	# T2--3 1 a--3 
	# 2->1	  2\45
	C2x2_LU = contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 012345->120345
	# reshape (12)(03)45->0123
	# C2x2--1
	# |\23
	# 0
	C2x2_LU = contiguous(permute(C2x2_LU,(1,2,0,3,4,5)))
	C2x2_LU = view(C2x2_LU, (T2.size(1)*a.size(2),T1.size(2)*a.size(3),dimsA[0],dimsA[0]))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

	#----- building C2x2_RU ----------------------------------------------------
	vec = (1,0)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,-1))]
	T1 = env.T[(shitf_coord,(1,0))]
	T2 = env.T[(shitf_coord,(0,-1))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	  2
	C2x2_RU = contract(C, T1, ([1],[0]))

	# 2<-0--T2--2 0--C
	#	 3<-1		 |
	#		   0<-1--T1
	#			  1<-2
	C2x2_RU = contract(C2x2_RU, T2, ([0],[2]))

	# 1<-2--T2------C
	#	    3	    |
	#	 45\0	    |
	# 2<-1--a--3 0--T1
	#	 3<-2  	 0<-1
	C2x2_RU = contract(C2x2_RU, a, ([0,3],[3,0]))

	# permute 012334->120345
	# reshape (12)(03)45->0123
	# 0--C2x2
	# 23/|
	#	 1
	C2x2_RU = contiguous(permute(C2x2_RU, (1,2,0,3,4,5)))
	C2x2_RU = view(C2x2_RU, (T2.size(0)*a.size(1),T1.size(2)*a.size(2), dimsA[0], dimsA[0]))
	if verbosity>0:
		print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

	#----- build upper part C2x2_LU--C2x2_RU -----------------------------------
	# C2x2_LU--1 0--C2x2_RU			  C2x2_LU------C2x2_RU
	# |\23->12	 |\23->45   & permute |\12->23	   |\45
	# 0			 1->3				  0			   3->1
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
	upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))
	upper_half = permute(upper_half, (0,3,1,2,4,5))

	#----- building C2x2_RD ----------------------------------------------------
	vec = (1,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,1))]
	T1 = env.T[(shitf_coord,(0,1))]
	T2 = env.T[(shitf_coord,(1,0))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	#	1<-0		0
	# 2<-1--T1--2 1--C
	C2x2_RD = contract(C, T1, ([1],[2]))

	#		  2<-0
	#	   3<-1--T2
	#			 2
	#	 0<-1	 0
	# 1<-2--T1---C
	C2x2_RD = contract(C2x2_RD, T2, ([0],[2]))

	#	 2<-0	 1<-2
	# 3<-1--a--3 3--T2
	#	    2\45	|
	#	    0	    |
	# 0<-1--T1------C
	C2x2_RD = contract(C2x2_RD, a, ([0,3],[2,3]))

	# permute 012345->120345
	# reshape (12)(03)45->0123
	C2x2_RD = contiguous(permute(C2x2_RD, (1,2,0,3,4,5)))
	C2x2_RD = view(C2x2_RD, (T2.size(0)*a.size(0),T1.size(1)*a.size(1), dimsA[0], dimsA[0]))

	#	 0
	#	 |/23
	# 1--C2x2
	if verbosity>0:
		print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))

	#----- building C2x2_LD ----------------------------------------------------
	vec = (0,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(-1,1))]
	T1 = env.T[(shitf_coord,(-1,0))]
	T2 = env.T[(shitf_coord,(0,1))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))

	# 0->1
	# T1--2
	# 1
	# 0
	# C--1->0
	C2x2_LD = contract(C, T1, ([0],[1]))

	# 1->0
	# T1--2->1
	# |
	# |	   	  0->2
	# C--0 1--T2--2->3
	C2x2_LD = contract(C2x2_LD, T2, ([0],[1]))

	# 0		   0->2
	# T1--1 1--a--3
	# |		   2\45
	# |		   2
	# C--------T2--3->1
	C2x2_LD = contract(C2x2_LD, a, ([1,2],[1,2]))

	# permute 012345->021345
	# reshape (02)(13)45->0123
	# 0
	# |/23
	# C2x2--1
	C2x2_LD = contiguous(permute(C2x2_LD, (0,2,1,3,4,5)))
	C2x2_LD = view(C2x2_LD, (T1.size(0)*a.size(0),T2.size(2)*a.size(3), dimsA[0], dimsA[0]))
	if verbosity>0:
		print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

	#----- build lower part C2x2_LD--C2x2_RD -----------------------------------
	# 0			    0->3				 0			  3->1
	# |/23->12	    |/23->45   & permute |/12->23	  |/45
	# C2x2_LD--1 1--C2x2_RD			     C2x2_LD------C2x2_RD
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
	lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))
	lower_half = permute(lower_half, (0,3,1,2,4,5))

	# construct reduced density matrix by contracting lower and upper halfs
	# C2x2_LU------C2x2_RU
	# |\23->01	   |\45->23
	# 0			   1	
	# 0			   1	
	# |/23->45	   |/45->67
	# C2x2_LD------C2x2_RD
	rdm = contract(upper_half,lower_half,([0,1],[0,1]))

	# permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
	# represent "ket"
	# 01234567->02461357
	# symmetrize and normalize
	rdm= contiguous(permute(rdm, (0,2,4,6,1,3,5,7)))
	rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
	
	return rdm
	
######################################################################################################

def fmap(n1, n2, n3):
	# function (n1,n2,n3) --> s that maps the basis of states in the fundamental irrep of SU(3) (states n=0,1,2) for the three sites of the unit cell to a single physical index s=0...26
	# NB: the 3 sites are labeled as:
	#        1---3
	#         \ /
	#          2
	return n3 + 3 * n2 + 9 * n1

def fmap_inv(s):
	# reverse mapping s --> (n1, n2, n3)
	n1 = s // 9
	n2 = (s - 9 * n1) // 3
	n3 = s - 9 * n1 - 3 * n2
	return (n1, n2, n3)

def double_layer_a(A_tensor, csites):
	dimsA = A_tensor.size()
	A_tensor_reshaped = torch.zeros((3,3,3,dimsA[1],dimsA[2],dimsA[3],dimsA[4]),dtype= A_tensor.dtype)
	for s in range(27):
		n1,n2,n3 = fmap_inv(s)
		A_tensor_reshaped[n1,n2,n3,:,:,:,:] = A_tensor[s,:,:,:,:]
	dimsa = tuple([dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2] + len(csites) * [3,3])
	if csites == [1,2,3]:
		contraction = 'mikefgh,njlabcd->eafbgchdmnijkl'
	if csites == [2,3]:
		contraction = 'mikefgh,mjlabcd->eafbgchdijkl'
	if csites == [1,3]:
		contraction = 'mikefgh,nilabcd->eafbgchdmnkl'
	if csites == [1,2]:
		contraction = 'mikefgh,njkabcd->eafbgchdmnij'
	if csites == [1]:
		contraction = 'mikefgh,nikabcd->eafbgchdmn'
	if csites == [2]:
		contraction = 'mikefgh,nilabcd->eafbgchdmnkl'
	if csites == [3]:
		contraction = 'mikefgh,njkabcd->eafbgchdmnij'
	if csites == []:
		contraction = 'mikefgh,mikabcd->eafbgchd'
	a = contiguous(einsum(contraction, A_tensor_reshaped,conj(A_tensor_reshaped)))
	a = view(a, dimsa) 
	return a
	
def enlarged_corner(coord, state, env, corner, csites=[1,2,3], verbosity=0):

	if corner == 'LU':
		C = env.C[(state.vertexToSite(coord),(-1,-1))]
		T1 = env.T[(state.vertexToSite(coord),(0,-1))]
		T2 = env.T[(state.vertexToSite(coord),(-1,0))]
		a = double_layer_a(state.site(coord), csites)
		
		# C--10--T1--2
		# 0	  1
		C2x2_LU = contract(C, T1, ([1],[0]))
	
		# C------T1--2->1
		# 0	     1->0
		# 0
		# T2--2->3
		# 1->2
		C2x2_LU = contract(C2x2_LU, T2, ([0],[0]))

		# C-------T1--1->0
		# |	      0
		# |	      0
		# T2--3 1 a--3 
		# 2->1	  2\...
		C2x2_LU = contract(C2x2_LU, a, ([0,3],[0,1]))

		# permute 0123...->1203...
		# reshape (12)(03)...->01...
		# C2x2--1
		# |\...
		# 0
		C2x2_LU = contiguous(permute(C2x2_LU,tuple([1,2,0,3]+list(range(4,4+2*len(csites))))))
		C2x2_LU = view(C2x2_LU, tuple([T2.size(1)*a.size(2),T1.size(2)*a.size(3)]+len(csites)*[3,3]))
		if verbosity>0:
			print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))
		return(C2x2_LU)
			
	if corner == 'RU':
		vec = (1,0)
		shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
		C = env.C[(shitf_coord,(1,-1))]
		T1 = env.T[(shitf_coord,(1,0))]
		T2 = env.T[(shitf_coord,(0,-1))]
		a = double_layer_a(state.site(shitf_coord),csites)

		# 0--C
		#	 1
		#	 0
		# 1--T1
		#	  2
		C2x2_RU = contract(C, T1, ([1],[0]))

		# 2<-0--T2--2 0--C
		#	 3<-1		 |
		#		   0<-1--T1
		#			  1<-2
		C2x2_RU = contract(C2x2_RU, T2, ([0],[2]))

		# 1<-2--T2------C
		#	 .. 3	    |
		#	   \0	    |
		# 2<-1--a--3 0--T1
		#	 3<-2  	 0<-1
		C2x2_RU = contract(C2x2_RU, a, ([0,3],[3,0]))

		# permute 0123...->1203...
		# reshape (12)(03)...->01...
		# 0--C2x2
		# ../|
		#	 1
		C2x2_RU = contiguous(permute(C2x2_RU, tuple([1,2,0,3]+list(range(4,4+2*len(csites))))))
		C2x2_RU = view(C2x2_RU, tuple([T2.size(0)*a.size(1),T1.size(2)*a.size(2)]+len(csites)*[3,3]))
		if verbosity>0:
			print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))
		
	if corner == 'RD':
		vec = (1,1)
		shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
		C = env.C[(shitf_coord,(1,1))]
		T1 = env.T[(shitf_coord,(0,1))]
		T2 = env.T[(shitf_coord,(1,0))]
		a = double_layer_a(state.site(shitf_coord),csites)

		#  	 1<-0 		 0
		# 2<-1--T1--2 1--C
		C2x2_RD = contract(C, T1, ([1],[2]))

		#		  2<-0
		#	   3<-1--T2
		#			 2
		#	 0<-1	 0
		# 1<-2--T1---C
		C2x2_RD = contract(C2x2_RD, T2, ([0],[2]))

		#	 2<-0	 1<-2
		# 3<-1--a--3 3--T2
		#	    2\...	|
		#	    0	    |
		# 0<-1--T1------C
		C2x2_RD = contract(C2x2_RD, a, ([0,3],[2,3]))

		# permute 0123...->1203...
		# reshape (12)(03)...->01...
		#	 0 ...
		#	 |/
		# 1--C2x2
		C2x2_RD = contiguous(permute(C2x2_RD, tuple([1,2,0,3]+list(range(4,4+2*len(csites))))))
		C2x2_RD = view(C2x2_RD, tuple([T2.size(0)*a.size(0),T1.size(1)*a.size(1)]+len(csites)*[3,3]))
		if verbosity>0:
			print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))
		
	if corner == 'LD':
		vec = (0,1)
		shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
		C = env.C[(shitf_coord,(-1,1))]
		T1 = env.T[(shitf_coord,(-1,0))]
		T2 = env.T[(shitf_coord,(0,1))]
		a = double_layer_a(state.site(shitf_coord),csites)

		# 0->1
		# T1--2
		# 1
		# 0
		# C--1->0
		C2x2_LD = contract(C, T1, ([0],[1]))

		# 1->0
		# T1--2->1
		# |
		# |	   	  0->2
		# C--0 1--T2--2->3
		C2x2_LD = contract(C2x2_LD, T2, ([0],[1]))

		# 0		   0->2
		# T1--1 1--a--3
		# |		   2\...
		# |		   2
		# C--------T2--3->1
		C2x2_LD = contract(C2x2_LD, a, ([1,2],[1,2]))

		# permute 0123...->0213...
		# reshape (02)(13)...->01...
		# 0 ...
		# |/
		# C2x2--1
		C2x2_LD = contiguous(permute(C2x2_LD, tuple([0,2,1,3]+list(range(4,4+2*len(csites))))))
		C2x2_LD = view(C2x2_LD, tuple([T1.size(0)*a.size(0),T2.size(2)*a.size(3)]+len(csites)*[3,3]))
		if verbosity>0:
			print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))
		
		return(C2x2_RU)

def rdm2x2_up_triangle(coord, state, env, operator=None, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) specifies upper left site of 2x2 subsystem 
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type verbosity: int
	:return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
	:rtype: torch.tensor
	"""
	who= "rdm2x2"
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]
	dimsA = state.site(coord).size()
	# contract all physical sites of this unit cell (index m)
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(coord), conj(state.site(coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
	
	# C--10--T1--2
	# 0	     1
	C2x2_LU = contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	     1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU = contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	      0
	# |	      0
	# T2--3 1 a--3 
	# 2->1	  2
	C2x2_LU = contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 0123->1203
	# reshape (12)(03)->01
	# C2x2--1
	# |
	# 0
	C2x2_LU = contiguous(permute(C2x2_LU,(1,2,0,3)))
	C2x2_LU = view(C2x2_LU, (T2.size(1)*a.size(2),T1.size(2)*a.size(3)))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))
	
	#----- building C2x2_RU ----------------------------------------------------
	vec = (1,0)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,-1))]
	T1 = env.T[(shitf_coord,(1,0))]
	T2 = env.T[(shitf_coord,(0,-1))]
	dimsA = state.site(shitf_coord).size()
	# reshape 27 = 3*3*3
	A_reshaped = torch.zeros((3,3,3,dimsA[1],dimsA[2],dimsA[3],dimsA[4]),dtype= state.site(shitf_coord).dtype)
	for s in range(27):
		n1,n2,n3 = fmap_inv(s)
		A_reshaped[n1,n2,n3,:,:,:,:] = state.site(shitf_coord)[s,:,:,:,:]
	# double layer tensor with sites 1 and 3 contracted
	a = contiguous(einsum('mikefgh,mjkabcd->eafbgchdij', A_reshaped,conj(A_reshaped)))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, 3, 3))

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	 2
	C2x2_RU = contract(C, T1, ([1],[0]))

	# 2<-0--T2--2 0--C
	#	 3<-1		 |
	#		   0<-1--T1
	#			  1<-2
	C2x2_RU = contract(C2x2_RU, T2, ([0],[2]))

	# 1<-2--T2------C
	#	    3	    |
	#	 45\0	    |
	# 2<-1--a--3 0--T1
	#	 3<-2	 0<-1
	C2x2_RU = contract(C2x2_RU, a, ([0,3],[3,0]))

	# permute 012334->120345
	# reshape (12)(03)45->0123
	# 0--C2x2
	# 23/|
	#	 1
	C2x2_RU = contiguous(permute(C2x2_RU, (1,2,0,3,4,5)))
	C2x2_RU = view(C2x2_RU, (T2.size(0)*a.size(1),T1.size(2)*a.size(2), 3,3))
	if verbosity>0:
		print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

	#----- build upper part C2x2_LU--C2x2_RU -----------------------------------
	# 
	# C2x2_LU--1 0--C2x2_RU  
	# |				 |\23	  
	# 0				 1	  
	#
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
	upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))

	#----- building C2x2_RD ----------------------------------------------------
	vec = (1,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,1))]
	T1 = env.T[(shitf_coord,(0,1))]
	T2 = env.T[(shitf_coord,(1,0))]
	dimsA = state.site(shitf_coord).size()
	# reshape 27 = 3*3*3
	A_reshaped = torch.zeros((3,3,3,dimsA[1],dimsA[2],dimsA[3],dimsA[4]),dtype= state.site(shitf_coord).dtype)
	for s in range(27):
		n1,n2,n3 = fmap_inv(s)
		A_reshaped[n1,n2,n3,:,:,:,:] = state.site(shitf_coord)[s,:,:,:,:]
	# double layer tensor with sites 2 and 3 contracted
	a = contiguous(einsum('mikefgh,nikabcd->eafbgchdmn', A_reshaped,conj(A_reshaped)))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, 3, 3))

	#	1<-0		0
	# 2<-1--T1--2 1--C
	C2x2_RD = contract(C, T1, ([1],[2]))

	#		  2<-0
	#	   3<-1--T2
	#			 2
	#	 0<-1	 0
	# 1<-2--T1---C
	C2x2_RD = contract(C2x2_RD, T2, ([0],[2]))

	#	 2<-0	 1<-2
	# 3<-1--a--3 3--T2
	#	    2\45	|
	#	    0	    |
	# 0<-1--T1------C
	C2x2_RD = contract(C2x2_RD, a, ([0,3],[2,3]))

	# permute 012345->120345
	# reshape (12)(03)45->0123
	C2x2_RD = contiguous(permute(C2x2_RD, (1,2,0,3,4,5)))
	C2x2_RD = view(C2x2_RD, (T2.size(0)*a.size(0),T1.size(1)*a.size(1), 3,3))

	#	 0
	#	 |/23
	# 1--C2x2
	if verbosity>0:
		print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))

	#----- building C2x2_LD ----------------------------------------------------
	vec = (0,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(-1,1))]
	T1 = env.T[(shitf_coord,(-1,0))]
	T2 = env.T[(shitf_coord,(0,1))]
	dimsA = state.site(shitf_coord).size()
	# reshape 27 = 3*3*3
	A_reshaped = torch.zeros((3,3,3,dimsA[1],dimsA[2],dimsA[3],dimsA[4]),dtype= state.site(shitf_coord).dtype)
	for s in range(27):
		n1,n2,n3 = fmap_inv(s)
		A_reshaped[n1,n2,n3,:,:,:,:] = state.site(shitf_coord)[s,:,:,:,:]
	# double layer tensor with sites 1 and 2 contracted
	a = contiguous(einsum('mikefgh,milabcd->eafbgchdkl', A_reshaped,conj(A_reshaped)))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, 3, 3))

	# 0->1
	# T1--2
	# 1
	# 0
	# C--1->0
	C2x2_LD = contract(C, T1, ([0],[1]))

	# 1->0
	# T1--2->1
	# |
	# |	      0->2
	# C--0 1--T2--2->3
	C2x2_LD = contract(C2x2_LD, T2, ([0],[1]))

	# 0		   0->2
	# T1--1 1--a--3
	# |		   2\45
	# |		   2
	# C--------T2--3->1
	C2x2_LD = contract(C2x2_LD, a, ([1,2],[1,2]))

	# permute 012345->021345
	# reshape (02)(13)45->0123
	# 0
	# |/23
	# C2x2--1
	C2x2_LD = contiguous(permute(C2x2_LD, (0,2,1,3,4,5)))
	C2x2_LD = view(C2x2_LD, (T1.size(0)*a.size(1),T2.size(1)*a.size(1), 3,3))
	if verbosity>0:
		print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

	#----- build lower part C2x2_LD--C2x2_RD -----------------------------------
	# 0			    0->3				 0			  3->1
	# |/23->12	    |/23->45   & permute |/12->23	  |/45
	# C2x2_LD--1 1--C2x2_RD			     C2x2_LD------C2x2_RD
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
	lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))
	lower_half = permute(lower_half, (0,3,1,2,4,5))

	# construct reduced density matrix by contracting lower and upper halfs
	# C2x2_LU------C2x2_RU
	# |				|\23->01
	# 0				1	
	# 0				1	
	# |/23			|/45
	# C2x2_LD------C2x2_RD
	rdm = contract(upper_half,lower_half,([0,1],[0,1]))

	# permute into order of s1,s2,s3;s1',s2',s3' where primed indices
	# represent "ket"
	# 012345 -> 024135
	# symmetrize and normalize
	# [EDIT FOR KAGOME]: the s0;s0' were already contracted out
	rdm= contiguous(permute(rdm, (0,2,4,1,3,5)))
	
	if operator == None:
		rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)	
	else:
		# contract the 3-site operator with the rdm
		value_operator = einsum('ikmjln,ikmjln->',rdm,operator)
		rdm = value_operator
	return rdm

def rdm2x2_dn_triangle(coord, state, env, operator=None, sym_pos_def=False, verbosity=0):
	who='rdm2x2_dn_triangle'
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]

	dimsA = state.site(coord).size()
	if operator==None:
		a = contiguous(einsum('mefgh,nabcd->eafbgchdmn',state.site(coord),conj(state.site(coord))))
		a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0]))
	else:
		a = contiguous(einsum('mefgh,mn,nabcd->eafbgchd',state.site(coord), operator, conj(state.site(coord))))
		a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# C--10--T1--2
	# 0	  	 1
	C2x2_LU = contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	     1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU = contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	      0
	# |	      0
	# T2--3 1 a--3 
	# 2->1	  2
	C2x2_LU = contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 0123->1203
	# reshape (12)(03)->01
	# C2x2--1
	# |\23
	# 0
	C2x2_LU = contiguous(permute(C2x2_LU,(1,2,0,3)))
	C2x2_LU = view(C2x2_LU, (T2.size(1)*a.size(2),T1.size(2)*a.size(3)))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

	#----- building C2x2_RU ----------------------------------------------------
	vec = (1,0)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,-1))]
	T1 = env.T[(shitf_coord,(1,0))]
	T2 = env.T[(shitf_coord,(0,-1))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	  2
	C2x2_RU = contract(C, T1, ([1],[0]))

	# 2<-0--T2--2 0--C
	#	 3<-1		 |
	#		   0<-1--T1
	#			  1<-2
	C2x2_RU = contract(C2x2_RU, T2, ([0],[2]))

	# 1<-2--T2------C
	#	    3	    |
	#	    0	    |
	# 2<-1--a--3 0--T1
	#	 3<-2  	 0<-1
	C2x2_RU = contract(C2x2_RU, a, ([0,3],[3,0]))

	# permute 0123->1203
	# reshape (12)(03)->01
	# 0--C2x2
	#    |
	#	 1
	C2x2_RU = contiguous(permute(C2x2_RU, (1,2,0,3)))
	C2x2_RU = view(C2x2_RU, (T2.size(0)*a.size(1),T1.size(2)*a.size(2)))
	if verbosity>0:
		print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

	#----- build upper part C2x2_LU--C2x2_RU -----------------------------------
	# C2x2_LU--1 0--C2x2_RU
	# |          	 |
	# 0			     1
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
	upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))

	#----- building C2x2_RD ----------------------------------------------------
	vec = (1,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,1))]
	T1 = env.T[(shitf_coord,(0,1))]
	T2 = env.T[(shitf_coord,(1,0))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	#	1<-0		0
	# 2<-1--T1--2 1--C
	C2x2_RD = contract(C, T1, ([1],[2]))

	#		  2<-0
	#	   3<-1--T2
	#			 2
	#	 0<-1	 0
	# 1<-2--T1---C
	C2x2_RD = contract(C2x2_RD, T2, ([0],[2]))

	#	 2<-0	 1<-2
	# 3<-1--a--3 3--T2
	#	    2   	|
	#	    0	    |
	# 0<-1--T1------C
	C2x2_RD = contract(C2x2_RD, a, ([0,3],[2,3]))

	# permute 0123->1203
	# reshape (12)(03)->01
	C2x2_RD = contiguous(permute(C2x2_RD, (1,2,0,3)))
	C2x2_RD = view(C2x2_RD, (T2.size(0)*a.size(0),T1.size(1)*a.size(1)))

	#	 0
	#	 |
	# 1--C2x2
	if verbosity>0:
		print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))

	#----- building C2x2_LD ----------------------------------------------------
	vec = (0,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(-1,1))]
	T1 = env.T[(shitf_coord,(-1,0))]
	T2 = env.T[(shitf_coord,(0,1))]
	dimsA = state.site(shitf_coord).size()
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord),conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# 0->1
	# T1--2
	# 1
	# 0
	# C--1->0
	C2x2_LD = contract(C, T1, ([0],[1]))

	# 1->0
	# T1--2->1
	# |
	# |	   	  0->2
	# C--0 1--T2--2->3
	C2x2_LD = contract(C2x2_LD, T2, ([0],[1]))

	# 0		   0->2
	# T1--1 1--a--3
	# |		   2
	# |		   2
	# C--------T2--3->1
	C2x2_LD = contract(C2x2_LD, a, ([1,2],[1,2]))

	# permute 0123->0213
	# reshape (02)(13)->01
	# 0
	# |
	# C2x2--1
	C2x2_LD = contiguous(permute(C2x2_LD, (0,2,1,3)))
	C2x2_LD = view(C2x2_LD, (T1.size(0)*a.size(0),T2.size(2)*a.size(3)))
	if verbosity>0:
		print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

	#----- build lower part C2x2_LD--C2x2_RD -----------------------------------
	# 0			    0->1
	# |   	        |         
	# C2x2_LD--1 1--C2x2_RD	
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
	lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))

	# construct reduced density matrix by contracting lower and upper halfs
	# C2x2_LU------C2x2_RU
	# |            |
	# 0			   1	
	# 0			   1	
	# |            |
	# C2x2_LD------C2x2_RD
	rdm = contract(upper_half,lower_half,([0,1],[0,1]))

	if operator == None:
		rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)	
	return rdm
	
def rdm2x2_nnn_1(coord, state, env, operator, verbosity=0):

	C2x2_LU = enlarged_corner(coord, state, env, 'LU', csites=[], verbosity=verbosity)
	C2x2_RD = enlarged_corner(coord, state, env, 'RD', csites=[], verbosity=verbosity)

	# bond 1--2
	C2x2_LD = enlarged_corner(coord, state, env, 'LD', csites=[1], verbosity=verbosity)
	C2x2_RU = enlarged_corner(coord, state, env, 'RU', csites=[2], verbosity=verbosity)
	upper_half = einsum('ij,jkab->ijab', C2x2_LU, C2x2_RU)
	lower_half = einsum('ijab,kj->ikab', C2x2_LD, C2x2_RD)
	bond12 = einsum('ijab,abcd,ijcd->',upper_half,operator,lower_half)

	# bond 3--1
	C2x2_LD = enlarged_corner(coord, state, env, 'LD', csites=[3], verbosity=verbosity)
	C2x2_RU = enlarged_corner(coord, state, env, 'RU', csites=[1], verbosity=verbosity)
	upper_half= einsum('ij,jkab->ijab', C2x2_LU, C2x2_RU)
	lower_half = einsum('ijab,kj->ikab', C2x2_LD, C2x2_RD)
	bond31 = einsum('ijab,abcd,ijcd->', upper_half, operator, lower_half)

	return(bond31 + bond12)

def rdm2x2_nnn_2(coord, state, env, operator, verbosity=0):

	#--------------upper half -------------------------------------------------
	 
	# build upper part C2x2_LU--C2x2_RU and contract with the 2-cell operator
	# C2x2_LU-----1     0-----C2x2_RU		
	# |\23________op_______23/| 
	# 0			              1
	
	# NNN bond 3--2
	C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[3], verbosity=verbosity)
	C2x2_RU = enlarged_corner(coord, state, env, corner='RU', csites=[2], verbosity=verbosity)
	upper_half_32 = einsum('ijab,abcd,jkcd->ij',C2x2_LU,operator,C2x2_RU)
	
	# NNN bond 2--1
	C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[2], verbosity=verbosity)
	C2x2_RU = enlarged_corner(coord, state, env, corner='RU', csites=[1], verbosity=verbosity)
	upper_half_21 = einsum('ijab,abcd,jkcd->ij',C2x2_LU,operator,C2x2_RU)
	
	#--------------bottom half-------------------------------------------------
	
	# 0			    0->1	
	# |     	    |        
	# C2x2_LD--1 1--C2x2_RD	
	C2x2_RD = enlarged_corner(coord, state, env, corner='RD', csites=[], verbosity=verbosity)
	C2x2_LD = enlarged_corner(coord, state, env, corner='LD', csites=[], verbosity=verbosity)
	lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))

	# contracting lower and upper halfs
	# C2x2_LU--op--C2x2_RU
	# |	           |
	# 0			   1	
	# 0			   1	
	# |	  		   |
	# C2x2_LD------C2x2_RD
	rdm = contract(upper_half_32 + upper_half_21,lower_half,([0,1],[0,1]))
	
	return(rdm)	
	
def rdm2x2_nnn_3(coord, state, env, operator, verbosity=0):

	#---------------- left half -----------------------------------
	
	# build left half and contract with the 2-cell operator
	# C2x2_LU--1->0
	# |\2345
	# |     \
	# 0      op
	# 0      /
	# |     /
	# |/2345
	# C2x2_LD--1
	
	# NN bond 3--1
	C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[3], verbosity=verbosity)
	C2x2_LD = enlarged_corner(coord, state, env, corner='LD', csites=[1], verbosity=verbosity)
	left_half_31 = einsum('ijab,abcd,ikcd->jk', C2x2_LU, operator, C2x2_LD)
	
	# NN bond 2--3
	C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[2], verbosity=verbosity)
	C2x2_LD = enlarged_corner(coord, state, env, corner='LD', csites=[3], verbosity=verbosity)
	left_half_23 = einsum('ijab,abcd,ikcd->jk', C2x2_LU, operator, C2x2_LD)
	
	#---------------- right half -----------------------------------

	# 0--C2x2_RU
	#    |
	#	 1
	#	 0
	#	 |
	# 1--C2x2_RD
	C2x2_RU = enlarged_corner(coord, state, env, corner='RU', csites=[], verbosity=verbosity)
	C2x2_RD = enlarged_corner(coord, state, env, corner='RD', csites=[], verbosity=verbosity)
	right_half = contract(C2x2_RU, C2x2_RD, ([1],[0]))

	# construct reduced density matrix by contracting left and right halves
	# C2x2_LU-0--0-C2x2_RU
	# |	           |
	# op		   |	
	# |		       |
	# |     	   |
	# C2x2_LD-1--1-C2x2_RD
	rdm = contract(left_half_31 + left_half_23,right_half,([0,1],[0,1]))

	return rdm

def rdm2x2_id(coord, state, env, sym_pos_def=False, verbosity=0):
	r"""
	:param coord: vertex (x,y) specifies upper left site of 2x2 subsystem 
	:param state: underlying wavefunction
	:param env: environment corresponding to ``state``
	:param verbosity: logging verbosity
	:type coord: tuple(int,int) 
	:type state: IPEPS
	:type env: ENV
	:type verbosity: int
	:return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
	:rtype: torch.tensor
	
	Special case where the operator = the identity. This funciton thus computes the norm of the wavefunction < psi | psi >
	"""
	who= "rdm2x2"
	#----- building C2x2_LU ----------------------------------------------------
	C = env.C[(state.vertexToSite(coord),(-1,-1))]
	T1 = env.T[(state.vertexToSite(coord),(0,-1))]
	T2 = env.T[(state.vertexToSite(coord),(-1,0))]
	dimsA = state.site(coord).size()
	# contract all physical sites of this unit cell (index m)
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(coord), conj(state.site(coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# C--10--T1--2
	# 0	     1
	C2x2_LU = contract(C, T1, ([1],[0]))

	# C------T1--2->1
	# 0	     1->0
	# 0
	# T2--2->3
	# 1->2
	C2x2_LU = contract(C2x2_LU, T2, ([0],[0]))

	# C-------T1--1->0
	# |	      0
	# |	      0
	# T2--3 1 a--3 
	# 2->1	  2
	C2x2_LU = contract(C2x2_LU, a, ([0,3],[0,1]))

	# permute 0123->1203
	# reshape (12)(03)->01
	# C2x2--1
	# |
	# 0
	C2x2_LU = contiguous(permute(C2x2_LU,(1,2,0,3)))
	C2x2_LU = view(C2x2_LU, (T2.size(1)*a.size(2),T1.size(2)*a.size(3)))
	if verbosity>0:
		print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))
	
	#----- building C2x2_RU ----------------------------------------------------
	vec = (1,0)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,-1))]
	T1 = env.T[(shitf_coord,(1,0))]
	T2 = env.T[(shitf_coord,(0,-1))]
	dimsA = state.site(shitf_coord).size()
	# contract all physical sites of this unit cell (index m)
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord), conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# 0--C
	#	 1
	#	 0
	# 1--T1
	#	 2
	C2x2_RU = contract(C, T1, ([1],[0]))

	# 2<-0--T2--2 0--C
	#	 3<-1		 |
	#		   0<-1--T1
	#			  1<-2
	C2x2_RU = contract(C2x2_RU, T2, ([0],[2]))

	# 1<-2--T2------C
	#	    3	    |
	#	    0	    |
	# 2<-1--a--3 0--T1
	#	 3<-2	 0<-1
	C2x2_RU = contract(C2x2_RU, a, ([0,3],[3,0]))

	# permute 012334->1203
	# reshape (12)(03)->01
	# 0--C2x2
	#    |
	#	 1
	C2x2_RU = contiguous(permute(C2x2_RU, (1,2,0,3)))
	C2x2_RU = view(C2x2_RU, (T2.size(0)*a.size(1),T1.size(2)*a.size(2)))
	if verbosity>0:
		print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

	#----- build upper part C2x2_LU--C2x2_RU -----------------------------------
	# 
	# C2x2_LU--1 0--C2x2_RU  
	# |				 |  
	# 0				 1	  
	#
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
	upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))

	#----- building C2x2_RD ----------------------------------------------------
	vec = (1,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(1,1))]
	T1 = env.T[(shitf_coord,(0,1))]
	T2 = env.T[(shitf_coord,(1,0))]
	dimsA = state.site(shitf_coord).size()
	# contract all physical sites of this unit cell (index m)
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord), conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	#	 1<-0	   	0
	# 2<-1--T1--2 1--C
	C2x2_RD = contract(C, T1, ([1],[2]))

	#		  2<-0
	#	   3<-1--T2
	#			 2
	#	 0<-1	 0
	# 1<-2--T1---C
	C2x2_RD = contract(C2x2_RD, T2, ([0],[2]))

	#	 2<-0	 1<-2
	# 3<-1--a--3 3--T2
	#	    2   	|
	#	    0	    |
	# 0<-1--T1------C
	C2x2_RD = contract(C2x2_RD, a, ([0,3],[2,3]))

	# permute 0123->1203
	# reshape (12)(03)->01
	C2x2_RD = contiguous(permute(C2x2_RD, (1,2,0,3)))
	C2x2_RD = view(C2x2_RD, (T2.size(0)*a.size(0),T1.size(1)*a.size(1)))

	#	 0
	#	 |
	# 1--C2x2
	if verbosity>0:
		print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))

	#----- building C2x2_LD ----------------------------------------------------
	vec = (0,1)
	shitf_coord = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
	C = env.C[(shitf_coord,(-1,1))]
	T1 = env.T[(shitf_coord,(-1,0))]
	T2 = env.T[(shitf_coord,(0,1))]
	dimsA = state.site(shitf_coord).size()
	# contract all physical sites of this unit cell (index m)
	a = contiguous(einsum('mefgh,mabcd->eafbgchd',state.site(shitf_coord), conj(state.site(shitf_coord))))
	a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

	# 0->1
	# T1--2
	# 1
	# 0
	# C--1->0
	C2x2_LD = contract(C, T1, ([0],[1]))

	# 1->0
	# T1--2->1
	# |
	# |	      0->2
	# C--0 1--T2--2->3
	C2x2_LD = contract(C2x2_LD, T2, ([0],[1]))

	# 0		   0->2
	# T1--1 1--a--3
	# |		   2
	# |		   2
	# C--------T2--3->1
	C2x2_LD = contract(C2x2_LD, a, ([1,2],[1,2]))

	# permute 0123->0213
	# reshape (02)(13)->01
	# 0
	# |
	# C2x2--1
	C2x2_LD = contiguous(permute(C2x2_LD, (0,2,1,3)))
	C2x2_LD = view(C2x2_LD, (T1.size(0)*a.size(1),T2.size(1)*a.size(1)))
	if verbosity>0:
		print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

	#----- build lower part C2x2_LD--C2x2_RD -----------------------------------
	# 0			    0->1
	# |     	    |        
	# C2x2_LD--1 1--C2x2_RD		
	# TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
	lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))

	# construct reduced density matrix by contracting lower and upper halfs
	# C2x2_LU------C2x2_RU
	# |				|
	# 0				1	
	# 0				1	
	# |  			|
	# C2x2_LD------C2x2_RD
	rdm = contract(upper_half,lower_half,([0,1],[0,1]))

	return rdm

