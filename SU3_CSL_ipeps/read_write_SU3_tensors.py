import torch
import numpy as np
import json
from ipeps.tensor_io import *

def write_json_to_file(tensor, outputfile):
	serialized_tensor = serialize_bare_tensor_legacy(tensor)
	with open(outputfile,'w') as f:
		json.dump(serialized_tensor, f, indent=4, separators=(',', ': '))

def write_SU3_D7_tensors():

	# D=7, virtual space V = 3 + \bar{3} + 1

	#______ Trivalent tensors _______
	S0 = torch.zeros((7,7,7),dtype=torch.complex128)
	S1 = torch.zeros((7,7,7),dtype=torch.complex128)
	S2 = torch.zeros((7,7,7),dtype=torch.complex128)
	S3 = torch.zeros((7,7,7),dtype=torch.complex128)
	S4 = torch.zeros((7,7,7),dtype=torch.complex128)
	S5 = torch.zeros((7,7,7),dtype=torch.complex128)
	S6 = torch.zeros((7,7,7),dtype=torch.complex128)
	S7 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	S8 = torch.zeros((7, 7, 7), dtype=torch.complex128)

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
	
	# S5: {1,1,1}, E1
	S5[0,5,6] = S5[2,3,6] = 1./3.
	S5[1,4,6] = -1./3.
	S5[3,6,2] = S5[5,6,0] = 1j *(1j+np.sqrt(3.))/6.
	S5[4,6,1] = (1.-1j*np.sqrt(3.))/6.
	S5[6,0,5] = -1j*(-1j+np.sqrt(3.))/6.
	S5[6,1,4] = (1+1j*np.sqrt(3.))/6.
	S5[6,2,3] = -1j*(-1j+np.sqrt(3.))/6.
	
	# S6: {1,1,1}, E1
	S6[0,6,5] = S6[2,6,3] = 1./3.
	S6[1,6,4] = -1./3.
	S6[3,2,6] = S6[5,0,6] = -1j *(-1j+np.sqrt(3.))/6.
	S6[4,1,6] = (1.+1j*np.sqrt(3.))/6.
	S6[6,5,0] = 1j*(1j+np.sqrt(3.))/6.
	S6[6,4,1] = (1-1j*np.sqrt(3.))/6.
	S6[6,3,2] = 1j*(1j+np.sqrt(3.))/6.

	# S7: {1,1,1}, E2
	S7[0, 5, 6] = S7[2, 3, 6] = 1. / 3.
	S7[1, 4, 6] = -1. / 3.
	S7[3, 6, 2] = S7[5, 6, 0] = -1j * (-1j + np.sqrt(3.)) / 6.
	S7[4, 6, 1] = (1. + 1j * np.sqrt(3.)) / 6.
	S7[6, 0, 5] = 1j * (1j + np.sqrt(3.)) / 6.
	S7[6, 1, 4] = (1 - 1j * np.sqrt(3.)) / 6.
	S7[6, 2, 3] = 1j * (1j + np.sqrt(3.)) / 6.

	# S8: {1,1,1}, E2
	S8[0, 6, 5] = S8[2, 6, 3] = 1. / 3.
	S8[1, 6, 4] = -1. / 3.
	S8[3, 2, 6] = S8[5, 0, 6] = 1j * (1j + np.sqrt(3.)) / 6.
	S8[4, 1, 6] = (1. - 1j * np.sqrt(3.)) / 6.
	S8[6, 5, 0] = -1j * (-1j + np.sqrt(3.)) / 6.
	S8[6, 4, 1] = (1 + 1j * np.sqrt(3.)) / 6.
	S8[6, 3, 2] = -1j * (-1j + np.sqrt(3.)) / 6.
	#________________________________


	#______ Bivalent tensors ________
	L0 = torch.zeros((3,7,7),dtype=torch.complex128)
	L1 = torch.zeros((3,7,7),dtype=torch.complex128)
	L2 = torch.zeros((3,7,7),dtype=torch.complex128)

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


	for tensor,name in zip([S0,S1,S2,S3,S4,S5,S6,S7,S8,L0,L1,L2],['S0','S1','S2','S3','S4','S5','S6','S7','S8','L0','L1','L2']):
		path = "SU3_CSL_ipeps/SU3_D7_tensors/"
		filename = path+name+'.json'
		write_json_to_file(tensor,filename)
	

def write_SU3_D6_tensors():
	# D=6, virtual space V = \bar{3} + \bar{3}

	# ______ Trivalent tensors _______
	M0 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M1 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M2 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M3 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M4 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M5 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M6 = torch.zeros((7, 7, 7), dtype=torch.complex128)
	M7 = torch.zeros((7, 7, 7), dtype=torch.complex128)

	# M0: {3,0}, A2
	M0[0, 1, 2] = M0[1, 2, 0] = M0[2, 0, 1] = -1 / np.sqrt(6)
	M0[0, 2, 1] = M0[1, 0, 2] = M0[2, 1, 0] = 1 / np.sqrt(6)

	# M1: {2,1}, A2
	M1[0, 1, 5] = M1[5, 0, 1] = M1[1, 5, 0] = -1 / (3 * np.sqrt(2))
	M1[0, 5, 1] = M1[5, 1, 0] = M1[1, 0, 5] = 1 / (3 * np.sqrt(2))
	M1[1, 2, 3] = M1[2, 3, 1] = M1[3, 1, 2] = -1 / (3 * np.sqrt(2))
	M1[1, 3, 2] = M1[2, 1, 3] = M1[3, 2, 1] = 1 / (3 * np.sqrt(2))
	M1[0, 4, 2] = M1[2, 0, 4] = M1[4, 2, 0] = -1 / (3 * np.sqrt(2))
	M1[0, 2, 4] = M1[2, 4, 0] = M1[4, 0, 2] = 1 / (3 * np.sqrt(2))

	# M2: {1,2}, A2
	M2[0, 4, 5] = M2[0, 5, 4] = M2[4, 5, 0] = -1 / (3 * np.sqrt(2))
	M2[0, 5, 4] = M2[0, 4, 5] = M2[4, 0, 5] = 1 / (3 * np.sqrt(2))
	M2[1, 5, 3] = M2[5, 3, 1] = M2[3, 1, 5] = -1 / (3 * np.sqrt(2))
	M2[1, 3, 5] = M2[5, 1, 3] = M2[3, 5, 1] = 1 / (3 * np.sqrt(2))
	M2[2, 3, 4] = M2[3, 4, 2] = M2[4, 2, 3] = -1 / (3 * np.sqrt(2))
	M2[2, 4, 3] = M2[3, 2, 4] = M2[4, 3, 2] = 1 / (3 * np.sqrt(2))

	# M3: {0,3}, A2
	M3[3, 4, 5] = M3[5, 3, 4] = M3[4, 5, 3] = -1 / np.sqrt(6)
	M3[3, 5, 4] = M3[5, 4, 3] = M3[4, 3, 5] = 1 / np.sqrt(6)

	# M4: {2,1}, E1
	om = np.exp(1j * np.pi / 3.)
	M4[0,1,5] = M4[1,2,3] = M4[2,0,4] = -1/(3*np.sqrt(2))
	M4[1,0,5] = M4[2,1,3] = M4[0,2,4] = 1/(3*np.sqrt(2))
	M4[0,5,1] = M4[1,3,2] = M4[2,4,0] = -om/(3*np.sqrt(2))
	M4[1,5,0] = M4[2,3,1] = M4[0,4,2] = om/(3*np.sqrt(2))
	M4[5,0,1] = M4[3,1,2] = M4[4,2,0] = (1/om) /(3*np.sqrt(2))
	M4[5,1,0] = M4[3,2,1] = M4[4,0,2] = - (1/om) / (3 * np.sqrt(2))

	# M5: {1,2}, E1
	M5[5, 3, 1] = M5[3, 4, 2] = M5[4, 5, 0] = - 1 / (3 * np.sqrt(2))
	M5[3, 5, 1] = M5[4, 3, 2] = M5[5, 4, 0] = 1 / (3 * np.sqrt(2))
	M5[5, 1, 3] = M5[3, 2, 4] = M5[4, 0, 5] = - om / (3 * np.sqrt(2))
	M5[3, 1, 5] = M5[4, 2, 3] = M5[5, 0, 4] = om / (3 * np.sqrt(2))
	M5[1, 3, 5] = M5[2, 4, 3] = M5[0, 5, 4] = - (1/om) / (3 * np.sqrt(2))
	M5[1, 5, 3] = M5[2, 3, 4] = M5[0, 4, 5] = (1/om) / (3 * np.sqrt(2))

	# M6: {2,1}, E2
	om = np.exp(-1j * np.pi / 3.)
	M6[0, 1, 5] = M6[1, 2, 3] = M6[2, 0, 4] = -1 / (3 * np.sqrt(2))
	M6[1, 0, 5] = M6[2, 1, 3] = M6[0, 2, 4] = 1 / (3 * np.sqrt(2))
	M6[0, 5, 1] = M6[1, 3, 2] = M6[2, 4, 0] = -om / (3 * np.sqrt(2))
	M6[1, 5, 0] = M6[2, 3, 1] = M6[0, 4, 2] = om / (3 * np.sqrt(2))
	M6[5, 0, 1] = M6[3, 1, 2] = M6[4, 2, 0] = (1/om) / (3 * np.sqrt(2))
	M6[5, 1, 0] = M6[3, 2, 1] = M6[4, 0, 2] = - (1/om) / (3 * np.sqrt(2))

	# M7: {1,2}, E2
	M7[5, 3, 1] = M7[3, 4, 2] = M7[4, 5, 0] = - 1 / (3 * np.sqrt(2))
	M7[3, 5, 1] = M7[4, 3, 2] = M7[5, 4, 0] = 1 / (3 * np.sqrt(2))
	M7[5, 1, 3] = M7[3, 2, 4] = M7[4, 0, 5] = - om / (3 * np.sqrt(2))
	M7[3, 1, 5] = M7[4, 2, 3] = M7[5, 0, 4] = om / (3 * np.sqrt(2))
	M7[1, 3, 5] = M7[2, 4, 3] = M7[0, 5, 4] = - (1/om) / (3 * np.sqrt(2))
	M7[1, 5, 3] = M7[2, 3, 4] = M7[0, 4, 5] = (1/om) / (3 * np.sqrt(2))
	# ________________________________

	# ______ Bivalent tensors ________
	L0 = torch.zeros((3, 7, 7), dtype=torch.complex128)
	L1 = torch.zeros((3, 7, 7), dtype=torch.complex128)
	L2 = torch.zeros((3, 7, 7), dtype=torch.complex128)
	L3 = torch.zeros((3, 7, 7), dtype=torch.complex128)

	# L0: {2, 0}, B
	L0[0, 0, 1] = L0[1, 0, 2] = L0[2, 1, 2] = -1. / np.sqrt(2.)
	L0[0, 1, 0] = L0[1, 2, 0] = L0[2, 2, 1] = 1. / np.sqrt(2.)

	# L1: {1, 1}, B
	L1[0, 4, 0] = L1[0, 1, 3] = L1[1, 5, 0] = L1[1, 2, 3] = L1[2, 2, 4] = L1[2, 5, 1] = 1. / 2.
	L1[0, 0, 4] = L1[0, 3, 1] = L1[1, 0, 5] = L1[1, 3, 2] = L1[2, 4, 2] = L1[2, 1, 5] = - 1. / 2.

	# L2: {0, 2}, B
	L2[0, 3, 4] = L2[1, 3, 5] = L2[2, 4, 5] = -1. / np.sqrt(2.)
	L2[0, 4, 3] = L2[1, 5, 3] = L2[2, 5, 4] = 1. / np.sqrt(2.)

	# L3: {1, 1}, A
	L3[0, 1, 3] = L3[0, 3, 1] = L3[1, 2, 3] = L3[1, 3, 2] = L3[2, 2, 4] = L3[2, 4, 2] = 1. / 2.
	L3[0, 0, 4] = L3[0, 4, 0] = L3[1, 0, 5] = L3[1, 5, 0] = L3[2, 1, 5] = L3[2, 5, 1] = - 1. / 2.
	# ________________________________

	for tensor, name in zip([M0, M1, M2, M3, M4, M5, M6, M7, L0, L1, L2, L3],
							['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'L0', 'L1', 'L2', 'L3']):
		path = "SU3_CSL_ipeps/SU3_D6_tensors/"
		filename = path + name + '.json'
		write_json_to_file(tensor, filename)


def load_SU3_tensor(name):
	with open(name+'.json') as j:
		# load tensor as a json file
		tensor = json.load(j)
		# convert to torch.tensor object
		tensor = torch.tensor(read_bare_json_tensor_np_legacy(tensor), dtype=torch.complex128)
		return(tensor)

