import torch
import numpy as np
import json
from ipeps.tensor_io import *

path = 'SU3_CSL_ipeps/SU3_tensors/'

def write_json_to_file(tensor, outputfile):
	serialized_tensor = serialize_bare_tensor_legacy(tensor)
	with open(outputfile,'w') as f:
		json.dump(serialized_tensor, f, indent=4, separators=(',', ': '))

def write_SU3_tensors():

	#______ Trivalent tensors _______
	S0 = torch.zeros((7,7,7),dtype=torch.complex128)
	S1 = torch.zeros((7,7,7),dtype=torch.complex128)
	S2 = torch.zeros((7,7,7),dtype=torch.complex128)
	S3 = torch.zeros((7,7,7),dtype=torch.complex128)
	S4 = torch.zeros((7,7,7),dtype=torch.complex128)

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

	for tensor,name in zip([S0,S1,S2,S3,S4,L0,L1,L2],['S0','S1','S2','S3','S4','L0','L1','L2']):
		filename = path+name+'.json'
		write_json_to_file(tensor,filename)
		

def load_SU3_tensor(name):
	with open(path+name+'.json') as j:
		# load tensor as a json file
		tensor = json.load(j)
		# convert to torch.tensor object
		tensor = torch.tensor(read_bare_json_tensor_np_legacy(tensor), dtype=torch.complex128)
		return(tensor)

