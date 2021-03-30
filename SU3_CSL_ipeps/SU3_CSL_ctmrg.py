import torch
import math
import config as cfg
import copy
from collections import OrderedDict
from u1sym.ipeps_u1 import IPEPS_U1SYM
from read_write_SU3_tensors import *
from models import SU3_chiral
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
import json
import unittest
import logging
log = logging.getLogger(__name__)
 

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--theta", type=float, default=0., help="rotation angle of the model")
parser.add_argument("--top_freq", type=int, default=-1, help="frequency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues of transfer operator to compute")
args, unknown_args= parser.parse_known_args()

def main():

	cfg.configure(args)
	cfg.print_config()
	print('\n')
	torch.set_num_threads(args.omp_cores)
	torch.manual_seed(args.seed)
	
	# Import all elementary tensors and build initial state
	elementary_tensors = []
	for name in ['S0','S1','S2','S3','S4','L0','L1','L2']:
		ts = load_SU3_tensor(name)
		elementary_tensors.append(ts)
	coeffs = {(0,0): torch.tensor([0.,0.,0.,0.,0.,0.],dtype=torch.complex128)}
	state = IPEPS_U1SYM(elementary_tensors, coeffs)
	
	model = SU3_chiral.SU3_CHIRAL(theta = args.theta)
	
	def energy_f(state, env):
		return model.energy_triangle(state,env)
	
	def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
		if not history:
			history=[]
		e_curr= energy_f(state,env)
		history.append(e_curr)
		print('Step n°'+str(len(history))+'     E_site = '+str(e_curr))
		if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
			or len(history) >= ctm_args.ctm_max_iter:
			log.info({"history_length": len(history), "history": history})
			return True, history
		return False, history

	ctm_env_init = ENV(args.chi, state)
	init_env(state, ctm_env_init)
	
	e_dn_init = energy_f(state, ctm_env_init)
	print('*** Energy per site (before CTMRG) -- down triangles: '+str(e_dn_init))
	
	ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
	
	e_dn_final = energy_f(state,ctm_env_out)
	print('*** Energy per site (after CTMRG) -- down triangles: '+str(e_dn_final))

	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()

