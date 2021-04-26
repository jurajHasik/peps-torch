import context
import argparse
import config as cfg
import math
import torch
import copy
from random import randint
from collections import OrderedDict
from u1sym.ipeps_u1 import IPEPS_U1SYM, write_coeffs, read_coeffs
from read_write_SU3_tensors import *
from models import SU3_chiral
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import rdm
from optim.fd_optim_lbfgs_mod import optimize_state
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--frac_theta", type=float, default=0., help="rotation angle of the model")
args, unknown_args= parser.parse_known_args()

def main():

	cfg.configure(args)
	cfg.print_config()
	print('\n')
	torch.set_num_threads(args.omp_cores)
	rseed = randint(1,args.seed)
	torch.manual_seed(rseed)
	#torch.manual_seed(args.seed)
	
	# Import all elementary tensors and build initial state
	elementary_tensors = []
	for name in ['S0','S1','S2','S3','S4','S5','S6','L0','L1','L2']:
		ts = load_SU3_tensor(name)
		elementary_tensors.append(ts)
	# define initial coefficients
	coeffs = {(0,0): torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],dtype=torch.float64)}
	# define which coefficients will be added a noise and will be allowed to vary
	var_coeffs_allowed = torch.tensor([0,0,0,0,0,1,1, 1,1,0])
	state = IPEPS_U1SYM(elementary_tensors, coeffs, var_coeffs_allowed)
	state.add_noise(args.instate_noise)
	print(f'Current state: {state.coeffs[(0,0)].data}')
	
	model = SU3_chiral.SU3_CHIRAL(theta = math.pi * args.frac_theta / 100.0)
	
	def energy_f(state, env):
		e_dn = model.energy_triangle(state,env)
		e_up = model.energy_triangle_up(state,env)
		#print(f'E_up={e_up.item()}, E_dn={e_dn.item()}')
		return((e_up+e_dn)/2)
		
	@torch.no_grad()
	def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
		if not history:
			history=[]
		e_curr= energy_f(state,env)
		history.append(e_curr.item())
		print('CTMRG step n°'+str(len(history))+'     E_site = '+str(e_curr.item()))
		if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
			or len(history) >= ctm_args.ctm_max_iter:
			log.info({"history_length": len(history), "history": history})
			return True, history
		return False, history
		
	ctm_env = ENV(args.chi, state)
	init_env(state, ctm_env)
	
	def loss_fn(state, ctm_env_in, opt_context):
		ctm_args= opt_context["ctm_args"]
		opt_args= opt_context["opt_args"]
		# build on-site tensors from su2sym components
		state.sites= state.build_onsite_tensors()
		# possibly re-initialize the environment
		#if opt_args.opt_ctm_reinit:
		#	init_env(state, ctm_env_in)
		# compute environment by CTMRG
		ctm_env_out, history, t_ctm, t_obs= ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=ctm_args)
		loss = energy_f(state, ctm_env_out)
		timings = (t_ctm, t_obs)
		return loss, ctm_env_out, history, timings
	
	optimize_state(state, ctm_env, loss_fn)
	e_dn_final = model.energy_triangle(state,ctm_env)
	e_up_final = model.energy_triangle_up(state,ctm_env)
	e_tot_final = (e_dn_final + e_up_final)/2
	print(f'\n\n E_up={e_up_final.item()}, E_dn={e_dn_final.item()}, E_tot={e_tot_final.item()}')
	print(e_tot_final.item())
	
	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()

