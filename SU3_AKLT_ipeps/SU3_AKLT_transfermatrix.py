import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models.SU3_AKLT import *
import unittest
import numpy as np

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()


model = SU3_AKLT()
def lattice_to_site(coord): return (0,0)
state = read_ipeps('SU3_AKLT_ipeps/SU3_AKLT_ipeps.json', vertexToSite=lattice_to_site)

def main():

	cfg.configure(args)
	cfg.print_config()
	print('\n')
	torch.set_num_threads(args.omp_cores)
	torch.manual_seed(args.seed)
	
	def ctmrg_conv_rdm(state, env, history, ctm_args=cfg.ctm_args):
		"""
		Convergence is defined wrt. the reduced density matrix. 
		We define the following quantity at step i:
			Delta_rho[i] = Norm( rho[i+1] - rho[i] )
		and the algoritm stops when Delta_rho < tolerance.
		"""
		with torch.no_grad():
			if not history:
				history=[]
			e_curr = model.energy_triangle(state, env)
			rdm_curr = model.rdm1x1(state,env)
			history.append([e_curr.item(), rdm_curr])
			if len(history) <= 1: 
				Delta_rho = 'not defined'
			else:
				Delta_rho = torch.norm(history[-1][1]-history[-2][1]).item()
			print('Step n°'+str(len(history))+'     E_down = '+str(e_curr.item())+'     Delta_rho = '+str(Delta_rho))
			if len(history) > 1 and Delta_rho < ctm_args.ctm_conv_tol:
				return True, history
		return False, history
		
	# initializes an environment for the ipeps
	ctm_env_init = ENV(args.chi, state)
	init_env(state, ctm_env_init)	
	# performs CTMRG and computes the obervables afterwards (energy and lambda_3)
	ctm_env_fin, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_rdm)
	
	indices = torch.tensor(range(args.top_n))
	TM_spectrum = transferops.get_Top_spec(args.top_n, (0,0), (1,0), state, ctm_env_fin, verbosity=0)
	indices_TM_spectrum = torch.tensor([list(indices), list(torch.transpose(TM_spectrum, 0, 1)[0]), list(torch.transpose(TM_spectrum, 0, 1)[1]), [0] + [1./np.log(np.sqrt(TM_spectrum[jj,0]**2 + TM_spectrum[jj,1]**2)) for jj in range(1,args.top_n)]])
	indices_TM_spectrum = torch.transpose(indices_TM_spectrum, 0, 1)
	print(indices_TM_spectrum)
	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()
