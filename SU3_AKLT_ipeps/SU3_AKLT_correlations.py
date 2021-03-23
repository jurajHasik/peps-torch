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
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-','serif':['Palatino']})
rc('text', usetex=True)
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)
rc('axes', labelsize=6)

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
	
	max_distance = 20
	
	corrf_L3, corrf_L8 = model.eval_corrf_LL((1,0),state,ctm_env_fin,dist=max_distance)
	corrf_PP = model.eval_corrf_PP((1,0),state,ctm_env_fin,dist=max_distance)
	corrf_L3, corrf_L8 = -np.array(corrf_L3), -np.array(corrf_L8)
	
	plt.figure(1, figsize=(2.5,2.))
	array_r = np.arange(1,max_distance+2,1)
	plt.plot(array_r, np.log(np.abs(corrf_L3)), '.',color='b')
	#plt.plot(array_r, corrf_L8, '.')
	p1, p0 = np.polyfit(array_r[2:20],np.log(np.abs(corrf_L3))[2:20],1)
	print('<LL> correlation length = '+str(-1/p1))
	plt.plot(array_r, p0+p1*array_r, color='b',linewidth=0.7)
	plt.xlabel(r'$r$')
	plt.ylabel(r'$\log | \langle \lambda_3(0) \lambda_3(r) \rangle |$')
	plt.xscale('linear')
	plt.yscale('linear')
	
	plt.tight_layout()
	plt.show()
	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()
