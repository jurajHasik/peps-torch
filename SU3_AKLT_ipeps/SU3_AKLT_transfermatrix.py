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
				print('*** Done chi = '+str(vchi))
				return True, history
		return False, history
		
	lchi = np.array([10,20,35,50])
	lenergy = []
	lxi_1 = []
	lxi_2 = []
	lxi_3 = []
	lxi_4 = []
	lxi_5 = []
	lxi_6 = []
	lxi_7 = []
	lxi_8 = []
	lxi_9 = []
	lxi_10 = []
	lxi_11 = []
	lxi_s = []
	lxi_d = []	
		
	for vchi in lchi:	
		
		# run the CTMRG algorithm and compute the environment
		ctm_env_init = ENV(vchi, state)
		init_env(state, ctm_env_init)
		ctm_env_fin, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_rdm)
		
		# compute the transfer matrix spectrum and associated correlation length
		T_spec = transferops.get_Top_spec(args.top_n,(0,0),(1,0),state, ctm_env_fin)
		xi_1 = -1/np.log(np.linalg.norm(T_spec[1]))
		xi_2 = -1/np.log(np.linalg.norm(T_spec[9]))
		xi_3 = -1/np.log(np.linalg.norm(T_spec[17]))
		xi_4 = -1/np.log(np.linalg.norm(T_spec[25]))
		xi_5 = -1/np.log(np.linalg.norm(T_spec[33]))
		xi_6 = -1/np.log(np.linalg.norm(T_spec[35]))
		xi_7 = -1/np.log(np.linalg.norm(T_spec[43]))
		xi_8 = -1/np.log(np.linalg.norm(T_spec[63]))
		xi_9 = -1/np.log(np.linalg.norm(T_spec[83]))
		xi_10 = -1/np.log(np.linalg.norm(T_spec[99]))
		
		lxi_1.append(xi_1)
		lxi_2.append(xi_2)
		lxi_3.append(xi_3)
		lxi_4.append(xi_4)
		lxi_5.append(xi_5)
		lxi_6.append(xi_6)
		lxi_7.append(xi_7)
		lxi_8.append(xi_8)
		lxi_9.append(xi_9)
		lxi_10.append(xi_10)
		
	print(lxi_1[-1])
	# plot xi_1, xi_2, etc... = f(1/chi)
	plt.figure(2, figsize=(2.5,2.))
	plt.grid()
	plt.plot(lchi, lxi_1, '*-', color='k',label=r'$\xi_1$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_2, '*-', color='k',label=r'$\xi_2$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_3, '*-', color='k',label=r'$\xi_3$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_4, '*-', color='k',label=r'$\xi_4$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_5, '*-', color='k',label=r'$\xi_5$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_6, '*-', color='k',label=r'$\xi_6$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_7, '*-', color='k',label=r'$\xi_5$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_8, '*-', color='k',label=r'$\xi_6$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_9, '*-', color='k',label=r'$\xi_5$',linewidth=0.7, markersize=2.5)
	plt.plot(lchi, lxi_10,'*-', color='k',label=r'$\xi_5$',linewidth=0.7, markersize=2.5)
	plt.xlabel(r'$\chi$')
	plt.xscale('linear')
	plt.yscale('linear')
	plt.show()
	
	
	
	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()
