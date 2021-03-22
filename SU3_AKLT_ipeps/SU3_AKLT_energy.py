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
	torch.set_num_threads(args.omp_cores)
	torch.manual_seed(args.seed)
	def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
		with torch.no_grad():
			if not history:
				history=[]
			e_curr = model.energy_triangle(state, env)
			print('Step n°'+str(len(history)+1)+'   E = '+str(e_curr.item()))
			history.append([e_curr.item()])
			if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
				return True, history
		return False, history
		
	# initializes an environment for the ipeps
	ctm_env_init = ENV(args.chi, state)
	init_env(state, ctm_env_init)
	# initial energy
	e_init_dn = model.energy_triangle(state, ctm_env_init)
	print('*** Energy per site (before CTMRG) -- down triangles: '+str(e_init_dn.item()))
	
	# performs CTMRG and computes the obervables afterwards (energy and lambda_3)
	ctm_env_fin, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
	
	e_final_dn = model.energy_triangle(state, ctm_env_fin)
	print('*** Energy per site (after CTMRG) -- down triangles: '+str(e_final_dn.item()))
	
	e_final_up = model.energy_triangle_up(state, ctm_env_fin)
	print('*** Energy per site (after CTMRG) -- up triangles: '+str(e_final_up.item()))
	
	colors3 = model.eval_color3(state,ctm_env_fin)
	print('*** <Lambda_3> for sites 1,2,3 (after CTMRG): '+str(colors3[0].item())+', '+str(colors3[1].item())+', '+str(colors3[2].item()))

	
if __name__=='__main__':
	if len(unknown_args)>0:
		print("args not recognized: "+str(unknown_args))
		raise Exception("Unknown command line arguments")
	main()
