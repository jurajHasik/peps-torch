import torch
import math
import numpy as np
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
    for name in ['S0','S1','S2','S3','S4','S5','S6','L0','L1','L2']:
        ts = load_SU3_tensor(name)
        elementary_tensors.append(ts)
    # define initial coefficients
    coeffs = {(0,0): torch.tensor([1.,0.,0.,0.,0.,0.,0.,1.,0.,0.],dtype=torch.float64)}
    # define which coefficients will be added a noise
    var_coeffs_allowed = torch.tensor([1,1,1,1,1,0,0, 1,1,1])
    state = IPEPS_U1SYM(elementary_tensors, coeffs, var_coeffs_allowed)
    state.add_noise(args.instate_noise)
    print(f'Current state: {state.coeffs[(0,0)].data}')
    
    model = SU3_chiral.SU3_CHIRAL(theta = args.theta)
    
    def energy_f(state, env):
        e_dn = model.energy_triangle(state,env)
        e_up = model.energy_triangle_up(state,env)
        return((e_up+e_dn)/2)
    
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state,env)
        history.append(e_curr.item())
        print('Step n°'+str(len(history))+'     E_site = '+str(e_curr.item()))
        
        for c_loc,c_ten in env.C.items():
            u,s,v= torch.svd(c_ten, compute_uv=False)
            print(f"\n\nspectrum C[{c_loc}]")
            for i in range(args.chi):
                print(f"{i} {s[i]}")

	ctm_env_init = ENV(args.chi, state)
	init_env(state, ctm_env_init)
	
	e_dn_init = energy_f(state, ctm_env_init)
	print('*** Energy per site (before CTMRG) -- down triangles: '+str(e_dn_init.item()))
	e_up_init = model.energy_triangle_up(state, ctm_env_init)
	print('*** Energy per site (before CTMRG) -- up triangles: '+str(e_up_init.item()))
	
	ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
	
	e_dn_final = energy_f(state,ctm_env_out)
	e_up_final = model.energy_triangle_up(state, ctm_env_out)
	colors3, colors8 = model.eval_lambdas(state,ctm_env_out)
	print('*** Energy per site (after CTMRG) -- down triangles: '+str(e_dn_final.item()))
	print('*** Energy per site (after CTMRG) -- up triangles: '+str(e_up_final.item()))
	print('*** <Lambda_3> (after CTMRG): '+str(colors3[0].item())+', '+str(colors3[1].item())+', '+str(colors3[2].item()))
	print('*** <Lambda_8> (after CTMRG): '+str(colors8[0].item())+', '+str(colors8[1].item())+', '+str(colors8[2].item()))

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    
    e_dn_init = model.energy_triangle(state, ctm_env_init)
    print('*** Energy per site (before CTMRG) -- down triangles: '+str(e_dn_init.item()))
    e_up_init = model.energy_triangle_up(state, ctm_env_init)
    print('*** Energy per site (before CTMRG) -- up triangles: '+str(e_up_init.item()))
    print(f'*** Energy per site (before CTMRG) -- total: {(e_up_init.item()+e_dn_init.item())/2}')
    
    ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
    
    e_dn_final = model.energy_triangle(state,ctm_env_out)
    e_up_final = model.energy_triangle_up(state, ctm_env_out)
    colors3, colors8 = model.eval_lambdas(state,ctm_env_out)
    print('*** Energy per site (after CTMRG) -- down triangles: '+str(e_dn_final.item()))
    print('*** Energy per site (after CTMRG) -- up triangles: '+str(e_up_final.item()))
    print('*** <Lambda_3> (after CTMRG): '+str(colors3[0].item())+', '+str(colors3[1].item())+', '+str(colors3[2].item()))
    print('*** <Lambda_8> (after CTMRG): '+str(colors8[0].item())+', '+str(colors8[1].item())+', '+str(colors8[2].item()))

	# environment diagnostics
	print("\n")
	print("Final environment")
	for c_loc,c_ten in ctm_env_out.C.items(): 
		u,s,v= torch.svd(c_ten, compute_uv=False)
		print(f"spectrum C[{c_loc}]")
		for i in range(args.chi):
			print(f"{i} {s[i]}")
	print("\n")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

