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
    coeffs = {(0,0): torch.tensor([1.,1.,1.,1.,1., 0.,0., 1.,1.,1.],dtype=torch.float64)}
    # define which coefficients will be added a noise
    var_coeffs_allowed = torch.tensor([1,1,1,1,1, 1,1, 1,1,1])
    state = IPEPS_U1SYM(elementary_tensors, coeffs, var_coeffs_allowed)
    state.add_noise(args.instate_noise)
    print(f'Current state: {state.coeffs[(0,0)].data}')
    
    model = SU3_chiral.SU3_CHIRAL(theta = args.theta)
        
    def energy_f(state, env):
        e_dn = model.energy_triangle_dn(state,env)
        e_up = model.energy_triangle_up(state,env)
        #print(f'Energy per site: E_up={e_up.item()*1/3}, E_dn={e_dn.item()*1/3}')
        return((e_up+e_dn)/3)
    
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state,env)
        history.append(e_curr.item())
        print('Step n°'+str(len(history))+'     E_site = '+str(e_curr.item()))
        
        #for c_loc,c_ten in env.C.items():
        #    u,s,v= torch.svd(c_ten, compute_uv=False)
        #    print(f"\n\nspectrum C[{c_loc}]")
        #    for i in range(args.chi):
        #        print(f"{i} {s[i]}")

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history
        
        
    def ctmrg_conv_corners(state,env,history,ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        S_list = []
        for c_loc,c_ten in env.C.items(): 
            u,s,v= torch.svd(c_ten, compute_uv=False)
            S_list.append(s.tolist())
        S_list = np.array(S_list)
        history.append(S_list)
        if len(history) <= 1: 
            Delta_C = 'not defined'
        else:
            Delta_C = np.linalg.norm(history[-1]-history[-2]).item()
        print('Step n°'+str(len(history))+'     Delta_C = '+str(Delta_C)) #+'     E_down = '+str(e_curr.item())
        if len(history) > 1 and Delta_C < ctm_args.ctm_conv_tol:
            return True, history
        return False, history
        

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    ctm_env_final, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)
    
    # energy per site
    e_dn_final = model.energy_triangle_dn(state,ctm_env_final) /3.
    e_up_final = model.energy_triangle_up(state,ctm_env_final) /3.
    e_tot_final = e_dn_final + e_up_final
    
    # P operators
    P_up = model.P_up(state,ctm_env_final)
    P_dn = model.P_dn(state,ctm_env_final)
    
    print(f'\n\n E_up={e_up_final.item()}, E_dn={e_dn_final.item()}, E_tot={e_tot_final.item()}')
    print(f' Re(P_up)={torch.real(P_up).item()}, Im(P_up)={torch.imag(P_up).item()}')
    print(f' Re(P_dn)={torch.real(P_dn).item()}, Im(P_dn)={torch.imag(P_dn).item()}')
    
    colors3, colors8 = model.eval_lambdas(state,ctm_env_final)
    print(f' <Lambda_3> = {torch.real(colors3[0]).item()}, {torch.real(colors3[1]).item()}, {torch.real(colors3[2]).item()}')
    print(f' <Lambda_8> = {torch.real(colors8[0]).item()}, {torch.real(colors8[1]).item()}, {torch.real(colors8[2]).item()}')


    # environment diagnostics
    print("\n")
    print("Final environment")
    for c_loc,c_ten in ctm_env_final.C.items(): 
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

