"""Optimizer for one step of Trotter-Suzuki decomposition. The optimizer
maximizes the ratio of overlaps w.r.t the coefficients of the basic Cs tensors
and updates them."""

import torch
from tqdm import tqdm  # progress bars
import tensors.base_tensors.base_tensor as bt
import tensors.onsite as ons
import optim.ts_lbfgs as ts
import output.observables as obs
import output.read_output as ro
# peps-torch imports
import config as cfg
from ipeps.ipeps_c4v import read_ipeps_c4v
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v.env_c4v import ENV_C4V, init_env
import logging
log = logging.getLogger(__name__)

############################ Initialization ##################################
# Get parser from config
parser = cfg.get_args_parser()
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--n", type=int, default=40, help="max number of optimization steps")
parser.add_argument("--it", type=int, default=10, help="imaginary time steps")
parser.add_argument("--tau", type=float, default=1/8, help=r"\tau = \frac{\beta}{N}")
parser.add_argument("--t", type=float, default=1e-4, help="threshold of optimizer")
parser.add_argument("--no", type=float, default=1e-2, help="noise to add to the coefficients")
parser.add_argument("--lr", type=float, default=1, help="learning rate")
parser.add_argument("--p", type=int, default=3, help="patience of the convergence")
parser.add_argument("--sf", type=str, default='output/obs/output_coeff', help="save file")
parser.add_argument("--out", type=str, default='output/obs/output_observables.txt', help="save file")
args, unknown_args = parser.parse_known_args()

# Create dictionary containing all the tensors 
base_tensor_dict = bt.base_tensor_dict(args.bond_dim)

# Create dictionary of the parameters
params_j1 = {'a': {'permutation': (0,1,2,3,4), 'new_symmetry' : 'Cx'},
             'b': {'permutation': (0,3,4,1,2), 'new_symmetry' : 'Cx'},
             'c': {'permutation': (0,2,3,4,1), 'new_symmetry' : ''},
             'd': {'permutation': (0,4,1,2,3), 'new_symmetry' : 'C4v'}}

params_j2 = {'a': {'permutation': (0,1,2,3,4), 'new_symmetry' : 'Cs', 'diag': 'diag'},
             'b': {'permutation': (0,1,2,3,4), 'new_symmetry' : 'Cs', 'diag' : 'diag'},
             'c': {'permutation': (0,1,2,3,4), 'new_symmetry' : '', 'diag' : 'off'},
             'd': {'permutation': (0,1,2,3,4), 'new_symmetry' : 'C4v', 'diag' : 'off'}}

coeff_ini = {'4': [0.,0.,0.,0.,1.,0.,0.,0.],
             '7': [0.,0.,0.,0.,1.]+[0.]*44}

params_onsite = {'symmetry':'C4v', 'coeff': coeff_ini[f'{args.bond_dim}'],
                 'base_tensor_dict': base_tensor_dict,  
                 'file': "tensors/input-states/init_tensor.json",
                 'bond_dim': args.bond_dim, 'dtype':torch.float64}

################################ Main ########################################
def main():
    ### Initialization ###
    # Parse command line arguments and configure simulation parameters
    cfg.configure(args)
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    # Define convergence criterion on the 2 sites reduced density matrix
    def ctmrg_conv_rdm2x1(state_ini, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history = dict({"log": []})
            rdm2x1 = rdm2x1_sl(state_ini, env, force_cpu=ctm_args.conv_check_cpu)
            dist = float('inf')
            if len(history["log"]) > 1:
                dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
            # update history
            history["rdm"] = rdm2x1
            history["log"].append(dist)
        return False, history

    # Initialize parameters for J1 and J2 term
    gate = ts.build_gate(args.j1, args.tau)
    gate2 = ts.build_gate(args.j2, args.tau)
    # Initialize onsite tensor
    onsite1 = ons.OnSiteTensor(params_onsite)
    model = obs.J1J2_ipepo(j1=args.j1, j2=args.j2, tau=args.tau, file=args.out)
    
    for step in tqdm(range(args.it)):
        
        # Read IPEPS from .json file and update environment
        state_env = read_ipeps_c4v("tensors/input-states/init_tensor.json")
        ctm_env = ENV_C4V(args.chi, state_env)
        init_env(state_env, ctm_env)
        ctm_env = ctmrg_c4v.run(state_env, ctm_env, conv_check=ctmrg_conv_rdm2x1)[0]
        
        if args.j1 != 0:
            # Apply j1 gate
            for bond_type in ['a','b','c','d']:
                new_symmetry = params_j1[bond_type]['new_symmetry']
                permutation = params_j1[bond_type]['permutation']
                
                # Optimize
                onsite1 = ts.optimization_2sites(onsite1=onsite1, new_symmetry=new_symmetry,
                            permutation=permutation, env=ctm_env, gate=gate,
                            const_w2=ts.const_w2_2sites, cost_function=ts.cost_function_2sites,
                            noise=args.no, max_iter=args.n, threshold=args.t, patience=args.p,
                            optimizer_class=torch.optim.LBFGS, lr=args.lr)

        if args.j2 != 0:
            # Apply j2 gate
            for bond_type in ['a','b','c','d']:
                new_symmetry = params_j2[bond_type]['new_symmetry']
                permutation = params_j2[bond_type]['permutation']
                diag = params_j2[bond_type]['diag']
                def const_w2(tensor, env, gate):
                    return ts.const_w2_NNN_plaquette(tensor, onsite1.site(), diag, env, gate)
                def cost_function(tensor1, tensor2, env, gate, w2):
                    return ts.cost_function_NNN_plaquette(tensor1, tensor2, onsite1.site(), diag, env, gate, w2)
                
                # Optimize
                onsite1 = ts.optimization_2sites(onsite1=onsite1, new_symmetry=new_symmetry,
                            permutation=permutation, env=ctm_env, gate=gate2,
                            const_w2=const_w2, cost_function=cost_function,
                            noise=args.no, max_iter=args.n, threshold=args.t, patience=args.p,
                            optimizer_class=torch.optim.LBFGS, lr=args.lr)
        
        # Save coefficients
        onsite1.write_to_json("tensors/input-states/init_tensor.json")
        onsite1.history(); model.save_obs(onsite1.site(), ctm_env, step)
        
    onsite1.save_coeff_to_bin(args.sf+'_bin')
    
if __name__ == '__main__':
    main()
    if args.j2!=0:
        ro.plot_j2(args.out, args.j1, args.j2, args.tau)
    else:
        ro.plot_j1(args.out, args.j1, args.tau)