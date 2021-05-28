""" Estimate the observables by reading the optimized coefficients and computing
the environment for each step of imaginary time. """

import torch
import pickle
import numpy as np
from tqdm import tqdm  # progress bars
# peps-torch imports
import tensors.onsite as ons
import tensors.base_tensors.base_tensor as bt
import optim.ts_lbfgs as ts
import groups.su2 as su2
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
parser.add_argument("--bd", type=int, default=4, help="bond dimension")
parser.add_argument("--f", type=str, default="output/obs/output_coeff_bin", help="name of the file")
args, unknown_args = parser.parse_known_args()

# Create dictionary containing all the tensors 
base_tensor_dict = bt.base_tensor_dict(args.bd)

################################ Main ########################################
# Class of the observables
class J1J2_ipepo():
    def __init__(self, j1=1.0, j2=0.0, tau=1/8, global_args=cfg.global_args):
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.tau = tau
        self.j1=j1
        self.j2=j2
        self.s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        
    def energy(self, tensor, env):
        expr_kron = 'ij,ab->iajb'
        # S_1 * S_2 but with S_2 rotated of pi to respect the metric
        SS = - torch.einsum(expr_kron, self.s2.SZ(), self.s2.SZ())\
            - 0.5*(torch.einsum(expr_kron, self.s2.SP(), self.s2.SP())
            + torch.einsum(expr_kron, self.s2.SM(), self.s2.SM()))
        rdm = ts.rdm2x1_sl_2sites(tensor, tensor, env)
        w0 = torch.einsum('abab', rdm)
        energy = torch.einsum('abcd, cdab', rdm, SS)
        return 2*self.j1*energy/w0
        
    def magnetization(self, tensor, env):
        Sz = self.s2.SZ()
        Sx = 0.5*(self.s2.SP() + self.s2.SM())
        Sy = 0.5j*(self.s2.SP() - self.s2.SM())
        rdm = ts.rdm2x1_sl_2sites(tensor, tensor, env)
        rdm1x1 = torch.einsum('abcb', rdm)
        w0 = torch.einsum('aa', rdm1x1)
        zmag = torch.einsum('ij,ji', rdm1x1, Sz)
        xmag = torch.einsum('ij,ji', rdm1x1, Sx)
        rdm1x1 = rdm1x1.type(torch.cdouble)
        ymag = torch.einsum('ij,ji', rdm1x1, Sy)
        return (xmag/w0, torch.abs(ymag)/w0, zmag/w0)


def main():
    cfg.configure(args)
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    # Convergence criterion
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

    # Initialize coefficients list
    if args.f.split('_')[-1] == 'bin':
        depickler = pickle.Unpickler(open(args.f, "rb"))
        coeff = depickler.load()
    else:
        coeff = list()
        with open(args.f, 'r') as fp:
            for line in fp.readlines():
                coeff_list = []
                for words in line.strip('\n').split(' '):
                    coeff_list.append(float(words))
                coeff.append(coeff_list)
    
    n_iter = len(coeff)
    results = {'energy': np.zeros(n_iter), 'magnetization': np.zeros((n_iter, 3))}
    model = J1J2_ipepo(j1=args.j1, j2=args.j2)
    
    for step in tqdm(range(n_iter)):
        onsite = ons.OnSiteTensor('C4v', coeff[step], base_tensor_dict=base_tensor_dict,  
                       file="tensors/input-states/init_tensor.json",
                       bond_dim=args.bd)
        onsite.write_to_json("tensors/input-states/init_tensor.json")
        # Read IPEPS from .json file and update environment
        state = read_ipeps_c4v("tensors/input-states/init_tensor.json")
        ctm_env = ENV_C4V(args.chi, state)
        init_env(state, ctm_env)
        ctm_env = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_rdm2x1)[0]

        results['energy'][step] = model.energy(tensor=onsite.site(), env=ctm_env)
        results['magnetization'][step] = model.magnetization(tensor=onsite.site(), env=ctm_env)
        
    file = open("output/obs/output_res", "wb")
    pickler = pickle.Pickler(file)
    pickler.dump(results)
    file.close
    
if __name__ == '__main__':
    main()