"""Optimizer for one step of Trotter-Suzuki decomposition. The optimizer
maximizes the ratio of overlaps w.r.t the coefficients of the basic Cs tensors
and updates them."""

import torch
import pickle
import numpy as np
from tqdm import tqdm  # progress bars
import tensors.tensor_sum as ts
# peps-torch imports
import groups.su2 as su2
import models.j1j2 as j1j2
import config as cfg
from ipeps.ipeps_c4v import read_ipeps_c4v
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v.env_c4v import ENV_C4V, init_env
import logging
log = logging.getLogger(__name__)


# Get parser from config
parser = cfg.get_args_parser()
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--n", type=int, default=40, help="max number of optimization steps")
parser.add_argument("--it", type=int, default=10, help="imaginary time steps")
parser.add_argument("--t", type=float, default=1e-4, help="threshold of optimizer")
parser.add_argument("--e", type=float, default=1e-1, help="noise to add to the coefficients")
parser.add_argument("--lr", type=float, default=1, help="learning rate")
parser.add_argument("--p", type=int, default=3, help="patience of the convergence")
parser.add_argument("--title", type=str, default='', help="title of the output files")
args, unknown_args = parser.parse_known_args()


def overlap(w0, w1, w2):
    return w1/torch.sqrt(w0*w2)


def build_gate(tau=1/8, H=j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)):
    """Part of j1j2.py code.
    H is a class with j1=1.0 and j2=0 by default.
    Simple pi/2 rotation of the gate gives the Hamiltonian on another bond."""
    s2 = su2.SU2(H.phys_dim, dtype=H.dtype, device=H.device)
    expr_kron = 'ij,ab->iajb'

    # Spin-spin operator
    #   s1|   |s2
    #     |   |
    #   [  S.S  ]
    #     |   |
    #  s1'|   |s2'
    
    # S_1 * S_2 but with S_2 rotated of pi to respect the metric
    SS = - torch.einsum(expr_kron, s2.SZ(), s2.SZ())\
        - 0.5*(torch.einsum(expr_kron, s2.SP(), s2.SP())
               + torch.einsum(expr_kron, s2.SM(), s2.SM()))
    SS = SS.view(4,4).contiguous()

    # Diagonalization of SS and creation of Hamiltonian Ha
    eig_va, eig_vec = np.linalg.eigh(SS)
    eig_va = np.exp(-tau*args.j1*eig_va)
    U = torch.tensor(eig_vec)
    D = torch.diag(torch.tensor(eig_va))
    
    # SS = U D U.T
    Ga = torch.einsum('ij,jk,lk->il', U, D, U)
    Ga = torch.eye(4,4) - tau*SS
    Ga = Ga.view(2,2,2,2).contiguous()
    return Ga


def compute_energy(tensor, env, bond_type='a',
                   H=j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)):
    """Compute the bond energy"""
    s2 = su2.SU2(H.phys_dim, dtype=H.dtype, device=H.device)
    expr_kron = 'ij,ab->iajb'

    # Spin-spin operator
    #   s1|   |s2
    #     |   |
    #   [  S.S  ]
    #     |   |
    #  s1'|   |s2'
    
    # S_1 * S_2 but with S_2 rotated of pi to respect the metric
    SS = - torch.einsum(expr_kron, s2.SZ(), s2.SZ())\
        - 0.5*(torch.einsum(expr_kron, s2.SP(), s2.SP())
               + torch.einsum(expr_kron, s2.SM(), s2.SM()))
    
    rdm = rdm2x1_sl(state=(tensor, tensor, bond_type), env=env)
    rdm = rdm.view((*[2]*8)).contiguous()
    rdm = torch.einsum('abcdaecf', rdm)
    w0 = torch.einsum('abab', rdm)
    energy = torch.einsum('abcd, cdab', rdm, SS)
    return 2*args.j1*energy/w0

    
def compute_w0(B_tensor, env, bond_type):
    """
    Be |phi'> the purified PEPS associated with A' tensor.
    Compute the overlap <phi'|phi'>.

    Parameters
    ----------
    B_tensor : torch.tensor(4,4,4,4,4)

    Returns
    -------
    w0 : float
    """
    rdm = rdm2x1_sl(state=(B_tensor, B_tensor, bond_type), env=env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdaecf', rdm)
    w0 = torch.einsum('abab', rdm)
    return w0


def compute_w1(A_tensor, B_tensor, gate, env, bond_type):
    """
    Compute the overlap <phi|G|phi'>.

    Parameters
    ----------
    A_tensor, B_tensor : torch.tensor(4,4,4,4,4)

    gate : torch.tensor

    Returns
    -------
    w1 : float
    """
    rdm = rdm2x1_sl(state=(A_tensor, B_tensor, bond_type), env=env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdaecf', rdm)
    w1 = torch.einsum('abcd, cdab', rdm, gate)
    return w1


def compute_w2(A_tensor, gate, env, bond_type):
    """
    Be |phi> the purified PEPS associated with A tensor.
    Compute the overlap <phi|G*G|phi>.

    Parameters
    ----------
    A_tensor : torch.tensor(4,4,4,4,4)

    gate : torch.tensor

    Returns
    -------
    w2 : float
    """
    rdm = rdm2x1_sl(state=(A_tensor, A_tensor, bond_type), env=env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdaecf', rdm)
    rdm = torch.einsum('cdij,ijkl->cdkl', rdm, gate)
    w2 = torch.einsum('abcd, cdab', rdm, gate)
    return w2


def run_optimization(init_tensor, init_coef, bond_type, gate, env,
                     max_iter, threshold,
                     optimizer_class, **optimizer_kwargs):
    """
    Run optimization to find the maximum of O = \frac{w0}{\sqrt{w1*w2}}. At
    each step of the optimization, 3 overlaps w0, w1 and w2 are computed using
    CTMRG. Then O is computed and Tb_coef is updated.

    Parameters
    ----------
    init_tensor : torch.tensor(4,4,4,4,4)
        Initial tensor. Can be either C4v or Cx symmetric.
        
    bond_type: str
        Type of the tensor given the bond where the gate is applied.
        Can be a, b, c, d.

    coef_to_optim : list of floats
        Coefficients we want to optimize. The size of the list corresponds to
        the number of tensors in the class of symmetry.

    gate : torch.tensor
        G is the TS IT operator.
        
    env : ctmrg_c4v class
        The environment of the active tensors.

    optimizer_class : object
        Optimizer class.

    max_iter : int
        Max number of iterations of the optimization.
        
    threshold : int
        Threshold to the optimization convergence.

    optimizer_kwargs : dict
        Additional parameters to be passed to the optimizer.

    Returns
    -------
    coef_opti : np.ndarray
        2D array of shape (max_iter, len(Tb_coef)). Where the rows represent the
        iteration and the columns represent the updated list.
    """
    # Normalize initial tensor
    init_coef = init_coef/np.max(np.abs(init_coef))
    
    # Selected basic tensors
    if bond_type == 'a' or bond_type =='b':
        basic_tensors_list = ts.contract_B_cx()
        coef_to_optim = ts.Cx(init_coef, epsilon=args.e)
    if bond_type == 'c':
        basic_tensors_list = ts.contract_D_all()
        coef_to_optim = ts.C(init_coef, epsilon=args.e)
    if bond_type == 'd':
        basic_tensors_list = ts.contract_A_c4v()
        coef_to_optim = ts.C4v(init_coef, epsilon=args.e)

    
    coef_to_optim_t = torch.tensor(coef_to_optim, dtype=torch.float64, requires_grad=True)
    optimizer = optimizer_class([coef_to_optim_t], max_iter=max_iter, **optimizer_kwargs)

    coef_optimized = np.zeros((1, len(coef_to_optim)))
    coef_optimized[0,:] = coef_to_optim

        
    # criterion for convergence of the L-BFGS optimizer
    n_bad_steps = 0
    best_loss = float('inf')
    threshold = threshold
    patience = args.p
    
    def closure():
        optimizer.zero_grad()
        final_tensor = ts.build_tensor(coef_to_optim_t, basic_tensors_list)
        w0 = compute_w0(final_tensor, env, bond_type)
        w1 = compute_w1(init_tensor, final_tensor, gate, env, bond_type)
        w2 = compute_w2(init_tensor, gate, env, bond_type)
        loss = -overlap(w0, w1, w2)
    
        # Compute gradient
        loss.backward()
    
        # Clip norm gradients to 1.0 to garantee they are not exploding
        torch.nn.utils.clip_grad_norm_(coef_to_optim_t, 1.0)
    
        return loss

    for i in range(1,max_iter):        
        
        # Update value of the coefficients
        loss_res = optimizer.step(closure)

        if abs(loss_res.item() - best_loss) > threshold:
                best_loss = loss_res.item()
                n_bad_steps = 0
        else:
                n_bad_steps += 1
        if n_bad_steps > patience:
                break

        coef_optimized = np.vstack([coef_optimized, coef_to_optim_t.detach().numpy()])
               
    # Build initial tensor of next step
    final_tensor = ts.build_tensor(coef_optimized[-1,:], basic_tensors_list)
    
    return coef_optimized[-1,:], final_tensor


def main():
    ## INITIALIZATION
    # Parse command line arguments and configure simulation parameters
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

    # Initialize parameters
    gate = build_gate()
    init_coef = [*[0]*4,1,*[0]*3]
    init_tensor = ts.build_A_tensor([*[0]*4,1,*[0]*3])
    coef_array = np.zeros((args.it+1, 8))
    coef_array[0] = init_coef
    energy_array = np.zeros((args.it+1, 1))
    energy_array[0] = 0
    
    ## OPTIMIZATION IN IMAGINARY TIME
    # Run optimization route: C4v -> Cx -> Cx -> all -> C4v
    for step in tqdm(range(args.it)):
        
        # Read IPEPS from .json file and update environment
        state = read_ipeps_c4v("tensors/input-states/A_tensor.json")
        ctm_env = ENV_C4V(args.chi, state)
        init_env(state, ctm_env)
        ctm_env = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_rdm2x1)[0]
        
        for bond_type in ['a','b','c','d']:
            init_coef, init_tensor = run_optimization(init_tensor=init_tensor, 
                        init_coef=init_coef, bond_type=bond_type, gate=gate,
                        env=ctm_env, max_iter=args.n, threshold = args.t,
                        optimizer_class=torch.optim.LBFGS, lr=args.lr)
            
        # update C4v tensor
        coef_array[step+1] = init_coef
        tensor = ts.build_A_tensor(init_coef)
        energy_array[step+1] = compute_energy(tensor=tensor, env=ctm_env)

    file = open("output/output_coef"+args.title, "wb")
    pickler = pickle.Pickler(file)
    pickler.dump(coef_array)
    file.close
    file2 = open("output/output_energy"+args.title, "wb")
    pickler = pickle.Pickler(file2)
    pickler.dump(energy_array)
    file.close

if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    ts.build_Ta_tensors()
    main()
