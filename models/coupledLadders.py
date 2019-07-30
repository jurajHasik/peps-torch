import torch
import su2
import ipeps
from ctm.generic.env import ENV
from ctm.generic import rdm
from args import GLOBALARGS
from math import sqrt
import itertools

class COUPLEDLADDERS():
    def __init__(self, alpha=0.0, global_args=GLOBALARGS()):
        r"""
        :param alpha: nearest-neighbour interaction
        :param global_args: global configuration
        :type alpha: float
        :type global_args: GLOBALARGS

        Build Hamiltonian of spin-1/2 coupled ladders

        .. math:: H = \sum_{i=(x,y)} h2_{i,i+\vec{x}} + \sum_{i=(x,2y)} h2_{i,i+\vec{y}}
                   + \alpha \sum_{i=(x,2y+1)} h2_{i,i+\vec{y}}

        on the square lattice. The spin-1/2 ladders are coupled with strength :math:`\alpha`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._a__a__a__a_...
            ..._|__|__|__|_...
            ..._a__a__a__a_...   
            ..._|__|__|__|_...   
                :  :  :  :      (a = \alpha) 

        where

        * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.alpha=alpha

        self.h2 = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        return SS

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1c4v(self,ipeps):
        pass

    def energy_2x2(self,ipeps):
        pass

    # assuming reduced density matrix of 2x2 cluster with indexing of DOFs
    # as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
    def energy_2x2(self,rdm2x2):
        energy = torch.einsum('ijklabkl,ijab',rdm2x2,self.h)
        return energy

    def energy_2x1_1x2(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float
        
        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D

        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        and without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::

               0       0       0
            1--A--3 1--B--3 1--A--3
               2       2       2
               0       0       0
            1--C--3 1--D--3 1--C--3
               2       2       2             A--3 1--B,      A  B  C  D
               0       0                     B--3 1--A,      2  2  2  2
            1--A--3 1--B--3                  C--3 1--D,      0  0  0  0
               2       2             , terms D--3 1--C, and  C, D, A, B
        """
        energy=0.
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            ss = torch.einsum('ijab,ijab',rdm2x1,self.h2)
            energy += ss
            if coord[1] % 2 == 0:
                ss = torch.einsum('ijab,ijab',rdm1x2,self.h2)
            else:
                ss = torch.einsum('ijab,ijab',rdm1x2,self.alpha * self.h2)
            energy += ss

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        return energy_per_site

    def eval_obs(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average magnetization over the unit cell,
            2. magnetization for each site in the unit cell
            3. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle` 
               for each site in the unit cell
            4. :math:`\mathbf{S}_i.\mathbf{S}_j` for all non-equivalent nearest neighbour
               bonds

        where the on-site magnetization is defined as
        
        .. math::
            
            \begin{align*}
            m &= \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
            =\sqrt{\langle S^z \rangle^2+1/4(\langle S^+ \rangle+\langle S^- 
            \rangle)^2 -1/4(\langle S^+\rangle-\langle S^-\rangle)^2} \\
              &=\sqrt{\langle S^z \rangle^2 + 1/2\langle S^+ \rangle \langle S^- \rangle)}
            \end{align*}

        Usual spin components can be obtained through the following relations
        
        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
        
            for coord,site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                obs[f"SS2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
                obs[f"SS1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2)

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels