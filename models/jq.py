import torch
import su2
from env import ENV
import ipeps
import rdm
from args import GLOBALARGS
from math import sqrt
import itertools

class JQ():
    def __init__(self, j1=1.0, q=0.0, global_args=GLOBALARGS()):
        r"""
        :param j1: nearest-neighbour interaction
        :param q: ring-exchange interaction
        :param global_args: global configuration
        :type j1: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J-Q` Hamiltonian

        .. math:: H = J_1\sum_{<i,j>} h2_{ij} - Q\sum_p h4_p  

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        all plaquettes `p`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to 
          :math:`s_i s_j;s'_i s'_j`
        * :math:`h4_p = (\mathbf{S}_i.\mathbf{S}_j-1/4)(\mathbf{S}_k.\mathbf{S}_l-1/4) + 
          (\mathbf{S}_i.\mathbf{S}_k-1/4)(\mathbf{S}_j.\mathbf{S}_l-1/4)` 
          where `i,j,k,l` labels the sites of a plaquette. Hence the `Q` term in the 
          Hamiltian correspond to the following action over plaquette::
          
            {ij,kl}   and    {ik,jl} (double lines denote the (S.S-1/4) terms)   

            i===j            i---j         
            |   |            || ||
            k===l      +     k---l 

          and the indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.q=q
        
        self.h2, self.h4 = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2= su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        expr_kron= 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        SSp= SS - 0.25*id2
        SSSS= torch.einsum('ijab,klcd->ijklabcd',SSp,SSp)
        SSSS= SSSS + SSSS.permute(0,2,1,3,4,6,5,7)
        return SS, SSSS

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    # evaluation of energy depends on the nature of underlying
    # ipeps state
    #
    # Ex.1 for 1-site c4v invariant iPEPS there is just a single 2site
    # operator which gives the energy-per-site
    #
    # Ex.2 for 1-site invariant iPEPS there are two two-site terms
    # which give the energy-per-site
    #    0       0
    # 1--A--3 1--A--3 
    #    2       2                          A
    #    0       0                          2
    # 1--A--3 1--A--3                       0
    #    2       2    , terms A--3 1--A and A have to be evaluated
    #
    # Ex.3 for 2x2 cluster iPEPS there are eight two-site terms
    #    0       0       0
    # 1--A--3 1--B--3 1--A--3
    #    2       2       2
    #    0       0       0
    # 1--C--3 1--D--3 1--C--3
    #    2       2       2             A--3 1--B      A B C D
    #    0       0                     B--3 1--A      2 2 2 2
    # 1--A--3 1--B--3                  C--3 1--D      0 0 0 0
    #    2       2             , terms D--3 1--C and  C D A B
    def energy_1x1c4v(self,ipeps):
        pass

    def energy_2x2_4site(self,state,env):
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
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` and :math:`h4`
        operators are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A 
        """
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        h2x2_nn= torch.einsum('ijab,klcd->ijklabcd',self.h2,id2)
        h2x2_nn= h2x2_nn + h2x2_nn.permute(2,0,3,1,6,4,7,5) \
            + h2x2_nn.permute(0,2,1,3,4,6,5,7) + h2x2_nn.permute(2,3,0,1,6,7,4,5)

        rdm2x2_00= rdm.rdm2x2((0,0),state,env)
        rdm2x2_10= rdm.rdm2x2((1,0),state,env)
        rdm2x2_01= rdm.rdm2x2((0,1),state,env)
        rdm2x2_11= rdm.rdm2x2((1,1),state,env)
        energy_nn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn)
        energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nn)
        energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,h2x2_nn)
        energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,h2x2_nn)
        energy_4 = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,self.h4)
        energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,self.h4)
        energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,self.h4)
        energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,self.h4)

        energy_per_site = 2.0*self.j1*(energy_nn/16.0) - self.q*(energy_4/4.0)
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
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels