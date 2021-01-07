import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from math import sqrt
import itertools

class HB():
    def __init__(self, spin_s=2, j1=1.0, k1=0.0, global_args=cfg.global_args):
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=spin_s
        self.j1=j1
        self.k1=k1
        
        self.h2, self.hp_h, self.hp_v, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    # build spin-S bilinear-biquadratic Hamiltonian
    # H = \sum_{<i,j>} h_i,j
    #  
    # y\x
    #    _:__:__:__:_
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    #     :  :  :  : 
    # 
    # where h_ij = S_i.S_j, indices of h correspond to s_i,s_j;s_i',s_j'
    def get_h(self):
        pd = self.phys_dim
        irrep = su2.SU2(pd, dtype=self.dtype, device=self.device)
        # identity operator on two spin-S spins
        idp= torch.eye(pd**2, dtype=self.dtype,device=self.device)
        idp= idp.view(pd,pd,pd,pd).contiguous()
        SS= irrep.SS() 
        SS= SS.view(pd**2,pd**2)
        h2= self.j1*SS + self.k1*SS@SS
        # Reshape back into rank-4 tensor for later use with reduced density matrices
        h2= h2.view(pd,pd,pd,pd).contiguous()

        h2x2_h2= torch.einsum('ijab,klcd->ijklabcd',h2,idp)
        # Create operators acting on four spins-S on plaquette. These are useful 
        # for computing energy by different rearragnement of Hamiltonian terms
        #
        # NN-terms along horizontal bonds of plaquette  
        hp_h= h2x2_h2 + h2x2_h2.permute(2,3,0,1,6,7,4,5)
        # NN-terms along vertical bonds of plaquette
        hp_v= h2x2_h2.permute(0,2,1,3,4,6,5,7) + h2x2_h2.permute(2,0,3,1,6,4,7,5)
        # All NN-terms within plaquette
        hp=hp_h+hp_v

        return h2, hp_h, hp_v, hp

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= irrep.SZ()
        obs_ops["sp"]= irrep.SP()
        obs_ops["sm"]= irrep.SM()
        obs_ops["SS"]= irrep.SS()

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
    def energy_2x1_1x2(self,state,env):
        energy=0.
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            energy += torch.einsum('ijab,ijab',rdm2x1,self.h2)
            energy += torch.einsum('ijab,ijab',rdm1x2,self.h2)

        energy_per_site=energy/len(state.sites.items())
        return energy_per_site

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
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A 
        """
        rdm2x2_00= rdm.rdm2x2((0,0),state,env)
        rdm2x2_10= rdm.rdm2x2((1,0),state,env)
        rdm2x2_01= rdm.rdm2x2((0,1),state,env)
        rdm2x2_11= rdm.rdm2x2((1,1),state,env)
        energy= torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,self.hp_h)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,self.hp_v)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,self.hp_v)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,self.hp_h)

        energy_per_site= energy/4.0
        return energy

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
            # one-site observables
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label in ["sz","sp","sm"]:
                    op= self.obs_ops[label]
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
        
            op_SS= self.obs_ops["SS"]
            for coord,site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                obs[f"SS2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,op_SS)
                obs[f"SS1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,op_SS)

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sp","sm"]))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels