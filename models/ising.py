import torch
import su2
from env import ENV
import ipeps
import rdm
from args import GLOBALARGS
from math import sqrt
import itertools

class ISING():
    def __init__(self, hx=0.0, q=0.0, global_args=GLOBALARGS()):
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.hx=hx
        self.q=q
        
        self.h2, self.h4, self.h1 = self.get_h()
        self.obs_ops = self.get_obs_ops()

    # build Ising Hamiltonian in transverse field with plaquette interaction
    # H = - \sum_{<i,j>} h2_i,j + q*\sum_{p} h4_p - hx*\sum_i h1_i 
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
    # where h2_ij = 4*S^z_i S^z_j, indices of h correspond to s_i,s_j;s'_i,s'_j
    #       h4_p  = 16*S^z_i S^z_j S^z_k S^z_l where ijkl labels sites of a plaquette
    #
    #       p: i---j
    #          |   |
    #          k---l, and indices of h_p correspond to s_i,s_j,s_k,s_l;s'_i,s'_j,s'_k,s'_l
    #
    #       h1_i  = 2*S^x_i
    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device) 
        SzSz = 4*torch.einsum('ij,ab->iajb',s2.SZ(),s2.SZ())
        SzSzSzSz = 4*torch.einsum('ijab,klcd->ijklabcd',SzSz,SzSz)
        Sx = s2.SP()+s2.SM()
        return SzSz, SzSzSzSz, Sx 

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
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
    def energy_1x1(self,state,env):
        rdm2x2= rdm.rdm2x2((0,0),state,env)
        eSx= torch.einsum('ijklajkl,ia',rdm2x2,self.h1)
        eSzSz= torch.einsum('ijklabkl,ijab',rdm2x2,self.h2) + \
            torch.einsum('ijklajcl,ikac',rdm2x2,self.h2)
        eSzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
        energy_per_site = -eSzSz - self.hx*eSx + self.q*eSzSzSzSz
        return energy_per_site 

    # assuming reduced density matrix of 2x2 cluster with indexing of DOFs
    # as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
    #
    # s0,s1
    # s2,s3
    #                
    #                          A3--1B   B3  1A
    #                          2 \/ 2   2 \/ 2
    #                A B       0 /\ 0   0 /\ 0
    # Ex.1 unit cell B A terms B3--1A & A3  1B
    #
    #                          A3--1B   B3--1A
    #                          2 \/ 2   2 \/ 2
    #                A B       0 /\ 0   0 /\ 0
    # Ex.2 unit cell A B terms A3--1B & B3--1A
    def energy_2x2_2site(self,state,env):
        pass

    # definition of other observables
    # sp=sx+isy, sm=sx-isy => sx=0.5(sp+sm), sy=-i0.5(sp-sm)
    # m=\sqrt(<sz>^2+<sx>^2+<sy>^2)=\sqrt(<sz>^2+0.25(<sp>+<sm>)^2-0.25(<sp>-<sm>)^2)
    #  =\sqrt(<sz>^2+0.5<sp><sm>)
    #
    # expect "list" of (observable label, value) pairs
    # TODO optimize/unify ?
    def eval_obs(self,state,env):
        obs= dict()
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"sx{coord}"]= 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])
        
        # prepare list with labels and values
        obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels