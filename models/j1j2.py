import torch
import su2
from env import ENV
import ipeps
import rdm
from args import GLOBALARGS
from math import sqrt
import itertools

class J1J2():
    def __init__(self, j1=1.0, j2=0.0, global_args=GLOBALARGS()):
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.h = self.get_h()
        self.obs_ops = self.get_obs_ops()

    # build spin-1/2 J1-J2 Hamiltonian
    # H = j1*\sum_{<i,j>} h_i,j + j2*\sum_<<i,j>> h_i,j
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
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        h2x2_nn= torch.einsum('ijab,klcd->ijklabcd',self.h,id2)
        h2x2_nn= h2x2_nn + h2x2_nn.permute(2,0,3,1,6,4,7,5) \
            + h2x2_nn.permute(0,2,1,3,4,6,5,7) + h2x2_nn.permute(2,3,0,1,6,7,4,5)
        h2x2_nnn= torch.einsum('ijab,klcd->ikljacdb',self.h,id2)
        h2x2_nnn= h2x2_nnn + h2x2_nnn.permute(1,0,3,2,5,4,7,6)

        rdm2x2_00= rdm.rdm2x2((0,0),state,env)
        rdm2x2_10= rdm.rdm2x2((1,0),state,env)
        energy_nn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn)
        energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nn)
        energy_nnn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nnn)
        energy_nnn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nnn)

        energy_per_site = 2.0*(self.j1*energy_nn/8.0 + self.j2*energy_nnn/4.0)
        return energy_per_site

    # definition of other observables
    # sp=sx+isy, sm=sx-isy => sx=0.5(sp+sm), sy=-i0.5(sp-sm)
    # m=\sqrt(<sz>^2+<sx>^2+<sy>^2)=\sqrt(<sz>^2+0.25(<sp>+<sm>)^2-0.25(<sp>-<sm>)^2)
    #  =\sqrt(<sz>^2+0.5<sp><sm>)
    #
    # expect "list" of (observable label, value) pairs
    # TODO optimize/unify ?
    def eval_obs(self,state,env):
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