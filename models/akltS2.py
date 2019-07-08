import torch
import su2
from env import ENV
import ipeps
import rdm
from args import GLOBALARGS

class AKLTS2():
    def __init__(self, global_args=GLOBALARGS()):
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim = 5

        self.h = self.get_h()
        self.obs_ops = self.get_obs()

    # build AKLT S=2 Hamiltonian <=> Projector from product of two S=2 DOFs
    # to S=4 DOF H = \sum_{<i,j>} h_ij, where h_ij= ...
    #
    # indices of h correspond to s_i,s_j;s_i',s_j'
    def get_h(self):
        pd = self.phys_dim
        s5 = su2.SU2(pd, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s5.SZ(),s5.SZ()) + 0.5*(torch.einsum(expr_kron,s5.SP(),s5.SM()) \
            + torch.einsum(expr_kron,s5.SM(),s5.SP()))
        SS = SS.view(pd*pd,pd*pd)
        h = (1./14)*(SS + (7./10.)*SS@SS + (7./45.)*SS@SS@SS + (1./90.)*SS@SS@SS@SS)
        h = h.view(pd,pd,pd,pd)
        return h

    def get_obs(self):
        obs_ops = dict()
        s5 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s5.SZ()
        obs_ops["sp"]= s5.SP()
        obs_ops["sm"]= s5.SM()
        return obs_ops

    # evaluation of energy depends on the nature of underlying
    # ipeps state
    #
    # Ex.1 for 1-site c4v invariant iPEPS there is just a single two-site
    # term which gives the energy-per-site
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
    def energy_1x1c4v(self,state,env):
        rdm2x1 = rdm.rdm2x1((0,0), state, env)
        # apply a rotation on physical index of every "odd" site
        # A A => A B
        # A A => B A
        rot_op = su2.get_rot_op(5)
        h_rotated = torch.einsum('jl,ilak,kb->ijab',rot_op,self.h,rot_op)
        energy = torch.einsum('ijab,ijab',rdm2x1,h_rotated)
        return energy       

    def energy_1x1(self,state,env):
        rdm2x1 = rdm.rdm2x1((0,0), state, env)
        rdm1x2 = rdm.rdm1x2((0,0), state, env)
        # apply a rotation on physical index of every "odd" site
        # A A => A B
        # A A => B A
        rot_op = su2.get_rot_op(5)
        h_rotated = torch.einsum('jl,ilak,kb->ijab',rot_op,self.h,rot_op)
        energy = torch.einsum('ijab,ijab',rdm2x1,h_rotated) + torch.einsum('ijab,ijab',rdm1x2,h_rotated)
        return energy       

    def energy_2x2(self,ipeps):
        pass

    # assuming reduced density matrix of 2x2 cluster with indexing of DOFs
    # as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
    def energy_2x2(self,rdm2x2):
        energy = torch.einsum('ijklabkl,ijab',rdm2x2,self.h)
        return energy

    def energy_2x1_1x2(self,state,env):
        energy=0.
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            energy += torch.einsum('ijab,ijab',rdm2x1,self.h)
            energy += torch.einsum('ijab,ijab',rdm1x2,self.h)

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        return energy_per_site
        
    # definition of other observables
    def eval_obs(self,state,env):
        obs= dict()
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[str(coord)+"|"+label] = torch.trace(rdm1x1@op)

        return obs