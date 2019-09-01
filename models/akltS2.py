import torch
import su2
from c4v import *
import config as cfg
import ipeps
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

class AKLTS2():
    def __init__(self, global_args=cfg.global_args):
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

class AKLTS2_C4V_BIPARTITE():
    def __init__(self, global_args=cfg.global_args):
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim = 5

        self.h2_rot, self.SS, self.SS_rot = self.get_h()
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
        rot_op = su2.get_rot_op(5)
        SS_rot = torch.einsum('jl,ilak,kb->ijab',rot_op,SS,rot_op)
        SS = SS.view(pd*pd,pd*pd)
        h = (1./14)*(SS + (7./10.)*SS@SS + (7./45.)*SS@SS@SS + (1./90.)*SS@SS@SS@SS)
        h = h.view(pd,pd,pd,pd)
        # apply a rotation on physical index of every "odd" site
        # A A => A B
        # A A => B A
        h_rot = torch.einsum('jl,ilak,kb->ijab',rot_op,h,rot_op)
        SS = SS.view(pd,pd,pd,pd)
        return h_rot, SS, SS_rot

    def get_obs(self):
        obs_ops = dict()
        s5 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s5.SZ()
        obs_ops["sp"]= s5.SP()
        obs_ops["sm"]= s5.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v):
        rdm2x1 = rdm_c4v.rdm2x1(state, env_c4v)
        energy = torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
        return energy
        
    # definition of other observables
    def eval_obs(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. magnetization
            2. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle`
    
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
        obs= dict()
        with torch.no_grad():
            # symmetrize on-site tensor
            symm_sites= {(0,0): make_c4v_symm(state.sites[(0,0)])}
            symm_sites[(0,0)]= symm_sites[(0,0)]/torch.max(torch.abs(symm_sites[(0,0)]))
            symm_state= ipeps.IPEPS(symm_sites)

            rdm1x1= rdm_c4v.rdm1x1(symm_state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
            rdm2x1 = rdm_c4v.rdm2x1(symm_state,env_c4v)
            obs[f"SS2x1"]= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist):
   
        # function generating properly rotated operators on every bi-partite site
        def get_bilat_op(op):
            rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                return op_rot if r%2==0 else op_0
            return _gen_op

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, self.obs_ops["sz"], \
            get_bilat_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf_c4v.corrf_1sO1sO(state, env_c4v, op_sx, get_bilat_op(op_sx), dist)
        nSy0SyR= corrf_c4v.corrf_1sO1sO(state, env_c4v, op_isy, get_bilat_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

    def eval_corrf_DD_H(self,state,env_c4v,dist,verbosity=0):
        # function generating properly rotated S.S operator on every bi-partite site
        rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
        # (S.S)_s1s2,s1's2' with rotation applied on "first" spin s1,s1' 
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sO2sO_H(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res