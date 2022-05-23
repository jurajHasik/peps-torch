import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

_cast_to_real= rdm._cast_to_real

class AKLTS2():
    def __init__(self, global_args=cfg.global_args):
        r"""
        :param global_args: global configuration
        :type global_args: GLOBALARGS

        Build AKLT S=2 Hamiltonian, equivalent to projector from product of two S=2 DOFs
        to S=4 DOF
        
        .. math::
            H = \sum_{<ij>} h_{ij},\ \ \ h_{ij}= \frac{1}{14} \vec{S}_i\cdot\vec{S}_j
                + \frac{7}{10} (\vec{S}_i\cdot\vec{S}_j)^2 + \frac{7}{45} (\vec{S}_i\cdot\vec{S}_j)^3
                + \frac{1}{90} (\vec{S}_i\cdot\vec{S}_j)^4

        where `<ij>` denote nearest neighbours.
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim= 5

        self.h, self.SS = self.get_h()
        self.obs_ops = self.get_obs()

    def get_h(self):
        pd = self.phys_dim
        s5 = su2.SU2(pd, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s5.SZ(),s5.SZ()) + 0.5*(torch.einsum(expr_kron,s5.SP(),s5.SM()) \
            + torch.einsum(expr_kron,s5.SM(),s5.SP()))
        SS = SS.view(pd*pd,pd*pd)
        h = (1./14)*(SS + (7./10.)*SS@SS + (7./45.)*SS@SS@SS + (1./90.)*SS@SS@SS@SS)
        h = h.view(pd,pd,pd,pd)
        SS = SS.view(pd,pd,pd,pd)
        return h, SS

    def get_obs(self):
        obs_ops = dict()
        s5 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s5.SZ()
        obs_ops["sp"]= s5.SP()
        obs_ops["sm"]= s5.SM()
        return obs_ops
 
    def energy_2x1_1x2(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float
        
        We assume iPEPS with 2x1 unit cell with tensors A, B and bipartite tiling or 
        2x2 unit cell containing four tensors A, B, C, and D with a simple PBC tiling::

            A B A B  or  A B A B
            B A B A      C D C D
            A B A B      A B A B
            B A B A      C D C D

        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        and without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site for the 
        case of 2x1 unit cell with bipartite tiling::

               0       0
            1--A--3 1--B--3
               2       2                              A  B
               0       0                              2  2  
            1--B--3 1--A--3           A--3 1--B,      0  0 
               2       2      , terms B--3 1--A, and  B, A

        and for the case of 2x2 unit cell::

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
            rdm2x1 = rdm.rdm2x1(coord,state,env,**kwargs)
            rdm1x2 = rdm.rdm1x2(coord,state,env,**kwargs)
            energy += torch.einsum('ijab,ijab',rdm2x1,self.h)
            energy += torch.einsum('ijab,ijab',rdm1x2,self.h)

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        energy_per_site= _cast_to_real(energy_per_site,**kwargs)

        return energy_per_site
        
    # definition of other observables
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
            4. nearest-neighbour spin-spin correlations on non-equivalent bonds

        where the on-site magnetization is defined as
        
        .. math::
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
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
                SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.SS)
                SS1x2= torch.einsum('ijab,ijab',rdm1x2,self.SS)
                obs[f"SS2x1{coord}"]= _cast_to_real(SS2x1)
                obs[f"SS1x2{coord}"]= _cast_to_real(SS1x2)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

class AKLTS2_C4V_BIPARTITE():
    def __init__(self, global_args=cfg.global_args):
        r"""
        :param global_args: global configuration
        :type global_args: GLOBALARGS

        Build AKLT S=2 Hamiltonian, equivalent to projector from product of two S=2 DOFs
        to S=4 DOF
        
        .. math::
            H = \sum_{<ij>} h_{ij},\ \ \ h_{ij}= \frac{1}{14} \vec{S}_i\cdot\vec{S}_j
                + \frac{7}{10} (\vec{S}_i\cdot\vec{S}_j)^2 + \frac{7}{45} (\vec{S}_i\cdot\vec{S}_j)^3
                + \frac{1}{90} (\vec{S}_i\cdot\vec{S}_j)^4

        where `<ij>` denote nearest neighbours.
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim= 5

        self.h2_rot, self.SS, self.SS_rot = self.get_h()
        self.obs_ops = self.get_obs()

    def get_h(self):
        pd = self.phys_dim
        s5 = su2.SU2(pd, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SS = torch.einsum(expr_kron,s5.SZ(),s5.SZ()) + 0.5*(torch.einsum(expr_kron,s5.SP(),s5.SM()) \
            + torch.einsum(expr_kron,s5.SM(),s5.SP()))
        rot_op = s5.BP_rot()
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

    def energy_1x1(self,state,env_c4v,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        We assume 1x1 C4v iPEPS which tiles the lattice with a bipartite pattern composed 
        of two tensors A, and B=RA, where R appropriately rotates the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation R => BIPARTITE
            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        Due to C4v symmetry it is enough to construct just a single nearest-neighbour
        reduced density matrix 

        .. math::
            e= \langle \mathcal{h} \rangle = Tr(\rho_{2x1} \mathcal{h})
        """
        rdm2x1 = rdm_c4v.rdm2x1(state, env_c4v,**kwargs)
        energy = torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
        energy = _cast_to_real(energy,**kwargs)
        return energy
        
    # definition of other observables
    def eval_obs(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. magnetization
            2. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle`
    
        where the on-site magnetization is defined as
        
        .. math::
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
        """
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict()
        with torch.no_grad():
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
            rdm2x1 = rdm_c4v.rdm2x1(state,env_c4v)
            obs[f"SS2x1"]= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` .
        """
   
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
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with horizontal dimer-dimer correlation function
        :rtype: dict(str: torch.Tensor)
        
        Evaluate horizontal dimer-dimer correlation functions 

        .. math::
            \langle(\mathbf{S}(r+3).\mathbf{S}(r+2))(\mathbf{S}(1).\mathbf{S}(0))\rangle 

        up to r = ``dist`` .
        """
        # function generating properly rotated S.S operator on every bi-partite site
        rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
        # (S.S)_s1s2,s1's2' with rotation applied on "first" spin s1,s1' 
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sOH2sOH_E1(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res