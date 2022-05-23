import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v 
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

class JQ():
    def __init__(self, j1=0.0, q=1.0, global_args=cfg.global_args):
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
        all plaquettes `p`, where

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
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.q=q
        
        self.h2, self.h4, self.hp_h_q, self.hp_v_q = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2= su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        expr_kron= 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SSp= SS - 0.25*id2
        SSpSSp= torch.einsum('ijab,klcd->ijklabcd',SSp,SSp)
        SSpSSp= SSpSSp + SSpSSp.permute(0,2,1,3,4,6,5,7)

        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',SS,id2)
        hp_h_q= self.j1*(h2x2_SS + h2x2_SS.permute(2,3,0,1,6,7,4,5)) - self.q*SSpSSp
        hp_v_q= self.j1*(h2x2_SS.permute(0,2,1,3,4,6,5,7) + h2x2_SS.permute(2,0,3,1,6,4,7,5)) \
            - self.q*SSpSSp
        return SS, SSpSSp, hp_h_q, hp_v_q

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

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

        and without assuming any symmetry on the indices of individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` and :math:`h4`
        operators are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A 
        """
        rdm2x2_00= rdm.rdm2x2((0,0),state,env)
        rdm2x2_10= rdm.rdm2x2((1,0),state,env)
        rdm2x2_01= rdm.rdm2x2((0,1),state,env)
        rdm2x2_11= rdm.rdm2x2((1,1),state,env)
        energy= torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,self.hp_h_q)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,self.hp_v_q)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,self.hp_v_q)
        energy+= torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,self.hp_h_q)
        # energy_nn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn)
        # energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nn)
        # energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,h2x2_nn)
        # energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,h2x2_nn)
        # energy_4 = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,self.h4)
        # energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,self.h4)
        # energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_01,self.h4)
        # energy_4 += torch.einsum('ijklabcd,ijklabcd',rdm2x2_11,self.h4)

        # energy_per_site = 2.0*self.j1*(energy_nn/16.0) - self.q*(energy_4/4.0)
        energy_per_site= energy/4.0
        energy_per_site= _cast_to_real(energy_per_site)   

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

            for coord,site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord,state,env)
                rdm1x2 = rdm.rdm1x2(coord,state,env)
                SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.h2)
                SS1x2= torch.einsum('ijab,ijab',rdm1x2,self.h2)
                obs[f"SS2x1{coord}"]= _cast_to_real(SS2x1)
                obs[f"SS1x2{coord}"]= _cast_to_real(SS1x2)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,coord,direction,state,env,dist):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            #rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op= torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                #return op_rot if r%2==0 else op_0
                return op_0
            return _gen_op

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res  

    def eval_corrf_DD_H(self,coord,direction,state,env,dist,verbosity=0):
        r"""
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :param direction: orientation of correlation function
        :type coord: tuple(int,int)
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with horizontal dimer-dimer correlation function
        :rtype: dict(str: torch.Tensor)
        
        Evaluate horizontal dimer-dimer correlation functions 

        .. math::

            \langle(\mathbf{S}(r+3).\mathbf{S}(r+2))(\mathbf{S}(1).\mathbf{S}(0))\rangle 

        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_2sOH2sOH_E1`.
        """
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOH2sOH_E1(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

    def eval_corrf_DD_V(self,coord,direction,state,env,dist,verbosity=0):
        r"""
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :param direction: orientation of correlation function
        :type coord: tuple(int,int)
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with vertical dimer-dimer correlation function
        :rtype: dict(str: torch.Tensor)
        
        Evaluate vertical dimer-dimer correlation functions 

        .. math::
            \langle(\mathbf{S}(r+1,1).\mathbf{S}(r+1,0))(\mathbf{S}(0,1).\mathbf{S}(0,0))\rangle 

        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_2sOV2sOV_E2`.
        """
        # function generating properly S.S operator
        def _gen_op(r):
            return self.h2
        
        D0DR= corrf.corrf_2sOV2sOV_E2(coord, direction, state, env, self.h2, _gen_op,\
            dist, verbosity=verbosity)
        
        res= dict({"dd": D0DR})
        return res

class JQ_C4V():
    def __init__(self, j1=0.0, q=1.0, global_args=cfg.global_args):
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
        all plaquettes `p`, where

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
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.q=q
        
        self.h2, self.h4, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2= su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        expr_kron= 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SSp= SS - 0.25*id2
        SSpSSp= torch.einsum('ijab,klcd->ijklabcd',SSp,SSp)
        SSpSSp= SSpSSp + SSpSSp.permute(0,2,1,3,4,6,5,7)

        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',SS,id2)
        
        #
        #      i===j   i---j         i===j   i---j
        # j1*( |   | + ||  | ) - q*( |   | + || || )
        #      k---l   k---l         k===l   k---l
        #
        hp= self.j1*(h2x2_SS + h2x2_SS.permute(0,2,1,3,4,6,5,7)) - self.q*SSpSSp
        return SS, SSpSSp, hp

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        We assume 1x1 C4v iPEPS which tiles the lattice with tensor A on every site::

            1x1 C4v

            A A A A
            A A A A
            A A A A
            A A A A

        Due to C4v symmetry it is enough to construct a single reduced density matrix 
        :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x2` of a 2x2 plaquette. Afterwards, 
        the energy per site `e` is computed by evaluating a single plaquette term :math:`h_p`
        containing two nearest-neighbour terms :math:`\bf{S}.\bf{S}` and `h4_p` as:

        .. math::

            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})

        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp)
        return energy_per_site

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
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
            rdm2x1 = rdm_c4v.rdm2x1(state,env_c4v)
            obs[f"SS2x1"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

class JQ_C4V_BIPARTITE():
    def __init__(self, j1=0.0, q=1.0, global_args=cfg.global_args):
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
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.q=q
        
        self.h2, self.h2_rot, self.h4_rot, self.hp_rot = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2= su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        expr_kron= 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        rot_op= s2.BP_rot()
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,SS,rot_op)

        SSp_rot= SS_rot - 0.25*id2
        SSpSSp_rot= torch.einsum('ijab,klcd->ijklabcd',SSp_rot,SSp_rot)
        SSpSSp_rot= SSpSSp_rot + SSpSSp_rot.permute(0,2,1,3,4,6,5,7)

        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',SS_rot,id2)
        
        #
        #      i===j   i---j         i===j   i---j
        # j1*( |   | + ||  | ) - q*( |   | + || || )
        #      k---l   k---l         k===l   k---l
        #
        hp_rot= self.j1*(h2x2_SS + h2x2_SS.permute(0,2,1,3,4,6,5,7)) - self.q*SSpSSp_rot
        return SS, SS_rot, SSpSSp_rot, hp_rot

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        We assume 1x1 C4v iPEPS which tiles the lattice with a bipartite pattern composed 
        of two tensors A, and B=RA, where R rotates approriately the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation P => BIPARTITE

            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        Due to C4v symmetry it is enough to construct a single reduced density matrix 
        :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x2` of a 2x2 plaquette. Afterwards, 
        the energy per site `e` is computed by evaluating a single plaquette term :math:`h_p`
        containing two nearest-neighbour terms :math:`\bf{S}.\bf{S}` and `h4_p` as:

        .. math::

            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})

        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp_rot)
        return energy_per_site

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
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
            rdm2x1 = rdm_c4v.rdm2x1(state,env_c4v)
            obs[f"SS2x1"]= torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
            
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
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.h2,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sOH2sOH_E1(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

    def eval_corrf_DD_V(self,state,env_c4v,dist,verbosity=0):
        r"""
        Evaluates correlation functions of two vertical dimers
        DD_v(r)= <(S(0).S(y))(S(r*x).S(y+r*x))>
             or= <(S(0).S(x))(S(r*y).S(x+r*y))> 
        """
        # function generating properly S.S operator
        # function generating properly rotated S.S operator on every bi-partite site
        rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
        # (S.S)_s1s2,s1's2' with rotation applied on "first" spin s1,s1' 
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.h2,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sOV2sOV_E2(state, env_c4v, op_rot, _gen_op,\
            dist, verbosity=verbosity)
        
        res= dict({"dd": D0DR})
        return res

class JQ_C4V_PLAQUETTE():
    def __init__(self, j1=0.0, q=1.0, q_inter=1.0, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param q: ring-exchange interaction
        :param global_args: global configuration
        :type j1: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J-Q` Hamiltonian for 1-site C4v symmetric iPEPS, where
        each tensor represent four spins on a plaquette, hence the physical dimension
        of each tensor becomes :math:`2^4`. The Hilbert spaces of fours spins on a 
        plaquette are merged into single Hilbert space in the following order::

            s0--s1
            |   | 
            s2--s3

        The original Hamiltonian now contains only nearest-neighbours and on-site terms:

        .. math:: H = \sum_{<i,j>} h2_{ij} + \sum_i h1_i  

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        all sites::

            y\x
               _:__:__:__:_              :    :
            ..._|_p|__|_p|_...  =>  ...__p____p__...
            ..._|__|__|__|_...           |    |
            ..._|_p|__|_p|_...      ...__p____p__...
            ..._|__|__|__|_...           |    |
            ..._|_p|__|_p|_...      ...__p____p__...
                :  :  :  :               :    :
 
        where

        * .. math::
            
            \begin{align*}
            h1_i &= q((\mathbf{S}_{s0_i}.\mathbf{S}_{s1_i}-1/4)(\mathbf{S}_{s2_i}.\mathbf{S}_{s3_i}-1/4)
            + (\mathbf{S}_{s0_i}.\mathbf{S}_{s2_i}-1/4)(\mathbf{S}_{s1_i}.\mathbf{S}_{s3_i}-1/4)) \\
            &+ J(\mathbf{S}_{s0_i}.\mathbf{S}_{s1_i} + \mathbf{S}_{s2_i}.\mathbf{S}_{s3_i} +
            \mathbf{S}_{s0_i}.\mathbf{S}_{s2_i} + \mathbf{S}_{s1_i}.\mathbf{S}_{s3_i})
            \end{align*}

        * .. math::
        
            \begin{align*}
            h2_{ij} &= h2_{horizontal; ij} + h2_{vertical; ij} \\
            &= q((\mathbf{S}_{s1_i}.\mathbf{S}_{s0_j}-1/4)(\mathbf{S}_{s3_i}.\mathbf{S}_{s2_j}-1/4)
            + (\mathbf{S}_{s1_i}.\mathbf{S}_{s3_i}-1/4)(\mathbf{S}_{s0_j}.\mathbf{S}_{s2_j}-1/4)) \\
            &+ J(\mathbf{S}_{s1_i}.\mathbf{S}_{s0_j} + \mathbf{S}_{s3_i}.\mathbf{S}_{s2_j}) \\
            &+ q((\mathbf{S}_{s2_i}.\mathbf{S}_{s0_j}-1/4)(\mathbf{S}_{s3_i}.\mathbf{S}_{s1_j}-1/4)
            + (\mathbf{S}_{s2_i}.\mathbf{S}_{s3_i}-1/4)(\mathbf{S}_{s0_j}.\mathbf{S}_{s1_j}-1/4)) \\
            &+ J(\mathbf{S}_{s2_i}.\mathbf{S}_{s0_j} + \mathbf{S}_{s3_i}.\mathbf{S}_{s1_j})
            \end{align*}
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2**4
        self.j1=j1
        self.q=q
        self.q_inter=q_inter
        
        self.h1, self.h2, self.h2_compressed, self.SS = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        # from "bra" tuple of indices return corresponing bra-ket pair of indices
        # 0 -> 0;1; 1,0 -> 1,0;3,2; ...
        def bk(*bras):
            kets=[b+len(bras) for b in bras]
            return tuple(list(bras)+kets)

        s2= su2.SU2(2, dtype=self.dtype, device=self.device)
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        id3= torch.eye(8,dtype=self.dtype,device=self.device)
        id3= id3.view(2,2,2,2,2,2).contiguous()
        expr_kron= 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SSp= SS - 0.25*id2
        SSid2= torch.einsum('ijab,klcd->ijklabcd',SS,id2)
        SSpSSp= torch.einsum('ijab,klcd->ijklabcd',SSp,SSp)
        SSpSSp= SSpSSp + SSpSSp.permute(bk(0,2,1,3))
       
        # on-site term
        h1= self.j1*(SSid2 +SSid2.permute(bk(2,3,0,1)) +SSid2.permute(bk(0,2,1,3))\
            +SSid2.permute(bk(2,0,3,1))) - self.q*SSpSSp
        h1= h1.view(self.phys_dim,self.phys_dim)
        
        # nearest-neighbour term:
        #
        # S.S terms:
        # (s0 s1 s2 s3)_i(s0 s1 s2 s3)_j;(s0's1's2's3')_i(s0's1's2's3')_j                     
        #     ^           ^                  ^            ^  
        # s0--s1~~s0--s1
        # |   |   |   |
        # s2--s3  s2--s3
        # i       j
        #
        SiSj= torch.einsum('ijab,efgmno,qrsxyz->eifgjqrsmanobxyz',SS,id3,id3)
        #
        # SSpSSp terms:
        # s0--s1  s0--s1 + s0--s1==s0--s1
        # |   ||  ||   |   |   |   |    |
        # s2--s3  s2--s3   s2--s3==s2--s3
        # i       j        i       j
        SSpiSSpj= torch.einsum('ijklabcd,efmn,ghxy->eifjkglhmanbcxdy',SSpSSp,id2,id2)
        h2= self.j1*(SiSj + SiSj.permute(bk(0,3,2,1,6,5,4,7))) - self.q_inter*(SSpiSSpj) 
        h2+= self.j1*(SiSj.permute(bk(0,2,1,3,4,5,6,7)) + SiSj.permute(bk(0,3,2,1,5,4,6,7)))\
            -self.q_inter*(SSpiSSpj.permute(bk(0,2,1,3,4,6,5,7)))
        h2= h2.contiguous().view(self.phys_dim*self.phys_dim,self.phys_dim*self.phys_dim)
        h2_U, h2_S, h2_V= torch.svd(h2)
        h2_S= h2_S[h2_S > 1.0e-14]
        h2_U= h2_U[:,:len(h2_S)]
        h2_V= h2_V[:,:len(h2_S)]
        h2= h2.view(self.phys_dim,self.phys_dim,self.phys_dim,self.phys_dim)
        return h1, h2, (h2_U, h2_S, h2_V), SS

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env: ENV_C4V
        :return: energy per site
        :rtype: float

        For 1-site invariant c4v iPEPS it's enough to construct a 1-site reduced
        density matrix :py:func:`ctm.one_site_c4v.rdm_c4v.rdm1x1`, effectively 
        representing a 2x2 plaquette, and 2-site reduced
        density matrix :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1` which represents 
        interaction between two plaquettes of the underlying physical system:
        
        .. math:: 

            e = \langle h1 \rangle_{\rho_{1x1}} + \langle h2 \rangle_{\rho_{2x1}}

        """
        rdm1x1= rdm_c4v.rdm1x1(state,env)
        rdm2x1= rdm_c4v.rdm2x1(state,env)
        e1s= torch.einsum('ij,ij',rdm1x1,self.h1)
        e2s= torch.einsum('ijab,ijab',rdm2x1,self.h2)
        energy_per_site= (e1s+e2s)/4
        return energy_per_site

    def eval_obs(self,state,env):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env: ENV_C4V
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
            rdm1x1= rdm_c4v.rdm1x1(state,env)
            rdm1x1= rdm1x1.view(2,2,2,2,2,2,2,2)
            expr_core='abc'
            for r in range(4):
                expr=expr_core[:r]+'i'+expr_core[r:]+expr_core[:r]+'j'+expr_core[r:]+',ij'
                for label,op in self.obs_ops.items():
                    obs[f"{label}{r}"]= torch.einsum(expr,rdm1x1,op)
                obs[f"m{r}"]= sqrt(abs(obs[f"sz{r}"]**2 + obs[f"sp{r}"]*obs[f"sm{r}"]))
                obs["avg_m"] += obs[f"m{r}"]
            obs["avg_m"]= obs["avg_m"]/4

            # for coord,site in state.sites.items():
            #     rdm2x1 = rdm_c4v.rdm2x1(coord,state,env)
            #     rdm1x2 = rdm.rdm1x2(coord,state,env)
            #     obs[f"SS2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
            #     obs[f"SS1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{r}" for r in range(4)]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(range(4), self.obs_ops.keys()))]
        # obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        # obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels