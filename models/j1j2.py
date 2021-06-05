import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import rdm_c4v_thermal
from ctm.one_site_c4v.rdm_c4v_specialized import rdm2x2_NNN_tiled,\
    rdm2x2_NN_tiled, rdm2x1_tiled
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

class J1J2():
    def __init__(self, j1=1.0, j2=0, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: H = J_1\sum_{<i,j>} h2_{ij} + J_2\sum_{<<i,j>>} h2_{ij}

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`)::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = \mathbf{S_i}.\mathbf{S_j}` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.h2, self.SS_rot, self.h2x2_nn, self.h2x2_nnn, self.h2x2_nn_rot, \
            self.h2x2_nnn_rot= self.get_h()
        self.obs_ops= self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        rot_op= s2.BP_rot()
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SS= SS.contiguous()

        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,SS,rot_op)
        
        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',SS,id2)
        h2x2_nn= h2x2_SS + h2x2_SS.permute(2,3,0,1,6,7,4,5) + h2x2_SS.permute(0,2,1,3,4,6,5,7)\
            + h2x2_SS.permute(2,0,3,1,6,4,7,5)
        h2x2_nnn= h2x2_SS.permute(0,3,2,1,4,7,6,5) + h2x2_SS.permute(2,0,1,3,6,4,5,7)
        
        h2x2_nn= h2x2_nn.contiguous()
        h2x2_nnn= h2x2_nnn.contiguous()

        # sublattice rotation for single-site bipartite (BP) ansatz
        h2x2_nn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
            h2x2_nn,rot_op,rot_op,rot_op,rot_op)
        h2x2_nnn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
            h2x2_nnn,rot_op,rot_op,rot_op,rot_op)

        h2x2_nn_rot= h2x2_nn_rot.contiguous()
        h2x2_nnn_rot= h2x2_nnn_rot.contiguous()

        return SS, SS_rot, h2x2_nn, h2x2_nnn, h2x2_nn_rot, h2x2_nnn_rot

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_2x2_1site_BP(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume 1x1 iPEPS which tiles the lattice with a bipartite pattern composed 
        of two tensors A, and B=RA, where R rotates approriately the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation P => BIPARTITE

            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        A single reduced density matrix :py:func:`ctm.rdm.rdm2x2` of a 2x2 plaquette
        is used to evaluate the energy.
        """
        tmp_rdm= rdm.rdm2x2((0,0),state,env)
        energy_nn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn_rot)
        energy_nnn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn_rot)
        energy_per_site= 2.0*(self.j1*energy_nn/4.0 + self.j2*energy_nnn/2.0)
        energy_per_site= _cast_to_real(energy_per_site) 

        return energy_per_site

    def energy_2x2_2site(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 2x1 unit cell containing two tensors A, B. We can
        tile the square lattice in two ways::

            BIPARTITE           STRIPE   

            A B A B             A B A B
            B A B A             A B A B
            A B A B             A B A B
            B A B A             A B A B

        Taking reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster with indexing 
        of sites as follows :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::
                
               0           
            1--A--3
               2
            
            Ex.1 unit cell A B, with BIPARTITE tiling

                A3--1B, B3--1A, A, B, A3  , B3  ,   1A,   1B
                                2  0   \     \      /     / 
                                0  2    \     \    /     /  
                                B  A    1A    1B  A3    B3  
            
            Ex.2 unit cell A B, with STRIPE tiling

                A3--1A, B3--1B, A, B, A3  , B3  ,   1A,   1B
                                2  0   \     \      /     / 
                                0  2    \     \    /     /  
                                A  B    1B    1A  B3    A3  
        """
        # A3--1B   B3  1A
        # 2 \/ 2   2 \/ 2
        # 0 /\ 0   0 /\ 0
        # B3--1A & A3  1B

        # A3--1B   B3--1A
        # 2 \/ 2   2 \/ 2
        # 0 /\ 0   0 /\ 0
        # A3--1B & B3--1A

        energy_nn=0
        energy_nnn=0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
            energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)
        energy_per_site= 2.0*(self.j1*energy_nn/8.0 + self.j2*energy_nnn/4.0)
        energy_per_site= _cast_to_real(energy_per_site)

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
        energy_nn=0
        energy_nnn=0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
            energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)
        energy_per_site= 2.0*(self.j1*energy_nn/16.0 + self.j2*energy_nnn/8.0)
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def energy_2x2_8site(self,state,env):
        r"""

        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 4x2 unit cell containing eight tensors A, B, C, D, 
        E, F, G, H with PBC tiling + SHIFT::

            A B E F
            C D G H
          A B E F
          C D G H
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of eight :math:`\rho_{2x2}`'s are needed over which :math:`h2` operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            A3--1B   B3--1E   E3--1F   F3--1A
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1G & G3--1H & H3--1C 

            C3--1D   D3--1G   G3--1H   H3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            B3--1E & E3--1F & F3--1A & A3--1B 
        """
        energy_nn=0
        energy_nnn=0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
            energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)
        energy_per_site= 2.0*(self.j1*energy_nn/32.0 + self.j2*energy_nnn/16.0)
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def eval_obs_1site_BP(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]
        
        Evaluates observables for single-site ansatz by including the sublattice
        rotation in the physical space. 
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
                SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
                SS1x2= torch.einsum('ijab,ijab',rdm1x2,self.SS_rot)
                obs[f"SS2x1{coord}"]= _cast_to_real(SS2x1)
                obs[f"SS1x2{coord}"]= _cast_to_real(SS1x2)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

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

class J1J2_C4V_BIPARTITE():
    def __init__(self, j1=1.0, j2=0, j3=0, hz_stag= 0.0, delta_zz=1.0, \
        global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param hz_stag: staggered magnetic field
        :param delta_zz: easy-axis (nearest-neighbour) anisotropy
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type hz_stag: float
        :type detla_zz: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: 

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
            = \sum_{p} h_p

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`)::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h_p = J_1(S^x_{r}.S^x_{r+\vec{x}} 
          + S^y_{r}.S^y_{r+\vec{x}} + \delta_{zz} S^z_{r}.S^z_{r+\vec{x}} + (x<->y))
          + J_2(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}+\vec{y}} + \mathbf{S}_{r+\vec{x}}.\mathbf{S}_{r+\vec{y}})
          + h_stag (S^z_{r} - S^z_{r+\vec{x}} - S^z_{r+\vec{y}} + S^z_{r+\vec{x}+\vec{y}})` 
          with indices of spins ordered as follows :math:`s_r s_{r+\vec{x}} s_{r+\vec{y}} s_{r+\vec{x}+\vec{y}};
          s'_r s'_{r+\vec{x}} s'_{r+\vec{y}} s'_{r+\vec{x}+\vec{y}}`

        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        self.j3=j3
        self.hz_stag=hz_stag
        self.delta_zz=delta_zz
        
        self.obs_ops = self.get_obs_ops()

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(self.phys_dim**2,dtype=self.dtype,device=self.device)
        id2= id2.view(tuple([self.phys_dim]*4)).contiguous()
        expr_kron = 'ij,ab->iajb'

        self.SS_delta_zz= self.delta_zz*torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + \
            0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        self.SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + \
            0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        hz_2x1_nn= torch.einsum(expr_kron,s2.SZ(),s2.I())+torch.einsum(expr_kron,s2.I(),-s2.SZ())

        rot_op= s2.BP_rot()
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS,rot_op)
        SS_delta_zz_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS_delta_zz,rot_op)
        hz_2x1_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,hz_2x1_nn,rot_op)
        self.SS_rot= SS_rot.contiguous()
        self.SS_delta_zz_rot= SS_delta_zz_rot.contiguous()
        self.hz_2x1_rot= hz_2x1_rot.contiguous()

        h2x2_SS_delta_zz= torch.einsum('ijab,klcd->ijklabcd',self.SS_delta_zz,id2) # nearest neighbours
        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',self.SS,id2) # next-nearest neighbours
        # 0 1     0 1   0 x   x x   x 1
        # 2 3 ... x x + 2 x + 2 3 + x 3
        hp= 0.5*self.j1*(h2x2_SS_delta_zz + h2x2_SS_delta_zz.permute(0,2,1,3,4,6,5,7)\
           + h2x2_SS_delta_zz.permute(2,3,0,1,6,7,4,5) + h2x2_SS_delta_zz.permute(3,1,2,0,7,5,6,4)) \
           + self.j2*(h2x2_SS.permute(0,3,2,1,4,7,6,5) + h2x2_SS.permute(2,1,0,3,6,5,4,7))\
           - 0.25*self.hz_stag*torch.einsum('ia,jb,kc,ld->ijklabcd',s2.SZ(),-s2.SZ(),-s2.SZ(),s2.SZ())
        hp= torch.einsum('xj,yk,ixylauvd,ub,vc->ijklabcd',rot_op,rot_op,hp,rot_op,rot_op)
        self.hp= hp.contiguous()

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v,**kwargs):
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
        containing two nearest-nighbour terms :math:`\bf{S}.\bf{S}` and two next-nearest 
        neighbour :math:`\bf{S}.\bf{S}`, as:

        .. math::

            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})
        
        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,sym_pos_def=True,\
            verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp)
        if abs(self.j3)>0:
            rdm3x1= rdm_c4v.rdm3x1(state,env_c4v,sym_pos_def=True,\
                force_cpu=False,verbosity=cfg.ctm_args.verbosity_rdm)
            ss_3x1= torch.einsum('ijab,ijab',rdm3x1,self.SS)
            energy_per_site= energy_per_site + 2*self.j3*ss_3x1

        energy_per_site= _cast_to_real(energy_per_site)
        return energy_per_site

    def energy_1x1_lowmem(self, state, env_c4v, force_cpu=False):
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

        Due to C4v symmetry it is enough to construct two reduced density matrices.
        In particular, :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1` of a NN-neighbour pair
        and :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1_diag` of NNN-neighbour pair. 
        Afterwards, the energy per site `e` is computed by evaluating a term :math:`h2_rot`
        containing :math:`\bf{S}.\bf{S}` for nearest- and :math:`h2` term for 
        next-nearest- expectation value as:

        .. math::

            e = 2*\langle \mathcal{h2} \rangle_{NN} + 2*\langle \mathcal{h2} \rangle_{NNN}
            = 2*Tr(\rho_{2x1} \mathcal{h2_rot}) + 2*Tr(\rho_{2x1_diag} \mathcal{h2})
        
        """
        rdm2x2_NN= rdm_c4v.rdm2x2_NN_lowmem_sl(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= 2.0*self.j1*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.SS_delta_zz_rot)\
            - 0.5*self.hz_stag * torch.einsum('ijkl,ijkl',rdm2x2_NN,self.hz_2x1_rot)
        if abs(self.j2)>0:
            rdm2x2_NNN= rdm_c4v.rdm2x2_NNN_lowmem_sl(state, env_c4v, sym_pos_def=True,\
                force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
            energy_per_site= energy_per_site \
                + 2.0*self.j2*torch.einsum('ijkl,ijkl',rdm2x2_NNN,self.SS)
        if abs(self.j3)>0:
            rdm3x1= rdm_c4v.rdm3x1_sl(state,env_c4v,sym_pos_def=True,\
                force_cpu=force_cpu,verbosity=cfg.ctm_args.verbosity_rdm)
            ss_3x1= torch.einsum('ijab,ijab',rdm3x1,self.SS)
            energy_per_site= energy_per_site + 2*self.j3*ss_3x1
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def energy_1x1_tiled(self, state, env_c4v, force_cpu=False):
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

        Due to C4v symmetry it is enough to construct two reduced density matrices.
        In particular, :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1` of a NN-neighbour pair
        and :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1_diag` of NNN-neighbour pair. 
        Afterwards, the energy per site `e` is computed by evaluating a term :math:`h2_rot`
        containing :math:`\bf{S}.\bf{S}` for nearest- and :math:`h2` term for 
        next-nearest- expectation value as:

        .. math::

            e = 2*\langle \mathcal{h2} \rangle_{NN} + 2*\langle \mathcal{h2} \rangle_{NNN}
            = 2*Tr(\rho_{2x1} \mathcal{h2_rot}) + 2*Tr(\rho_{2x1_diag} \mathcal{h2})
        
        """
        rdm2x2_NN= rdm2x2_NN_tiled(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= 2.0*self.j1*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.SS_delta_zz_rot)\
            - 0.5*self.hz_stag * torch.einsum('ijkl,ijkl',rdm2x2_NN,self.hz_2x1_rot)
        if abs(self.j2)>0:
            rdm2x2_NNN= rdm2x2_NNN_tiled(state, env_c4v, sym_pos_def=True,\
                force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
            energy_per_site= energy_per_site \
                + 2.0*self.j2*torch.einsum('ijkl,ijkl',rdm2x2_NNN,self.SS)
        if abs(self.j3)>0:
            rdm3x1= rdm_c4v.rdm3x1_sl(state,env_c4v,sym_pos_def=True,\
                force_cpu=force_cpu,verbosity=cfg.ctm_args.verbosity_rdm)
            ss_3x1= torch.einsum('ijab,ijab',rdm3x1,self.SS)
            energy_per_site= energy_per_site + 2*self.j3*ss_3x1
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def eval_obs(self,state,env_c4v,force_cpu=False):
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
            if abs(self.j3)>0:
                rdm3x1= rdm_c4v.rdm3x1(state,env_c4v,force_cpu=force_cpu,\
                    verbosity=cfg.ctm_args.verbosity_rdm)
                obs[f"SS3x1"]= torch.einsum('ijab,ijab',rdm3x1,self.SS)

            rdm2x1= rdm_c4v.rdm2x1_sl(state,env_c4v,force_cpu=force_cpu,\
                verbosity=cfg.ctm_args.verbosity_rdm)
            SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
            obs[f"SS2x1"]= _cast_to_real(SS2x1)
            # reduce rdm2x1 to 1x1
            rdm1x1= torch.einsum('ijaj->ia',rdm2x1)
            rdm1x1= rdm1x1/torch.trace(rdm1x1)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        if abs(self.j3)>0: obs_labels += [f"SS3x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_obs_tiled(self,state,env_c4v,force_cpu=False):
        obs= dict()
        with torch.no_grad():
            rdm2x1= rdm2x1_tiled(state,env_c4v,force_cpu=force_cpu,\
                verbosity=cfg.ctm_args.verbosity_rdm)
            SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
            obs[f"SS2x1"]= _cast_to_real(SS2x1)

            # reduce rdm2x1 to 1x1
            rdm1x1= torch.einsum('ijaj->ia',rdm2x1)
            rdm1x1= rdm1x1/torch.trace(rdm1x1)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist,canonical=False):
        Sop_zxy= torch.zeros((3,self.phys_dim,self.phys_dim),dtype=self.dtype,device=self.device)
        Sop_zxy[0,:,:]= self.obs_ops["sz"]
        Sop_zxy[1,:,:]= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        Sop_zxy[2,:,:]= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"])

        # compute vector of spontaneous magnetization
        if canonical:
            s_vec_zpm=[]
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label in ["sz","sp","sm"]:
                op= self.obs_ops[label]
                s_vec_zpm.append(torch.trace(rdm1x1@op))
            # 0) transform into zxy basis and normalize
            s_vec_zxy= torch.tensor([s_vec_zpm[0],0.5*(s_vec_zpm[1]+s_vec_zpm[2]),\
                0.5*(s_vec_zpm[1]-s_vec_zpm[2])],dtype=self.dtype,device=self.device)
            s_vec_zxy= s_vec_zxy/torch.norm(s_vec_zxy)
            # 1) build rotation matrix
            R= torch.tensor([[s_vec_zxy[0],-s_vec_zxy[1],0],[s_vec_zxy[1],s_vec_zxy[0],0],[0,0,1]],\
                dtype=self.dtype,device=self.device).t()
            # 2) rotate the vector of operators
            Sop_zxy= torch.einsum('ab,bij->aij',R,Sop_zxy)

        # function generating properly rotated operators on every bi-partite site
        def get_bilat_op(op):
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                return op_rot if r%2==0 else op_0
            return _gen_op

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[0,:,:], \
            get_bilat_op(Sop_zxy[0,:,:]), dist)
        Sx0sxR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[1,:,:], get_bilat_op(Sop_zxy[1,:,:]), dist)
        nSy0SyR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[2,:,:], get_bilat_op(Sop_zxy[2,:,:]), dist)

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
        
        D0DR= corrf_c4v.corrf_2sOH2sOH_E1(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

    def eval_corrf_DD_V(self,state,env_c4v,dist,verbosity=0):
        # function generating properly rotated S.S operator on every bi-partite site
        rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
        # (S.S)_s1s2,s1's2' with rotation applied on "first" spin s1,s1' 
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sOV2sOV_E2(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res

class J1J2_C4V_BIPARTITE_THERMAL(J1J2_C4V_BIPARTITE):

    def __init__(self, j1=1.0, j2=0, j3=0, hz_stag= 0.0, delta_zz=1.0, \
        beta=0., global_args=cfg.global_args):

        self.beta= beta
        super().__init__(j1=j1, j2=j2, j3=j3, hz_stag= hz_stag, delta_zz=delta_zz, \
            global_args=global_args)

    def energy_1x1(self,state,env_c4v,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V_THERMAL
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
        containing two nearest-nighbour terms :math:`\bf{S}.\bf{S}` and two next-nearest 
        neighbour :math:`\bf{S}.\bf{S}`, as:

        .. math::

            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})
        
        """
        rdm2x2= rdm_c4v_thermal.rdm2x2(state,env_c4v,sym_pos_def=True,\
            verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp)
        if abs(self.j3)>0:
            raise NotImplementedError()

        energy_per_site= _cast_to_real(energy_per_site)
        return energy_per_site

    def eval_obs(self,state,env_c4v,force_cpu=False):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V_THERMAL
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
            if abs(self.j3)>0:
                raise NotImplementedError()

            rdm2x1= rdm_c4v_thermal.rdm2x1_sl(state,env_c4v,force_cpu=force_cpu,\
                verbosity=cfg.ctm_args.verbosity_rdm)
            SS2x1= torch.einsum('ijab,ijab',rdm2x1,self.SS_rot)
            obs[f"SS2x1"]= _cast_to_real(SS2x1)

            # reduce rdm2x1 to 1x1
            rdm1x1= torch.einsum('ijaj->ia',rdm2x1)
            rdm1x1= rdm1x1/torch.trace(rdm1x1)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        if abs(self.j3)>0: obs_labels += [f"SS3x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels
