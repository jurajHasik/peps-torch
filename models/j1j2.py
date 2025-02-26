import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v.rdm_c4v_specialized import rdm2x2_NNN_tiled,\
    rdm2x2_NN_tiled, rdm2x1_tiled
from ctm.one_site_c4v import corrf_c4v
from math import sqrt
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

# function generating properly rotated operators on every bi-partite site
def _conjugate_op(op,model):
    rot_op= su2.get_rot_op(model.phys_dim, dtype=model.dtype, device=model.device)
    op_0= op
    op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
    def _gen_op(r):
        return op_rot if r%2==0 else op_0
    return _gen_op

def eval_nnnn_per_site(coord,state,env,obs_ops):
    # Next-to-next nearest neighbour interaction in horizontal(x) and vertical(y) directions
    def _conj_id(op):
        return lambda r: op
    szsz_x = corrf.corrf_1sO1sO(coord, (1, 0), state, env, obs_ops["sz"],
                                _conj_id(obs_ops["sz"]), 2)
    smsp_x = corrf.corrf_1sO1sO(coord, (1, 0), state, env, obs_ops["sm"],
                                _conj_id(obs_ops["sp"]), 2)
    spsm_x = corrf.corrf_1sO1sO(coord, (1, 0), state, env, obs_ops["sp"],
                                _conj_id(obs_ops["sm"]), 2)
    szsz_y = corrf.corrf_1sO1sO(coord, (0, 1), state, env, obs_ops["sz"],
                                _conj_id(obs_ops["sz"]), 2)
    spsm_y = corrf.corrf_1sO1sO(coord, (0, 1), state, env, obs_ops["sp"],
                                _conj_id(obs_ops["sm"]), 2)
    smsp_y = corrf.corrf_1sO1sO(coord, (0, 1), state, env, obs_ops["sm"],
                                _conj_id(obs_ops["sp"]), 2)
    nnnn_per_site= (szsz_x[1] + szsz_y[1] + 0.5 * (spsm_x[1] + spsm_y[1] + smsp_x[1] + smsp_y[1]))
    return nnnn_per_site

class J1J2():
    def __init__(self, j1=1.0, j2=0, j3=0, hz_stag= 0, delta_zz=1, lmbd=0, h_uni= [0,0,0],
        global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j3: next-to-next nearest-neighbour interaction
        :param hz_stag: staggered magnetic field in spin-z direction
        :param delta_zz: easy-axis (nearest-neighbour) anisotropy
        :param global_args: global configuration
        :param lmbd: chiral 4-site (plaquette) interaction
        :type lmbd: float
        :param h_uni: uniform magnetic field in direction [h^z, h^x, h^y]
        :type h_uni: list(float)
        :type j1: float
        :type j2: float
        :type j3: float
        :type hz_stag: float
        :type detla_zz: float
        :type global_args: GLOBALARGS
        
        Build Spin-1/2 :math:`J_1-J_2-J_3-\lambda-h^z_{stag}-h^y` Hamiltonian

        .. math:: 

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j 
                + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
                + J_3\sum_{<<<i,j>>>} \mathbf{S}_i.\mathbf{S}_j
                + i\lambda \sum_p P_p - P^{-1}_p
                - \sum_i (-1) h^z_{stag} S^z_i + \vec{h}_{uni}\cdot\vec{S}_i
        
        on the square lattice. Where 
            * the first sum runs over the pairs of sites `i,j` which are nearest-neighbours (denoted as `<.,.>`), 
            * the second sum runs over pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`), 
            * the third sum runs over pairs of sites `i,j` which are next-to-next nearest-neighbours (denoted as `<<<.,.>>>`),
            * the fourth sum runs over all plaquettes `p`, the chiral term P permutes spins on the plaquette in clockwise order 
              and its inverse P^{-1} in anti-clockwise order,
            * the fifth sum runs over all sites applying staggered field in spin-z direction and uniform
              field in :math:`\vec{h}_uni` direction
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        self.j3=j3
        self.lmbd= lmbd
        self.hz_stag=hz_stag
        self.h_uni=torch.as_tensor(h_uni,device=self.device,dtype=self.dtype)
        self.delta_zz=delta_zz

        if self.lmbd != 0: assert torch.rand(1, dtype=self.dtype).is_complex(),\
            "Invalid dtype: Lambda requires complex numbers"
        if self.h_uni[2] != 0: assert torch.rand(1, dtype=self.dtype).is_complex(),\
            "Invalid dtype: h^y field requires complex numbers"

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= s2.I_N(N=2)
        id3= s2.I_N(N=3)
        expr_kron = 'ij,ab->iajb'

        self.SS_delta_zz= s2.SS(xyz=(delta_zz,1.,1.))
        self.SS= s2.SS()
        h_uni_1x1= torch.einsum('x,xia->ia',self.h_uni,s2.S())
        hz_2x1_nn= torch.einsum(expr_kron,s2.SZ(),s2.I())\
            +torch.einsum(expr_kron,s2.I(),-s2.SZ())
        huni_2x1_nn= torch.einsum(expr_kron,h_uni_1x1,s2.I())\
            +torch.einsum(expr_kron,s2.I(),h_uni_1x1)

        rot_op= s2.BP_rot()
        self.SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS,rot_op).contiguous()
        self.SS_delta_zz_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.SS_delta_zz,rot_op).contiguous()
        self.hz_2x1_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,hz_2x1_nn,rot_op).contiguous()
        self.huni_2x1_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,huni_2x1_nn,rot_op).contiguous()

        h2x2_SS_delta_zz= torch.einsum('ijab,klcd->ijklabcd',self.SS_delta_zz,id2) # nearest neighbours
        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',self.SS,id2) # next-nearest neighbours
        h2x2_hz_stag= torch.einsum('ia,jklbcd->ijklabcd',s2.SZ(),id3)
        h2x2_huni= torch.einsum('ia,jklbcd->ijklabcd', h_uni_1x1 ,id3)
        
        # hp aggregates all terms within plaquette, such that energy-per-site= <h_p>= <H>/number-of-sites
        #
        # 0 1     0 1   0 x   x x   x 1
        # 2 3 ... x x + 2 x + 2 3 + x 3
        def get_hp(coord):
            hp= 0.5*self.j1*(h2x2_SS_delta_zz + h2x2_SS_delta_zz.permute(0,2,1,3,4,6,5,7)\
               + h2x2_SS_delta_zz.permute(2,3,0,1,6,7,4,5) + h2x2_SS_delta_zz.permute(3,1,2,0,7,5,6,4)) \
               + self.j2*(h2x2_SS.permute(0,3,2,1,4,7,6,5) + h2x2_SS.permute(2,1,0,3,6,5,4,7))\
               - 0.25*self.hz_stag*((-1)**(coord[0]+coord[1]))*(h2x2_hz_stag\
                    -h2x2_hz_stag.permute(3,0,1,2, 7,4,5,6)\
                    -h2x2_hz_stag.permute(2,3,0,1, 6,7,4,5) +h2x2_hz_stag.permute(1,2,3,0, 5,6,7,4))\
               + 0.25*(h2x2_huni + h2x2_huni.permute(2,3,0,1, 6,7,4,5)\
                    + h2x2_huni.permute(3,0,1,2, 7,4,5,6) + h2x2_huni.permute(1,2,3,0, 5,6,7,4))
            return hp

        self.get_hp= get_hp

        self.hp_rot= torch.einsum('xj,yk,ixylauvd,ub,vc->ijklabcd',\
            rot_op,rot_op,self.get_hp((0,0)),rot_op,rot_op).contiguous()

        self.chiral_term=self.chiral_term_rot= 0*s2.I_N(N=4)
        self.hp_chiral=self.hp_chiral_rot= self.chiral_term_rot
        if self.phys_dim==2 and self.lmbd!=0:
            # ----- phys_dim= 2 specific code -----
            # chiral term
            # build permutation operator i->j->k->l->i
            P12= torch.as_tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], \
                dtype=self.dtype, device=self.device)
            P12= P12.view(2,2,2,2)

            # 0<->1 , Id, Id
            P12II= torch.einsum('abij,cdkl->abcdijkl',P12, id2)
            PI12I= P12II.permute(3,0,1,2, 7,4,5,6).contiguous()
            PII12= P12II.permute(2,3,0,1, 6,7,4,5).contiguous()
            # Id, Id, 2<-3>
            # Id, 1<->2, Id
            # 0<->1, Id, Id
            P4= torch.tensordot(PI12I, P12II, ([4,5,6,7],[0,1,2,3]))
            P4= torch.tensordot(PII12, P4, ([4,5,6,7],[0,1,2,3]))
            chiral_term= 1.0j*( P4 - P4.view(16,16).t().view(2,2,2,2,2,2,2,2) )

            # spins are ordered as s0 s1 hence, to be compatible with 2x2 RDM permute
            #                      s2 s3
            # 
            # s0 s1 => s0 s1
            # s2 s3    s3 s2
            self.chiral_term= chiral_term.permute(0,1,3,2, 4,5,7,6)
            self.chiral_term_rot= torch.einsum('xj,yk,ixylauvd,ub,vc->ijklabcd',\
                rot_op,rot_op,self.chiral_term,rot_op,rot_op).contiguous()
            self.hp_chiral_rot= self.lmbd*self.chiral_term_rot
        
        self.obs_ops = self.get_obs_ops()

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
        of two tensors A, and B=RA, where R appropriately rotates the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation R => BIPARTITE

            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        A single reduced density matrix :py:func:`ctm.rdm.rdm2x2` of a 2x2 plaquette
        is used to evaluate the energy.
        """
        assert self.h_uni[:2].norm()==0,\
            "only spin-y direction uniform field is compatible with single-site C4v symmetric ansatz"
        tmp_rdm= rdm.rdm2x2((0,0),state,env)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.hp_rot)
        if abs(self.lmbd)>0:
            energy_per_site+= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.hp_chiral_rot)
        if abs(self.j3)>0:
            energy_nnnn_per_site= eval_nnnn_per_site((0,0),state,env,self.obs_ops)
            energy_per_site+= self.j3 * energy_nnnn_per_site
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def energy_per_site(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        Evaluates all non-equivalent energy contributions within unit cell
        by evaluating all terms aggragated within 4-site plaquette operator over 
        non-equivalent plaquettes and all non-equivalent next-to-next nearest neighbour terms.
        """
        energy_plaquettes=0
        energy_nnnn= 0
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env)
            energy_plaquettes += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.get_hp(coord))
            if abs(self.lmbd)>0:
                energy_plaquettes += self.lmbd*torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.chiral_term)
            if abs(self.j3)>0:
                energy_nnnn += self.j3*eval_nnnn_per_site((0,0),state,env,self.obs_ops)
        energy_cell= energy_plaquettes+energy_nnnn
        energy_per_site= _cast_to_real(energy_cell/len(state.sites))
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

        and without assuming any symmetry on the indices of individual tensors following
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
        return self.energy_per_site(state,env)

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
        return self.energy_2x2_2site(state,env)

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
        return self.energy_2x2_2site(state,env)

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
            m &= \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
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

    def eval_corrf_SS(self,coord,direction,state,env,dist,conjugate=False):
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
        :param conjugate: conjugate operator by sublattice rotation :math:`-i\sigma^y` on every
                          sublattice-B site of bipartite lattice
        :type conjugate: bool
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        conjugate_op= (lambda op: _conjugate_op(op,self)) if conjugate else lambda op: (lambda r: op) 

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res  

    def eval_corrf_SpSm(self,coord,direction,state,env,dist,conjugate=False):
        r"""
        :return: dictionary with correlation functions
        :rtype: dict(str: torch.Tensor)

        Evaluates :math:`\langle S^+(0)S^-(r) \rangle` and :math:`\langle S^-(0)S^+(r) \rangle`
        correlation functions.

        See :meth:`eval_corrf_SS`.
        """
        op_sp = self.obs_ops["sp"]
        op_sm = self.obs_ops["sm"]
   
        conjugate_op= (lambda op: _conjugate_op(op,self)) if conjugate else lambda op: (lambda r: op)

        Sp0smR = corrf.corrf_1sO1sO(coord,direction,state,env, op_sp, conjugate_op(op_sm), dist)
        Sm0spR = corrf.corrf_1sO1sO(coord,direction,state,env, op_sm, conjugate_op(op_sp), dist)

        res= dict({"spsm": Sp0smR, "smsp": Sm0spR})
        return res


class J1J2_C4V_BIPARTITE(J1J2):
    def __init__(self, j1=1.0, j2=0, j3=0, hz_stag= 0, delta_zz=1, lmbd=0, h_uni= [0,0,0],
        global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j3: next-to-next nearest-neighbour interaction
        :param hz_stag: staggered magnetic field in spin-z direction
        :param delta_zz: easy-axis (nearest-neighbour) anisotropy
        :param global_args: global configuration
        :param lmbd: chiral 4-site (plaquette) interaction
        :type lmbd: float
        :param h_uni: uniform magnetic field in direction [h^z, h^x, h^y]
        :type h_uni: list(float)
        :type j1: float
        :type j2: float
        :type j3: float
        :type hz_stag: float
        :type detla_zz: float
        :type global_args: GLOBALARGS
        
        See :class:`J1J2`.

        .. math::

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
              + J_3\sum_{<<<i,j>>>} \mathbf{S}_i.\mathbf{S}_j
        
        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`), and 
        the last sum runs over pairs of sites `i,j` which are next-to-next nearest-neighbours 
        (denoted as `<<<.,.>>>`).
        
        .. note::
            Unifrom field in z-, and x-directions has no effect on single-site C4v symmetric state
            by construction. Uniform magnetization in y-direction is prohibited by projection
            of the on-site tensor to :math:`A_1 + iA_2` irrep.
            
        """
        # where
        # * :math:`h_p = J_1(S^x_{r}.S^x_{r+\vec{x}} 
        #   + S^y_{r}.S^y_{r+\vec{x}} + \delta_{zz} S^z_{r}.S^z_{r+\vec{x}} + (x<->y))
        #   + J_2(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}+\vec{y}} + \mathbf{S}_{r+\vec{x}}.\mathbf{S}_{r+\vec{y}})
        #   + h_stag (S^z_{r} - S^z_{r+\vec{x}} - S^z_{r+\vec{y}} + S^z_{r+\vec{x}+\vec{y}})` 
        #   with indices of spins ordered as follows :math:`s_r s_{r+\vec{x}} s_{r+\vec{y}} s_{r+\vec{x}+\vec{y}};
        #   s'_r s'_{r+\vec{x}} s'_{r+\vec{y}} s'_{r+\vec{x}+\vec{y}}`
        super().__init__(j1=j1, j2=j2, j3=j3, hz_stag=hz_stag, delta_zz=delta_zz, lmbd=lmbd, 
            h_uni= h_uni, global_args=global_args)
        self.obs_ops= self.get_obs_ops()

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v,force_cpu=False,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
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

        Due to C4v symmetry it is enough to construct just one or two different reduced
        density matrices to evaluate energy per site. 

        In the case of :math:`J_3=0`, it is sufficient to only consider :meth:`ctm.one_site_c4v.rdm_c4v.rdm2x2` 
        of a 2x2 plaquette. Afterwards, the energy per site `e` is computed by evaluating a plaquette term 
        :math:`h_p` containing two nearest-nighbour terms :math:`\bf{S}.\bf{S}` and two next-nearest 
        neighbour :math:`\bf{S}.\bf{S}`, as:
        
        .. math::
            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})
        
        If :math:`J_3 \neq 0`, additional reduced density matrix :meth:`ctm.one_site_c4v.rdm_c4v.rdm3x1`
        is constructed to evaluate next-to-next nearest neighbour interaction.
        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,sym_pos_def=True,\
            verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp_rot)
        if abs(self.lmbd)>0:
            energy_per_site+= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp_chiral_rot)
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
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float

        Analogous to :meth:`energy_1x1`. However, the evaluation of energy is realized
        by individually constructing low-memory versions of reduced density matrices for
        nearest (NN), next-nearest (NNN), and next-to-next nearest neighbours (NNNN). In particular:

            * NN: :meth:`ctm.one_site_c4v.rdm_c4v.rdm2x2_NN_lowmem_sl`
            * NNN: :meth:`ctm.one_site_c4v.rdm_c4v.rdm2x2_NNN_lowmem_sl`
            * NNNN: :meth:`ctm.one_site_c4v.rdm_c4v.rdm3x1_sl`
        """
        assert self.lmbd==0,"energy_1x1_lowmem does not account for lambda term"
        rdm2x2_NN= rdm_c4v.rdm2x2_NN_lowmem_sl(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= 2.0*self.j1*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.SS_delta_zz_rot)\
            - 0.5*self.hz_stag * torch.einsum('ijkl,ijkl',rdm2x2_NN,self.hz_2x1_rot)
        if abs(self.h_uni.norm())>0:
            energy_per_site+= 0.5*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.huni_2x1_rot)
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
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Analogous to :meth:`energy_1x1`. However, the evaluation of energy is realized
        by individually constructing low-memory tiled versions of reduced density matrices for
        nearest (NN), next-nearest (NNN), and next-to-next nearest neighbours (NNNN). 
        In particular:

            * NN: :meth:`ctm.one_site_c4v.rdm_c4v_specialized.rdm2x2_NN_tiled`
            * NNN: :meth:`ctm.one_site_c4v.rdm_c4v_specialized.rdm2x2_NNN_tiled`
            * NNNN: :meth:`ctm.one_site_c4v.rdm_c4v.rdm3x1_sl`
        """
        assert self.lmbd==0,"energy_1x1_lowmem does not account for lambda term"
        rdm2x2_NN= rdm2x2_NN_tiled(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= 2.0*self.j1*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.SS_delta_zz_rot)\
            - 0.5*self.hz_stag * torch.einsum('ijkl,ijkl',rdm2x2_NN,self.hz_2x1_rot)
        if abs(self.h_uni.norm())>0:
            energy_per_site+= 0.5* torch.einsum('ijkl,ijkl',rdm2x2_NN,self.huni_2x1_rot)
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
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]
        
        Computes the following observables in order
        
            1. magnetization
            2. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle`
            3. :math:`\langle S.S \rangle_{NN}`, (optionally) :math:`\langle S.S \rangle_{NNNN}`
    
        where the on-site magnetization is defined as
        
        .. math::
        
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }

        """
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict()
        with torch.no_grad():
            if abs(self.j3)>0:
                rdm3x1= rdm_c4v.rdm3x1(state,env_c4v,force_cpu=force_cpu,\
                    verbosity=cfg.ctm_args.verbosity_rdm)
                obs[f"SS3x1"]= torch.einsum('ijab,ijab',rdm3x1,self.SS)

            if abs(self.lmbd)>0:
                rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,force_cpu=force_cpu,\
                    verbosity=cfg.ctm_args.verbosity_rdm)
                obs[f"ChiralT"]= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.chiral_term_rot)

            if abs(self.j2)>0:
                rdm2x2diag= rdm_c4v.rdm2x2_NNN_lowmem_sl(state, env_c4v,\
                force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
                obs[f"SS_nnn"]= torch.einsum('ijab,ijab',rdm2x2diag,self.SS)

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
        if abs(self.j2)>0: obs_labels += [f"SS_nnn"]
        if abs(self.j3)>0: obs_labels += [f"SS3x1"]
        if abs(self.lmbd)>0: obs_labels += [f"ChiralT"]
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

    def eval_corrf_SS(self,state,env_c4v,dist,canonical=False,rl_0=None):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param dist: maximal distance of correlator
        :type dist: int
        :param canonical: decompose correlations wrt. to vector of spontaneous magnetization
                          into longitudinal and transverse parts
        :type canonical: bool
        :param rl_0: right and left leading eigenvector of width-1 transfer matrix
                     as obtained from :meth:`ctm.one_site_c4v.transferops_c4v.get_Top_spec_c4v`.
        :type rl_0: tuple(torch.Tensor) 
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` .
        """
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
            rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                return op_rot if r%2==0 else op_0
            return _gen_op

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[0,:,:], \
            get_bilat_op(Sop_zxy[0,:,:]), dist, rl_0=rl_0)
        Sx0sxR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[1,:,:], get_bilat_op(Sop_zxy[1,:,:]), \
            dist, rl_0=rl_0)
        nSy0SyR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[2,:,:], get_bilat_op(Sop_zxy[2,:,:]), \
            dist, rl_0=rl_0)

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

    def eval_corrf_DD_V(self,state,env_c4v,dist,verbosity=0):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_C4V
        :type env_c4v: ENV_C4V
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with vertical dimer-dimer correlation function
        :rtype: dict(str: torch.Tensor)
        
        Evaluate vertical dimer-dimer correlation functions 

        .. math::
            \langle(\mathbf{S}(r+1,1).\mathbf{S}(r+1,0))(\mathbf{S}(0,1).\mathbf{S}(0,0))\rangle 

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
        
        D0DR= corrf_c4v.corrf_2sOV2sOV_E2(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res