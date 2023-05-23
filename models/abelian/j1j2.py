from math import sqrt
import numpy as np
import itertools
import config as cfg
import yastn.yastn as yastn
from tn_interface_abelian import contract, permute
import groups.su2_abelian as su2
from ctm.generic_abelian import rdm
from ctm.generic_abelian import corrf
from ctm.one_site_c4v_abelian import rdm_c4v
from ctm.one_site_c4v_abelian import corrf_c4v


class J1J2_NOSYM():
    def __init__(self, settings, j1=1.0, j2=0.0, global_args=cfg.global_args):
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
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.h2, self.h2x2_nn, self.h2x2_nnn= self.get_h()
        self.obs_ops= self.get_obs_ops()

    def get_h(self):
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        I1= irrep.I()
        SS= irrep.SS()
        
        # 0        0->2                 0     1 
        # I1--(x)--I1   => transpose => I1----I1 
        # 1        1->3                 2     3
        I2= contract(I1,I1,([],[])) 
        I2= permute(I2, (0,2,1,3))
        
        # 0 1      0 1->4 5                  0 1   2 3 
        # SS--(x)--I2        => transpose => SS----I2 
        # 2 3      2 3->6 7                  4 5   6 7
        h2x2_SS= contract(SS,I2,([],[]))
        h2x2_SS= permute(h2x2_SS, (0,1,4,5, 2,3,6,7))

        h2x2_nn= h2x2_SS + permute(h2x2_SS, (2,3,0,1,6,7,4,5)) + permute(h2x2_SS, (0,2,1,3,4,6,5,7))\
            + permute(h2x2_SS, (2,0,3,1,6,4,7,5))
        h2x2_nnn= permute(h2x2_SS, (0,3,2,1,4,7,6,5)) + permute(h2x2_SS, (2,0,1,3,6,4,5,7))
        
        return SS, h2x2_nn, h2x2_nnn

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        obs_ops["sz"]= irrep.SZ()
        obs_ops["sp"]= irrep.SP()
        obs_ops["sm"]= irrep.SM()
        return obs_ops

    # def energy_2x2_1site_BP(self,state,env):
    #     r"""
    #     :param state: wavefunction
    #     :param env: CTM environment
    #     :type state: IPEPS_ABELIAN
    #     :type env: ENV_ABELIAN
    #     :return: energy per site
    #     :rtype: float

    #     We assume 1x1 iPEPS which tiles the lattice with a bipartite pattern composed 
    #     of two tensors A, and B=RA, where R rotates approriately the physical Hilbert space 
    #     of tensor A on every "odd" site::

    #         1x1 C4v => rotation P => BIPARTITE

    #         A A A A                  A B A B
    #         A A A A                  B A B A
    #         A A A A                  A B A B
    #         A A A A                  B A B A

    #     A single reduced density matrix :py:func:`ctm.rdm.rdm2x2` of a 2x2 plaquette
    #     is used to evaluate the energy.
    #     """
    #     if not (hasattr(self, 'h2x2_nn_rot') or hasattr(self, 'h2x2_nn_nrot')):
    #         irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
    #         rot_op= irrep.BP_rot()
    #         self.h2x2_nn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
    #             self.h2x2_nn,rot_op,rot_op,rot_op,rot_op)
    #         self.h2x2_nnn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
    #             self.h2x2_nnn,rot_op,rot_op,rot_op,rot_op)

    #     tmp_rdm= rdm.rdm2x2((0,0),state,env)
    #     energy_nn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn_rot)
    #     energy_nnn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn_rot)
    #     energy_per_site = 2.0*(self.j1*energy_nn/4.0 + self.j2*energy_nnn/2.0)

    #     return energy_per_site

    def energy_2x1_or_2Lx2site_2x2rdms(self,state,env,**kwargs):
        r"""

        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :return: energy per site
        :rtype: float

        Covered cases:
    
        1) Assume iPEPS with 2x1 unit cell containing two tensors A, B. We can
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

        All the above terms can be captured by evaluating correlations over two 
        reduced density matrices if 2x2 cluster::
        
            A3--1B   B3  1A
            2 \/ 2   2 \/ 2
            0 /\ 0   0 /\ 0
            B3--1A & A3  1B

            A3--1B   B3--1A
            2 \/ 2   2 \/ 2
            0 /\ 0   0 /\ 0
            A3--1B & B3--1A

        2) We assume iPEPS with 2Lx2 unit cell. Simplest example would be a 2x2 unit cell 
        containing four tensors A, B, C, and D with simple PBC tiling::

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

        A more complex example 4x2 unit cell containing eight tensors A, B, C, D, 
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
        N= len(state.sites)

        energy_nn=yastn.zeros(self.engine)
        energy_nnn=yastn.zeros(self.engine)
        _ci= ([0,1,2,3, 4,5,6,7],[4,5,6,7, 0,1,2,3])
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env).to_nonsymmetric()
            energy_nn += contract(tmp_rdm,self.h2x2_nn,_ci)
            energy_nnn += contract(tmp_rdm,self.h2x2_nnn,_ci)
        energy_per_site= 2.0*(self.j1*energy_nn/(4*N) + self.j2*energy_nnn/(2*N))
        energy_per_site= rdm._cast_to_real(energy_per_site,**kwargs)

        return energy_per_site

    def eval_obs(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
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
        _ci= ([0,1],[1,0])
        for coord,site in state.sites.items():
            rdm1x1 = rdm.rdm1x1(coord,state,env).to_nonsymmetric()
            for label,op in self.obs_ops.items():
                obs[f"{label}{coord}"]= contract(rdm1x1, op, _ci).to_number()
            obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
            obs["avg_m"] += obs[f"m{coord}"]
        obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

        _ci= ([0,1,2,3],[2,3,0,1])
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env).to_nonsymmetric()
            rdm1x2 = rdm.rdm1x2(coord,state,env).to_nonsymmetric()
            SS2x1= contract(rdm2x1,self.h2,_ci)
            SS1x2= contract(rdm1x2,self.h2,_ci)
            obs[f"SS2x1{coord}"]= rdm._cast_to_real(SS2x1,**kwargs).to_number()
            obs[f"SS1x2{coord}"]= rdm._cast_to_real(SS1x2,**kwargs).to_number()
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,coord,direction,state,env,dist,rl_0=None):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :param dist: maximal distance of correlator
        :type dist: int
        :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
        :type rl_0: tuple(function(tuple(int,int))->yastn.Tensor, function(tuple(int,int))->yastn.Tensor)
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: np.ndarray)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        # function allowing for additional site-dependent conjugation of op
        # r=0 is nearest-neighbour
        def conjugate_op(op):
            def _gen_op(r):
                return op
            return _gen_op

        s2_U1= su2.SU2_U1(state.engine, 2)

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SZ(), \
            conjugate_op(s2_U1.SZ()), dist, rl_0=rl_0)
        Sp0smR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SP(), conjugate_op(s2_U1.SM()),\
            dist, rl_0=rl_0)
        Sm0SpR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SM(), conjugate_op(s2_U1.SP()),\
            dist, rl_0=rl_0)

        res= dict({"ss": Sz0szR+0.5*(Sp0smR+Sm0SpR), "szsz": Sz0szR, "spsm": Sp0smR, "smsp": Sm0SpR})
        return res

class J1J2_C4V_BIPARTITE_NOSYM():
    def __init__(self, settings, j1=1.0, j2=0.0, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: 

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
            = \sum_{p} h_p

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`).

        The plaquette term is defined as

        * :math:`h_p = J_1(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}} + \mathbf{S}_{r}.\mathbf{S}_{r+\vec{y}})
          +J_2(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}+\vec{y}} + \mathbf{S}_{r+\vec{x}}.\mathbf{S}_{r+\vec{y}})` 
          with indices of spins ordered as follows :math:`s_r s_{r+\vec{x}} s_{r+\vec{y}} s_{r+\vec{x}+\vec{y}};
          s'_r s'_{r+\vec{x}} s'_{r+\vec{y}} s'_{r+\vec{x}+\vec{y}}`

        """
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.SS, self.SS_rot, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        I1= irrep.I()
        SS= irrep.SS()
        rot_op= irrep.BP_rot()
        
        #      1                    0       1
        #      R                    --SS_nn--
        #      0                    2       3
        # 0->1 1    1->0    0->1            0
        # --SS-- => --SS_nn--    =>         R
        # 2    3    2       3               1->3  
        SS_nn= contract(rot_op,SS,([0],[1]))
        SS_nn= permute(SS_nn,(1,0,2,3))
        SS_nn= contract(SS_nn,rot_op.conj(),([3],[0]))

        # 0        0->2                 0     1 
        # I1--(x)--I1   => transpose => I1----I1 
        # 1        1->3                 2     3
        I2= contract(I1,I1,([],[]))
        I2= permute(I2, (0,2,1,3))
        I2_nn= contract(rot_op,I2,([0],[1]))
        I2_nn= permute(I2_nn,(1,0,2,3))
        I2_nn= contract(I2_nn,rot_op.conj(),([3],[0]))

        # 0   1       0 1->4 5                  0 1      2 3 
        # SS_nn--(x)--I2_nn     => transpose => SS_nn----I2_nn
        # 2   3       2 3->6 7                  4 5      6 7
        h2x2_SSnn= contract(SS_nn,I2_nn,([],[]))
        h2x2_SSnn= permute(h2x2_SSnn, (0,1,5,4, 2,3,7,6))
        
        h2x2_SSnnn= contract(SS,I2.flip_signature(),([],[]))
        h2x2_SSnnn= permute(h2x2_SSnnn, (0,1,4,5, 2,3,6,7))

        # all nearest-neighbour S.S terms on 2x2 plaquette
        #        S  SR        x  xR                                S  xR     x  SR
        #        xR x         SR S                                 SR x      xR S 
        h2x2_nn= h2x2_SSnn + permute(h2x2_SSnn, (3,2,1,0,7,6,5,4)) + \
            permute(h2x2_SSnn, (0,2,1,3,4,6,5,7)) + permute(h2x2_SSnn, (3,1,2,0,7,5,6,4))
        # all next nearest-neighbour S.S terms on 2x2 plaquette
        #                 S  xR                                    x  SR
        #                 xR S                                     SR x    
        h2x2_nnn= permute(h2x2_SSnnn, (0,3,2,1,4,7,6,5)) + \
            permute(h2x2_SSnnn.flip_signature(), (2,0,1,3,6,4,5,7))
        h2x2= 0.5*self.j1*h2x2_nn + self.j2*h2x2_nnn

        return SS, SS_nn, h2x2

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        obs_ops["sz"]= irrep.SZ()
        obs_ops["sp"]= irrep.SP()
        obs_ops["sm"]= irrep.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v,force_cpu=False,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_ABELIAN_C4V
        :type env_c4v: ENV_ABELIAN_C4V
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
        _ci= ([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])
        # _ci= ([0,1,2,3,4,5,6,7], [4,5,6,7,0,1,2,3])
        rdm2x2= rdm_c4v.rdm2x2(state, env_c4v, sym_pos_def=False,\
            verbosity=cfg.ctm_args.verbosity_rdm, force_cpu=force_cpu).to_nonsymmetric()
        energy_per_site= contract(rdm2x2,self.hp,_ci)
        energy_per_site= rdm._cast_to_real(energy_per_site,**kwargs).to_number()
        return energy_per_site

    def energy_1x1_lowmem(self,state,env_c4v,force_cpu=False,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_ABELIAN_C4V
        :type env_c4v: ENV_ABELIAN_C4V
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
        # _ci= ([0,1,2,3],[2,3,0,1])
        _ci= ([0,1,2,3],[0,1,2,3])
        rdm2x2_NN= rdm_c4v.rdm2x2_NN(state, env_c4v, sym_pos_def=False,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm).to_nonsymmetric()
        rdm2x2_NNN= rdm_c4v.rdm2x2_NNN(state, env_c4v, sym_pos_def=False,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm).to_nonsymmetric()
        SS_nn= contract(rdm2x2_NN,self.SS_rot,_ci)
        SS_nnn= contract(rdm2x2_NNN,self.SS,_ci)
        energy_per_site= 2.0*self.j1*SS_nn + 2.0*self.j2*SS_nnn
        energy_per_site= rdm._cast_to_real(energy_per_site,**kwargs).to_number()
        return energy_per_site

    def eval_obs(self,state,env_c4v,force_cpu=False,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS_ABELIAN_C4V
        :type env_c4v: ENV_ABELIAN_C4V
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
        # _ci= ([0,1,2,3],[2,3,0,1])
        _ci= ([0,1,2,3],[0,1,2,3])
        rdm2x1= rdm_c4v.rdm2x1(state,env_c4v,force_cpu=force_cpu,\
            verbosity=cfg.ctm_args.verbosity_rdm).to_nonsymmetric()
        # if np.all(np.asarray(rdm2x1.s)/np.asarray(self.SS_rot.s)==-1):
        #     rdm2x1= rdm2x1.flip_signature()
        obs[f"SS2x1"]= contract(rdm2x1,self.SS_rot,_ci).to_number()
        
        # TODO reduce rdm2x1 to 1x1
        # _ci= ([0,1],[1,0])
        _ci= ([0,1],[0,1])
        rdm1x1 = rdm_c4v.rdm1x1(state,env_c4v,force_cpu=force_cpu,\
            verbosity=cfg.ctm_args.verbosity_rdm).to_nonsymmetric()
        # if np.all(np.asarray(rdm1x1.s)/np.asarray(self.obs_ops["sz"].s)==-1):
        #     rdm1x1= rdm1x1.flip_signature()
        for label,op in self.obs_ops.items():
            obs[f"{label}"]= contract(rdm1x1, op, _ci).to_number()
        obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
        
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist):
        import examples.abelian.settings_U1_torch as settings_U1
        irrep = su2.SU2_U1(settings_U1, self.phys_dim-1)
        
        # manually define the rotated operators, since rotation operator
        # cannot be expressed as U(1)-symmetric tensors
        def get_bilat_op_SZ():
            def _gen_op(r):
                return (-1)*irrep.SZ().flip_signature() if r%2==0 else irrep.SZ()
            return _gen_op

        def get_bilat_op_SP():
            def _gen_op(r):
                return (-1)*irrep.SM().flip_signature() if r%2==0 else irrep.SP()
            return _gen_op

        def get_bilat_op_SM():
            def _gen_op(r):
                return (-1)*irrep.SP().flip_signature() if r%2==0 else irrep.SM()
            return _gen_op

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, irrep.SZ(), \
            get_bilat_op_SZ(), dist)
        Sp0smR= corrf_c4v.corrf_1sO1sO(state, env_c4v, irrep.SP(), get_bilat_op_SM(), dist)
        Sm0SpR= corrf_c4v.corrf_1sO1sO(state, env_c4v, irrep.SM(), get_bilat_op_SP(), dist)

        res= dict({"ss": Sz0szR+0.5*(Sp0smR+Sm0SpR), "szsz": Sz0szR, "spsm": Sp0smR, "smsp": Sm0SpR})
        return res