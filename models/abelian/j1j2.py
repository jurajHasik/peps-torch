from math import sqrt
import itertools
import config as cfg
import yamps.tensor as TA
from tn_interface_abelian import contract, permute
import groups.su2_abelian as su2
from ctm.generic_abelian import rdm
#from ctm.generic import corrf

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
        assert settings.nsym==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.backend= settings.back
        self.dtype=settings.dtype
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

    def energy_2x2_1site_BP(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
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
        if not (hasattr(self, 'h2x2_nn_rot') or hasattr(self, 'h2x2_nn_nrot')):
            irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
            rot_op= irrep.BP_rot()
            self.h2x2_nn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
                self.h2x2_nn,rot_op,rot_op,rot_op,rot_op)
            self.h2x2_nnn_rot= torch.einsum('irtlaxyd,jr,kt,xb,yc->ijklabcd',\
                self.h2x2_nnn,rot_op,rot_op,rot_op,rot_op)

        tmp_rdm= rdm.rdm2x2((0,0),state,env)
        energy_nn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn_rot)
        energy_nnn= torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn_rot)
        energy_per_site = 2.0*(self.j1*energy_nn/4.0 + self.j2*energy_nnn/2.0)

        return energy_per_site

    def energy_2x1_or_2Lx2site_2x2rdms(self,state,env):
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
        N= state.lX*state.lY
        assert N==len(state.sites), "size of the unit cell does not match number of sites"

        energy_nn=TA.zeros(self.engine)
        energy_nnn=TA.zeros(self.engine)
        _ci= ([0,1,2,3, 4,5,6,7],[4,5,6,7, 0,1,2,3])
        for coord in state.sites.keys():
            tmp_rdm= rdm.rdm2x2(coord,state,env).to_dense()
            energy_nn += contract(tmp_rdm,self.h2x2_nn,_ci)
            energy_nnn += contract(tmp_rdm,self.h2x2_nnn,_ci)
        energy_per_site = 2.0*(self.j1*energy_nn/(4*N) + self.j2*energy_nnn/(2*N))

        return energy_per_site

    def eval_obs(self,state,env):
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
        for coord,site in state.sites.items():
            rdm1x1 = rdm.rdm1x1(coord,state,env).to_dense()
            for label,op in self.obs_ops.items():
                obs[f"{label}{coord}"]= contract(rdm1x1, op, ([0,1],[1,0])).to_number()
            obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
            obs["avg_m"] += obs[f"m{coord}"]
        obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

        _ci= ([0,1,2,3],[2,3,0,1])
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env).to_dense()
            rdm1x2 = rdm.rdm1x2(coord,state,env).to_dense()
            obs[f"SS2x1{coord}"]= contract(rdm2x1,self.h2,_ci).to_number()
            obs[f"SS1x2{coord}"]= contract(rdm1x2,self.h2,_ci).to_number()
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    # def eval_corrf_SS(self,coord,direction,state,env,dist):
   
    #     # function allowing for additional site-dependent conjugation of op
    #     def conjugate_op(op):
    #         #rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
    #         rot_op= torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
    #         op_0= op
    #         op_rot= einsum('ki,kj->ij',rot_op, mm(op_0,rot_op))
    #         def _gen_op(r):
    #             #return op_rot if r%2==0 else op_0
    #             return op_0
    #         return _gen_op

    #     op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
    #     op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

    #     Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
    #         conjugate_op(self.obs_ops["sz"]), dist)
    #     Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
    #     nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

    #     res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
    #     return res  