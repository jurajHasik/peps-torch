import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from math import sqrt
from tn_interface import einsum, mm
from tn_interface import view, permute, contiguous
import itertools

class BaTiOCu2Po44():
    def __init__(self, j1=1.0, j2=0.0, jp2=0.0, jp11=0.0, jp12=0.0, \
        global_args=cfg.global_args):
        r"""
        :param j1: nearest neighbour interaction (strong plaquettes)
        :param j2: next-nearest neighbour interaction (strong plaquettes)
        :param jp2: next-nearest neighbour interaction (weak plaquettes)
        :param jp11: nearest neighbour interaction (weak plaquettes)
        :param jp12: nearest neighbour interaction (weak plaquettes)
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type jp2: float
        :type jp11: float
        :type jp12: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`baTiOCu2Po44` Hamiltonian on the square lattice.
        Using 8-site unit cell - containing 8 plaquettes::
   
            |0 |1 |2 |3       |      |      |      |
            A--B--C--D--     (0,0)--(1,0)--(2,0)--(3,0)--
            |4 |5 |6 |7  <=>  |      |      |      |     
            E--F--G--H--     (1,0)--(1,1)--(2,1)--(3,1)--

        Heisenberg couplings (:math:`\vec{S}\cdot\vec{S}`)::

                                                        \/    \/             
                                                        /\    /\             
                o--o  o--o         o  o  o  o          o  o  o  o           
            J1  |  |  |  |  +  J2   \/    \/   +  Jp2      \/    \/  + 
                |  |  |  |          /\    /\               /\    /\   
                o--o  o--o         o  o  o  o          o  o  o  o   

                     |        |               |  |
                     |        |               |  |
                     o  o--o  o            o  o  o  o--
            + Jp11                + Jp12  
                     o  o  o  o--          o  o--o  o

            and DMI terms 

            o-^-o  o-v-o
            |   |  |   | 
            <   >  >   <
            |   |  |   |
            o-v-o  o-^-o

        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        self.jp2=jp2
        self.jp11=jp11
        self.jp12=jp12

        self.SS, self.SS_nn, self.SS_nnn, self.plq= self.get_h()
        self.obs_ops= self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(4, dtype=self.dtype, device=self.device)
        id2= view(id2, (2,2,2,2))
        expr_kron = 'ij,ab->iajb'
        SS= einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(einsum(expr_kron,s2.SP(),s2.SM()) \
            + einsum(expr_kron,s2.SM(),s2.SP()))
        SS= contiguous(SS)
        
        # o--o
        # 
        # o  o
        h2x2_SS= einsum('ijab,klcd->ijklabcd',SS,id2)
        
        # o--o   o--o      o  o     o  o     o  o  
        # |  | =         +       +  |     +     |
        # o--o   o  o      o--o     o  o     o  o
        h2x2_nn= h2x2_SS + permute(h2x2_SS, (2,3,0,1,6,7,4,5)) \
            + permute(h2x2_SS, (0,2,1,3,4,6,5,7)) + permute(h2x2_SS, (2,0,3,1,6,4,7,5))

        # o  o
        #  \/
        #  /\
        # o  o
        h2x2_nnn= permute(h2x2_SS, (0,3,2,1,4,7,6,5)) + permute(h2x2_SS, (2,0,1,3,6,4,5,7))
        h2x2_nn= contiguous(h2x2_nn)
        h2x2_nnn= contiguous(h2x2_nnn)

        plq_op= torch.zeros([8]+[self.phys_dim]*8, dtype=self.dtype, device=self.device)
        # build plaquette operators
        # 
        # o--o
        # |\/|
        # |/\|
        # o--o for plaquettes 4,6
        plq_op[4,:,:,:,:, :,:,:,:]= self.j1 * h2x2_nn + self.j2 * h2x2_nnn
        plq_op[6,:,:,:,:, :,:,:,:]= self.j1 * h2x2_nn + self.j2 * h2x2_nnn
        
        # o--o
        #  \/
        #  /\
        # o--o for plaquettes 5,7
        plq_op[5,:,:,:,:, :,:,:,:]= self.jp2 * h2x2_nnn + self.jp11 * h2x2_SS \
            + self.jp12 * permute(h2x2_SS, (2,3,0,1,6,7,4,5))
        plq_op[7,:,:,:,:, :,:,:,:]= self.jp2 * h2x2_nnn + self.jp12 * h2x2_SS \
            + self.jp11 * permute(h2x2_SS, (2,3,0,1,6,7,4,5))

        # o  o
        # |\/|
        # |/\|
        # o  o for plaquettes 0,2
        plq_op[0,:,:,:,:, :,:,:,:]= self.jp2 * h2x2_nnn \
            + self.jp11 * permute(h2x2_SS, (0,2,1,3,4,6,5,7)) \
            + self.jp12 * permute(h2x2_SS, (2,0,3,1,6,4,7,5))
        plq_op[2,:,:,:,:, :,:,:,:]= self.jp2 * h2x2_nnn \
            + self.jp12 * permute(h2x2_SS, (0,2,1,3,4,6,5,7)) \
            + self.jp11 * permute(h2x2_SS, (2,0,3,1,6,4,7,5))

        return SS, h2x2_nn, h2x2_nnn, plq_op

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

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

              A B C D A B C D
              E F G H E F G H
          A B C D A B C D
          E F G H E F G H
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of eight :math:`\rho_{2x2}`'s are needed over which operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            (4)      (5)      (6)      (7)
 
            A3--1B   B3--1C   C3--1D   D3--1A
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            E3--1F & F3--1G & G3--1H & H3--1C 

            (0)

            C3--1D   D3--1G   G3--1H   H3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            B3--1E & E3--1F & F3--1A & A3--1B 
        """
        energy= 0
        rdm4= rdm.rdm2x2((0,0),state,env)
        rdm6= rdm.rdm2x2((2,0),state,env)
        energy += einsum('ijklabcd,ijklabcd',rdm4, self.plq[4,:,:,:,:, :,:,:,:])
        energy += einsum('ijklabcd,ijklabcd',rdm6, self.plq[6,:,:,:,:, :,:,:,:])
        rdm5= rdm.rdm2x2((1,0),state,env)
        rdm7= rdm.rdm2x2((3,0),state,env)
        energy += einsum('ijklabcd,ijklabcd',rdm5, self.plq[5,:,:,:,:, :,:,:,:])
        energy += einsum('ijklabcd,ijklabcd',rdm7, self.plq[7,:,:,:,:, :,:,:,:])
        rdm0= rdm.rdm2x2((0,-1),state,env)
        rdm2= rdm.rdm2x2((2,-1),state,env)
        energy += einsum('ijklabcd,ijklabcd',rdm0, self.plq[0,:,:,:,:, :,:,:,:])
        energy += einsum('ijklabcd,ijklabcd',rdm2, self.plq[2,:,:,:,:, :,:,:,:])
        return energy

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
                    # obs[f"{label}{coord}"]= torch.sum(torch.diagonal(mm(rdm1x1, op)))
                    obs[f"{label}{coord}"]= einsum('ij,ji',rdm1x1, op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

            for coord,site in state.sites.items():
               rdm2x1 = rdm.rdm2x1(coord,state,env)
               rdm1x2 = rdm.rdm1x2(coord,state,env)
               obs[f"SS2x1{coord}"]= einsum('ijab,ijab',rdm2x1,self.SS)
               obs[f"SS1x2{coord}"]= einsum('ijab,ijab',rdm1x2,self.SS)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_nnn_SS(self,state,env):
        id2= torch.eye(4, dtype=self.dtype, device=self.device)
        id2= view(id2, (2,2,2,2))
        
        # o  o and  o  o
        #  \          /
        #   \        /
        # o  o      o  o
        nnn_11= einsum('ijab,klcd->ikljacdb',self.SS,id2)
        nnn_m11= nnn_11.permute(1,0,3,2, 5,4,7,6)

        obs= dict()
        for xy in itertools.product(range(4),range(2)):
            rdm2x2= rdm.rdm2x2(xy,state,env)
            obs[f"SS2x2_11{xy}"]= einsum('ijklabcd,abcdijkl',rdm2x2,nnn_11)
            obs[f"SS2x2_m11{xy}"]= einsum('ijklabcd,abcdijkl',rdm2x2,nnn_m11)
        return obs


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