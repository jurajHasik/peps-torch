import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm, rdm_mc
from ctm.generic import corrf
from math import sqrt, pi
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

class J1J2J4():
    def __init__(self, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0,\
        global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type global_args: GLOBALARGS

        Build Spin-S :math:`J_1-J_2-J_4-J_\chi` Hamiltonian

        .. math:: H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} 
                  \mathbf{S}_i.\mathbf{S}_j 
                  + \sum_{\langle i,j,k,l \rangle}[ 
                    (\mathbf{S}_i.\mathbf{S}_j)(\mathbf{S}_k.\mathbf{S}_l) 
                  + (\mathbf{S}_i.\mathbf{S}_l)(\mathbf{S}_j.\mathbf{S}_k)
                  - (\mathbf{S}_i.\mathbf{S}_k)(\mathbf{S}_j.\mathbf{S}_l) ]
                  + J_\chi \sum_{i,j,k\in\Delta} \mathbf{S}_i.(\mathbf{S}_j \cross \mathbf{S}_k)

        on the triangular lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`),
        and finally the last sums runs over unique plaquettes composed of 
        pairs of edge sharing triangles.
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=phys_dim
        self.j1=j1
        self.j2=j2
        self.j4=j4
        self.jchi=jchi
        
        self.SS, self.SSSS, self.h_p, self.h_p_and_nnn, self.h_nn_only, self.h_chi= self.get_h()
        self.obs_ops= self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(self.phys_dim**2,dtype=self.dtype,device=self.device)
        id2= id2.view([self.phys_dim]*4).contiguous()
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        SS= SS.contiguous()
        
        SSId= torch.einsum('ijab,klcd->ijklabcd',SS,id2)
        SSSS= torch.einsum('ijab,klcd->ijklabcd',SS,SS)
        #
        #             k
        #           / |
        #   l--k   l__j
        #  /\ /    | /
        # i--j     i
        #
        #    (01)(23)[<=>(ij)(kl)] + (il)(jk) - (ik)(jl)
        h_p= SSSS + SSSS.permute(0,3,2,1,4,7,6,5) - SSSS.permute(0,2,1,3,4,6,5,7)

        h_p_and_nnn= self.j4 * h_p + self.j2 * SSId.permute(0,2,1,3, 4,6,5,7).contiguous()
        #                      ij   + il ... + lk + 
        h_nn_only= self.j1 * ( SSId + SSId.permute(0,3,2,1, 4,7,6,5).contiguous()\
         + SSId.permute(2,3,0,1, 6,7,4,5).contiguous()\
         + SSId.permute(2,0,1,3, 6,4,5,7).contiguous())

        if self.jchi != 0:
            assert self.dtype==torch.complex128 or self.dtype==torch.complex64,"jchi requires complex dtype"
        Svec= s2.S()
        levicivit3= torch.zeros(3,3,3, dtype=self.dtype, device=self.device)
        levicivit3[0,1,2]=levicivit3[1,2,0]=levicivit3[2,0,1]=1.
        levicivit3[0,2,1]=levicivit3[2,1,0]=levicivit3[1,0,2]=-1.
        SxSS_t= torch.einsum('abc,bij,ckl,amn->ikmjln',levicivit3,Svec,Svec,Svec).contiguous()

        return SS, SSSS, h_p, h_p_and_nnn, h_nn_only, SxSS_t

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_per_site(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float
    
        Computes energy per site of an arbitrary iPEPS by computing contributions
        from 3 non-equivalent rhombuses for each unique site.

        As an example, assume 1x3 iPEPS which tiles the lattice with a tri-partite pattern composed 
        of three tensors A, B, and C::
            
            A--B--C--A--B--C
            | /| /| /| /| /|
            C--A--B--C--A--B
            | /| /| /| /| /|
            B--C--A--B--C--A
            | /| /| /| /| /|
            A--B--C--A--B--C

        The NN of site A are only sites B and C. The evaluation of all NN terms 
        requires two NN-RDMs and one NNN-RDM per site::
        
            A            B C
            |             /
            C, A--B, and A B   

        For NNN terms and plaquette terms, there are again 3 non-equivalent patches, 
        which can be accounted for by one 2x2 RDM and two 2x3 and 3x2 RDMs::

                             C   A
                               / | 
            A B  B C _A      B   C
             \    /_-/       | /
            C A, A  B C, and A   B
        """
        energy_nn=0.
        energy_nnn=0.
        energy_p=0.
        energy_chi=0.
        if abs(self.j2)>0 or abs(self.j4)>0 or abs(self.jchi)>0:
            for coord in state.sites.keys():
                # (0,-1)  (1,-1) (2,-1)      x  s3 s2
                # (0,0) --(1,0)  (2,0)  <=>  s0 s1 x
                tmp_rdm_2x3= None
                if compressed>0:
                    tmp_rdm_2x3= rdm.rdm2x3_compressed(coord,state,env,compressed,\
                        ctm_args=ctm_args,global_args=global_args) 
                elif looped:
                    tmp_rdm_2x3= rdm_mc.rdm2x3_loop(coord,state,env,\
                        use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
                else:
                    tmp_rdm_2x3= rdm.rdm2x3(coord,state,env)
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x3,self.SS)
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,self.h_p)
                #
                # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
                energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_2x3,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_2x3,self.h_chi)
                

                #
                # (0,-2) (1,-2)     x  s2     x k
                # (0,-1) (1,-1)     s3 s1     l j
                # (0,0)  (1,0)  <=> s0 x  <=> i x
                tmp_rdm_3x2= None
                if compressed>0:
                    tmp_rdm_3x2= rdm.rdm3x2_compressed(coord,state,env,compressed,\
                        ctm_args=ctm_args,global_args=global_args) 
                elif looped:
                    tmp_rdm_3x2= rdm_mc.rdm3x2_loop(coord,state,env,\
                        use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
                else:
                    tmp_rdm_3x2= rdm.rdm3x2(coord,state,env)
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_3x2,self.SS)
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,self.h_p)
                energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_3x2,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_3x2,self.h_chi)

                # 
                # (0,0) (1,0)     s0 s1                 s0 s1     i j
                # (0,1) (1,1) <=> s2 s3 => (permute) => s3 s2 <=> l k
                tmp_rdm_2x2= rdm.rdm2x2(coord,state,env).permute(0,1,3,2, 4,5,7,6).contiguous()
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x2,self.SS)
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,self.h_p)
                #
                # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
                energy_chi+= torch.einsum('ijclabcd,adbilj',tmp_rdm_2x2,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bdcjlk',tmp_rdm_2x2,self.h_chi)

            num_sites= len(state.sites)
            energy_per_site= self.j1*energy_nn/(4*num_sites) + self.j2*energy_nnn/num_sites \
                + self.j4*energy_p/num_sites + self.jchi*energy_chi/(2*num_sites)
            
        else:
            for coord in state.sites.keys():
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',self.SS,tmp_rdm_1x2)
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',self.SS,tmp_rdm_2x1)
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',self.SS,tmp_rdm_2x2_NNN_1n1)
        
            num_sites= len(state.sites)
            energy_per_site= self.j1*energy_nn/(4*num_sites) + self.j2*energy_nnn/num_sites \
                    + self.j4*energy_p/num_sites + self.jchi*energy_chi/(2*num_sites)
        
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def energy_per_site_compressed(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float
    
        Computes energy per site of an arbitrary iPEPS by computing contributions
        from 3 non-equivalent rhombuses for each unique site. See :meth:`energy_per_site`.

        This version uses compressed 2x3 and 3x2 RDMs, :meth:`rdm.rdm2x3_compressed` 
        and :meth:`rdm.rdm3x2_compressed`.
        """
        return self.energy_per_site(state,env,compressed=compressed,looped=looped,\
            ctm_args=ctm_args,global_args=global_args)

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
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env)
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)
                SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,self.SS)
                SS2x1= torch.einsum('ijab,abij',tmp_rdm_2x1,self.SS)
                SS2x2_NNN_1n1= torch.einsum('ijab,abij',self.SS,tmp_rdm_2x2_NNN_1n1)
                obs[f"SS2x1{coord}"]= _cast_to_real(SS2x1)
                obs[f"SS1x2{coord}"]= _cast_to_real(SS1x2)
                obs[f"SS2x2_NNN_1n1{coord}"]= _cast_to_real(SS2x2_NNN_1n1)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS2x2_NNN_1n1{coord}" for coord in state.sites.keys()]
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
        # r=0 is nearest-neighbour
        def conjugate_op(op):
            def _gen_op(r):
                return op
            return _gen_op

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

class J1J2J4_1SITE(J1J2J4):
    def __init__(self, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0, global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type global_args: GLOBALARGS
        
        See :class:`J1J2J4`.
        """
        super().__init__(phys_dim=phys_dim, j1=j1, j2=j2, j4=j4, jchi=jchi,
            global_args=global_args)
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        self.R= torch.linalg.matrix_exp( (2*pi/3)*(s2.SP()-s2.SM()) )
        self.SS_rot= torch.einsum('ixay,xj,yb->ijab',self.SS,self.R,self.R)
        self.SS_rot2= torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R)

    def energy_1x3(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float

        We assume 1x3 iPEPS which tiles the lattice with a tri-partite pattern composed 
        of three tensors A, B, and C::
            
            A--B--C--A--B--C
            | /| /| /| /| /|
            C--A--B--C--A--B
            | /| /| /| /| /|
            B--C--A--B--C--A
            | /| /| /| /| /|
            A--B--C--A--B--C

        The tensors B and C are obtained from tensor A by applying a unitary R on 
        the physical index as

        .. math: 

            B^s = R^{ss'}A^{s'}
            C^s = (R^2)^{ss'}A^{s'}

        where the unitary R is a rotation around spin y-axis by :math:`2\pi/3`, i.e.
        :math:`R = exp(-i\frac{2\pi}{3}\sigma^y)`.

        For example, the NN of site A are only sites B and C.
        The evaluation of all NN terms requires two NN-RDMs and one NNN-RDM per site::
        
            A            B C
            |             /
            C, A--B, and A B   

        For NNN terms, there are again 3 non-equivalent terms, which can be accounted for 
        by one NNN-RDM and two NNNN-RDMs::

                              C   A
                                 /
            A B  B  C _A      B / C
             \    _ -          /
            C A, A  B  C, and A   B

        TODO plaquette
        """
        energy_nn=0.
        energy_nnn=0.
        energy_p=0.
        energy_chi=0.
        if abs(self.j2)>0 or abs(self.j4)>0 or abs(self.jchi)>0:
            for coord in state.sites.keys():
                # B  C--A     x  s3 s2
                # A--B  C <=> s0 s1 x
                tmp_rdm_2x3= None
                if compressed>0:
                    tmp_rdm_2x3= rdm.rdm2x3_compressed(coord,state,env,compressed,\
                        ctm_args=ctm_args,global_args=global_args) 
                elif looped:
                    tmp_rdm_2x3= rdm_mc.rdm2x3_loop(coord,state,env,\
                        use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
                else:
                    tmp_rdm_2x3= rdm.rdm2x3(coord,state,env)
                tmp_rdm_2x3= torch.einsum(tmp_rdm_2x3,[0,10,4,12,1,11,5,13],\
                    self.R, [2,10], self.R, [3,11],\
                    self.R@self.R, [6,12], self.R@self.R, [7,13], [0,2,4,6,1,3,5,7])
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x3,self.SS) # A--A nnn
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,self.h_p)
                #
                # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
                energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_2x3,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_2x3,self.h_chi)

                #
                # C A     x  s2     x k
                # B C     s3 s1     l j
                # A B <=> s0 x  <=> i x
                tmp_rdm_3x2= None
                if compressed>0:
                    tmp_rdm_3x2= rdm.rdm3x2_compressed(coord,state,env,compressed,\
                        ctm_args=ctm_args,global_args=global_args) 
                elif looped:
                    tmp_rdm_3x2= rdm_mc.rdm3x2_loop(coord,state,env,\
                        use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
                else:
                    tmp_rdm_3x2= rdm.rdm3x2(coord,state,env)
                tmp_rdm_3x2= torch.einsum(tmp_rdm_3x2,[0,10,4,12,1,11,5,13],\
                    self.R@self.R, [2,10], self.R@self.R, [3,11],\
                    self.R, [6,12], self.R, [7,13], [0,2,4,6,1,3,5,7])
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_3x2,self.SS) # A--A nnn
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,self.h_p)
                energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_3x2,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_3x2,self.h_chi)

                # 
                # A B     s0 s1                 s0 s1     i j
                # C A <=> s2 s3 => (permute) => s3 s2 <=> l k
                tmp_rdm_2x2= rdm.rdm2x2(coord,state,env).permute(0,1,3,2, 4,5,7,6).contiguous()
                tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[0,10,4,12,1,11,5,13],\
                    self.R, [2,10], self.R, [3,11],\
                    self.R@self.R, [6,12], self.R@self.R, [7,13], [0,2,4,6,1,3,5,7])
                energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,self.h_nn_only)
                energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x2,self.SS) # A--A nnn
                energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,self.h_p)
                #
                # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
                energy_chi+= torch.einsum('ijclabcd,adbilj',tmp_rdm_2x2,self.h_chi)
                energy_chi+= torch.einsum('ajklabcd,bdcjlk',tmp_rdm_2x2,self.h_chi)

                num_sites= len(state.sites)
                # the ratio between #nn (the number of) and #nn(diag) is 2:1
                energy_per_site= self.j1*energy_nn/(4*num_sites) + self.j2*energy_nnn/num_sites \
                    + self.j4*energy_p/num_sites + self.jchi*energy_chi/(2*num_sites)
        else:
            for coord in state.sites.keys():
                #
                # A--B
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R,self.R),
                    tmp_rdm_2x1)
                
                #
                # A
                # |
                # C
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R),
                    tmp_rdm_1x2)
                #
                # B      C(1,-1)
                #       /
                # A(0,0) B
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)
                energy_nn+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R),
                    tmp_rdm_2x2_NNN_1n1)

                num_sites= len(state.sites)
                energy_per_site= self.j1*energy_nn/num_sites + self.j2*energy_nnn/num_sites \
                    + self.j4*energy_p/num_sites

        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def energy_1x3_compressed(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float
    
        Computes energy per site of an arbitrary iPEPS by computing contributions
        from 3 non-equivalent rhombuses for each unique site. See :meth:`energy_per_site`.

        This version uses compressed 2x3 and 3x2 RDMs, :meth:`rdm.rdm2x3_compressed` 
        and :meth:`rdm.rdm3x2_compressed`.
        """
        return self.energy_1x3(state,env,compressed=compressed,looped=looped,\
            ctm_args=ctm_args,global_args=global_args)

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
            # single-site
            #
            # magnetization vector <\vec{S}> for B, and C sublattice is obtained simply
            # by applying appropriate rotation, i.e. <\vec{S}>_B= R <\vec{S}>_A
            #                                        <\vec{S}>_C= R^2 <\vec{S}>_A   
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

            # two-site
            for coord,site in state.sites.items():
                # s0
                # s1
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env)
                # s0 s1
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)
                #
                # A--B
                SS2x1= torch.einsum('ijab,abij',tmp_rdm_2x1,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R,self.R))
                obs[f"SS2x1AB{coord}"]= _cast_to_real(SS2x1)

                # #
                # # B--C
                # SS2x1= torch.einsum('ijab,abij',
                #     torch.einsum('mnxy,mi,xa,nj,yb->ijab',self.SS,self.R,self.R,
                #         self.R@self.R, self.R@self.R),
                #     tmp_rdm_2x1)
                # obs[f"SS2x1BC{coord}"]= _cast_to_real(SS2x1)

                # #
                # # C--A
                # SS2x1= torch.einsum('ijab,abij',
                #     torch.einsum('mjxb,mi,xa->ijab',self.SS,
                #         self.R@self.R, self.R@self.R),
                #     tmp_rdm_2x1)
                # obs[f"SS2x1CA{coord}"]= _cast_to_real(SS2x1)

                #
                # A
                # |
                # C
                SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R))
                obs[f"SS1x2AC{coord}"]= _cast_to_real(SS1x2)

                # #
                # # C
                # # |
                # # B
                # SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,
                #     torch.einsum('mnxy,mi,xa,nj,yb->ijab',self.SS,
                #         self.R@self.R, self.R@self.R,
                #         self.R,self.R))
                # obs[f"SS1x2CB{coord}"]= _cast_to_real(SS1x2)

                # #
                # # B
                # # |
                # # A
                # SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,
                #     torch.einsum('mjxb,mi,xa->ijab',self.SS,
                #         self.R, self.R))
                # obs[f"SS1x2BA{coord}"]= _cast_to_real(SS1x2)



                #
                # B C
                #  /
                # A B
                SS2x2_NNN_1n1= torch.einsum('ijab,abij',tmp_rdm_2x2_NNN_1n1,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R))
                
                obs[f"SS2x2_NNN_1n1{coord}"]= _cast_to_real(SS2x2_NNN_1n1)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1AB{coord}" for coord in state.sites.keys()]
            # + [f"SS2x1BC{coord}" for coord in state.sites.keys()] \
            # + [f"SS2x1CA{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2AC{coord}" for coord in state.sites.keys()]
            # + [f"SS1x2CB{coord}" for coord in state.sites.keys()] \
            # + [f"SS1x2BA{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS2x2_NNN_1n1{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_obs_chirality(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float

        We assume 1x3 iPEPS which tiles the lattice with a tri-partite pattern composed 
        of three tensors A, B, and C::
            
            A--B--C--A--B--C
            | /| /| /| /| /|
            C--A--B--C--A--B
            | /| /| /| /| /|
            B--C--A--B--C--A
            | /| /| /| /| /|
            A--B--C--A--B--C

        The tensors B and C are obtained from tensor A by applying a unitary R on 
        the physical index as

        .. math: 

            B^s = R^{ss'}A^{s'}
            C^s = (R^2)^{ss'}A^{s'}

        where the unitary R is a rotation around spin y-axis by :math:`2\pi/3`, i.e.
        :math:`R = exp(-i\frac{2\pi}{3}\sigma^y)`.

        For example, the NN of site A are only sites B and C.
        The evaluation of all NN terms requires two NN-RDMs and one NNN-RDM per site::
        
            A            B C
            |             /
            C, A--B, and A B   

        For NNN terms, there are again 3 non-equivalent terms, which can be accounted for 
        by one NNN-RDM and two NNNN-RDMs::

                              C   A
                                 /
            A B  B  C _A      B / C
             \    _ -          /
            C A, A  B  C, and A   B

        TODO plaquette
        """
        obs=dict()
        for coord in state.sites.keys():
            # B  C--A     x  s3 s2
            # A--B  C <=> s0 s1 x
            tmp_rdm_2x3= None
            if compressed>0:
                tmp_rdm_2x3= rdm.rdm2x3_compressed(coord,state,env,compressed,\
                    ctm_args=ctm_args,global_args=global_args) 
            elif looped:
                tmp_rdm_2x3= rdm_mc.rdm2x3_loop(coord,state,env,\
                    use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
            else:
                tmp_rdm_2x3= rdm.rdm2x3(coord,state,env)
            tmp_rdm_2x3= torch.einsum(tmp_rdm_2x3,[0,10,4,12,1,11,5,13],\
                self.R, [2,10], self.R, [3,11],\
                self.R@self.R, [6,12], self.R@self.R, [7,13], [0,2,4,6,1,3,5,7])
            #
            # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
            obs[f'2x3_013_{coord}']= torch.einsum('ijclabcd,abdijl',tmp_rdm_2x3,self.h_chi)
            obs[f'2x3_123_{coord}']= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_2x3,self.h_chi)

            #
            # C A     x  s2     x k
            # B C     s3 s1     l j
            # A B <=> s0 x  <=> i x
            tmp_rdm_3x2= None
            if compressed>0:
                tmp_rdm_3x2= rdm.rdm3x2_compressed(coord,state,env,compressed,\
                    ctm_args=ctm_args,global_args=global_args) 
            elif looped:
                tmp_rdm_3x2= rdm_mc.rdm3x2_loop(coord,state,env,\
                    use_checkpoint=ctm_args.fwd_checkpoint_loop_rdm)
            else:
                tmp_rdm_3x2= rdm.rdm3x2(coord,state,env)
            tmp_rdm_3x2= torch.einsum(tmp_rdm_3x2,[0,10,4,12,1,11,5,13],\
                self.R@self.R, [2,10], self.R@self.R, [3,11],\
                self.R, [6,12], self.R, [7,13], [0,2,4,6,1,3,5,7])
            obs[f'3x2_013_{coord}']= torch.einsum('ijclabcd,abdijl',tmp_rdm_3x2,self.h_chi)
            obs[f'3x2_123_{coord}']= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_3x2,self.h_chi)

            # 
            # A B     s0 s1                 s0 s1     i j
            # C A <=> s2 s3 => (permute) => s3 s2 <=> l k
            tmp_rdm_2x2= rdm.rdm2x2(coord,state,env).permute(0,1,3,2, 4,5,7,6).contiguous()
            tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[0,10,4,12,1,11,5,13],\
                self.R, [2,10], self.R, [3,11],\
                self.R@self.R, [6,12], self.R@self.R, [7,13], [0,2,4,6,1,3,5,7])
            #
            # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
            obs[f'2x2_013_{coord}']= torch.einsum('ijclabcd,adbilj',tmp_rdm_2x2,self.h_chi)
            obs[f'2x2_123_{coord}']= torch.einsum('ajklabcd,bdcjlk',tmp_rdm_2x2,self.h_chi)

        return obs

    def eval_corrf_SS(self,coord,direction,state,env,dist,canonical=False,conj_s=True,rl_0=None):
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
        :param canonical: decompose correlations wrt. to vector of spontaneous magnetization
                          into longitudinal and transverse parts
        :type canonical: bool 
        :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
        :type rl_0: tuple(function(tuple(int,int))->torch.Tensor, function(tuple(int,int))->torch.Tensor)
        :param conj_s: apply spin-rotation to spin operators to obtain corresponding 120deg corr. functions
        :type conj: bool 
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        s3 = su2.SU2(self.phys_dim+1, dtype=self.dtype, device=self.device)
        S_zxiy= s2.S().real
        # pass to real, i.e. iS^y= 0.5(S^+ - S^-)
        S_zxiy[2,:,:]= 0.5*(s2.SP()-s2.SM())

        # compute vector of spontaneous magnetization
        if canonical:
            Sexp_zxiy= None
            if rl_0 is None:
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                Sexp_zxiy= torch.einsum('ij,aji->a',rdm1x1,S_zxiy)
            else:
                _tmp= self.eval_corrf_SId(coord,direction,state,env,0,rl_0=rl_0)
                Sexp_zxiy= torch.as_tensor([_tmp['sz'], _tmp['sx'], _tmp['isy']],\
                    dtype=self.dtype, device=self.device)

            Sexp_zxiy= Sexp_zxiy/torch.norm(Sexp_zxiy)
            # 1) build rotation matrix
            SR= s3.I()
            SR[:2,:2]= Sexp_zxiy[0]*s2.I() + Sexp_zxiy[1]*(2*S_zxiy[2,:,:])

            # 2) rotate the vector of operators
            S_zxiy= torch.einsum('ab,bij->aij',SR,S_zxiy)

            rdm1x1 = rdm.rdm1x1(coord,state,env)
            Sexp_zxiy= torch.einsum('ij,aji->a',rdm1x1,S_zxiy)

        # function allowing for additional site-dependent conjugation of op.
        # r=0 is nearest-neighbour
        def conjugate_op(op):
            op_0= op
            op_rot= torch.einsum('ki,kj->ij', self.R, op_0 @ self.R)
            op_rot2= torch.einsum('ki,kj->ij', self.R, op_rot @ self.R)
            if direction in [(1,0), (0,-1)]:
                r_to_op= {0: op_rot, 1: op_rot2, 2: op_0}
            elif direction in [(-1,0), (0,1)]:
                r_to_op= {0: op_rot2, 1: op_rot, 2: op_0}
            else:
                raise RuntimeError("Invalid direction "+str(direction))
            def _gen_op(r):
                return r_to_op[abs(r%3)] if conj_s else op_0
            return _gen_op

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[0,:,:], \
            conjugate_op(S_zxiy[0,:,:]), dist, rl_0=rl_0)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[1,:,:], conjugate_op(S_zxiy[1,:,:]), dist,  rl_0=rl_0)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[2,:,:], conjugate_op(S_zxiy[2,:,:]), dist,  rl_0=rl_0)

        Sz0szR, Sx0sxR, nSy0SyR= _cast_to_real(Sz0szR), _cast_to_real(Sx0sxR), _cast_to_real(nSy0SyR)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

    def eval_corrf_SId(self,coord,direction,state,env,dist,rl_0=None):
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
        :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
        :type rl_0: tuple(function(tuple(int,int))->torch.Tensor, function(tuple(int,int))->torch.Tensor)
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        # function allowing for additional site-dependent conjugation of op
        # r=0 is nearest-neighbour
        id1= torch.eye(self.phys_dim,dtype=self.dtype,device=self.device)
        def _gen_op(r):
            return id1

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= 0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            _gen_op, dist, rl_0=rl_0)
        Sx0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, _gen_op, dist, rl_0=rl_0)
        iSy0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, _gen_op, dist, rl_0=rl_0)

        Sz0IdR, Sx0IdR, iSy0IdR= _cast_to_real(Sz0IdR), _cast_to_real(Sx0IdR), _cast_to_real(iSy0IdR)

        res= dict({"sz": Sz0IdR, "sx": Sx0IdR, "isy": iSy0IdR})
        return res