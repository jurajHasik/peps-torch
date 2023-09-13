import numpy as np
import torch
import config as cfg
import yastn.yastn as yastn
import groups.su2 as su2
import groups.su2_abelian as su2_abelian
from models.spin_triangular import J1J2J4, J1J2J4_1SITEQ
from ctm.generic.rdm import _cast_to_real
from ctm.generic_abelian import rdm
from ctm.generic_abelian import corrf
from math import sqrt, pi
import itertools

class J1J2J4_NOSYM(J1J2J4):
    def __init__(self, settings, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0, diag=1,
        global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param diag: strength of "diagonal" interaction on effective square lattice,
                     (diag*J1) S_r.S_{r+(1,1)}. Default ``diag=1.`` reproduces triangular lattice.
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type diag: float
        :type global_args: GLOBALARGS
        
        See :class:`J1J2J4`.
        """
        super().__init__(phys_dim=phys_dim, j1=j1, j2=j2, j4=j4, jchi=jchi,\
            diag=diag, global_args=global_args)
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.dtype=torch.complex128 if settings.default_dtype=='complex128' else torch.float64
        self.device='cpu' if not hasattr(settings, 'device') else settings.device

    def energy_per_site(self,state,env,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float

        The evaluation of all NN terms requires two NN-RDMs and one NNN-RDM per site::
        
            (0,0)                    (0,-1) (1,-1) 
             |                             /
            (0,1), (0,0)--(1,0), and (0,0)  (1,0)   

        For NNN terms, there are again 3 non-equivalent terms, which can be accounted for 
        by one NNN-RDM and two NNNN-RDMs::

                                                    (0,-2)  (1,-2)
                                                           /
            (0,0) (1,0)  (0,-1)  (1,-1) _(2,-1)     (0,-1)/ (1,-1)
                 \            _   -                      /
            (0,1) (1,1), (0,0)   (1,0)   (2,0), and (0,0)   (1,0)

        TODO plaquette
        """
        energy_nn=0.
        energy_nn_diag=0.
        energy_nnn=0.
        energy_p=0.
        energy_chi=0.

        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
            t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
            D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense_2s={i: l for i,l in enumerate(_tmp_t.get_legs())}

        if abs(self.j2)>0 or abs(self.j4)>0 or abs(self.jchi)>0:
            raise RuntimeError("Not implemented")
        else:
            for coord in state.sites.keys():
                #
                # A--B
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                tmp_rdm_2x1= tmp_rdm_2x1.to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn+= torch.einsum('ijab,abij',tmp_rdm_2x1,self.SS)
                
                #
                # A
                # |
                # C
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn+= torch.einsum('ijab,abij',tmp_rdm_1x2,self.SS)
                #
                # B      C(1,-1)
                #       /
                # A(0,0) B
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)\
                    .to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn_diag+= torch.einsum('ijab,abij',tmp_rdm_2x2_NNN_1n1,self.SS)

                num_sites= len(state.sites)
                energy_per_site= self.j1*(energy_nn + self.diag*energy_nn_diag)/num_sites

        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def eval_obs(self,state,env,q=None):
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
        _tmp_t= yastn.ones(config=state.engine, s=(-1, 1),
                t=((-1, 1), (-1, 1)),
                D=((1, 1), (1, 1)))
        _lss_dense_1s={i: l for i,l in enumerate(_tmp_t.get_legs())}

        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
                t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
                D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense_2s={i: l for i,l in enumerate(_tmp_t.get_legs())}

        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            # single-site
            #
            # magnetization vector <\vec{S}> for B, and C sublattice is obtained simply
            # by applying appropriate rotation, i.e. <\vec{S}>_B= R <\vec{S}>_A
            #                                        <\vec{S}>_C= R^2 <\vec{S}>_A   
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env).to_dense(legs=_lss_dense_1s,reverse=True)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

            # two-site
            for coord,site in state.sites.items():
                # s0
                # s1
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                # s0 s1
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                # x  s1
                # s0 x
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)\
                    .to_dense(legs=_lss_dense_2s,reverse=True)
               
                SS2x1= torch.einsum('ijab,abij',tmp_rdm_2x1,self.SS)
                obs[f"SS2x1{coord}"]= _cast_to_real(SS2x1)

                SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,self.SS)
                obs[f"SS1x2{coord}"]= _cast_to_real(SS1x2)

                SS2x2_NNN_1n1= torch.einsum('ijab,abij',tmp_rdm_2x2_NNN_1n1,self.SS)
                obs[f"SS2x2_NNN_1n1{coord}"]= _cast_to_real(SS2x2_NNN_1n1)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS2x2_NNN_1n1{coord}" for coord in state.sites.keys()]
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

        s2_U1= su2_abelian.SU2_U1(state.engine, 2)

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SZ(), \
            conjugate_op(s2_U1.SZ()), dist, rl_0=rl_0)
        Sp0smR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SP(), conjugate_op(s2_U1.SM()),\
            dist, rl_0=rl_0)
        Sm0SpR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SM(), conjugate_op(s2_U1.SP()),\
            dist, rl_0=rl_0)

        res= dict({"ss": Sz0szR+0.5*(Sp0smR+Sm0SpR), "szsz": Sz0szR, "spsm": Sp0smR, "smsp": Sm0SpR})
        return res

    def eval_corrf_SId(self,coord,direction,state,env,dist,rl_0=None):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_1S_Q
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
        :type rl_0: tuple(function(tuple(int,int))->yastn.Tensor, function(tuple(int,int))->yastn.Tensor)
        :return: dictionary with spin-Id correlation functions
        :rtype: dict(str: np.ndarray)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        # function allowing for additional site-dependent conjugation of op
        # r=0 is nearest-neighbour
        s2_U1= su2_abelian.SU2_U1(state.engine, 2)
        id1= s2_U1.I()
        def _gen_op(r):
            return id1

        Sz0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, s2_U1.SZ(), _gen_op, dist, rl_0=rl_0)

        res= dict({"sz": Sz0IdR, "sx": np.zeros(len(Sz0IdR)), "isy": np.zeros(len(Sz0IdR))})
        return res

class J1J2J4_1SITEQ_NOSYM(J1J2J4_1SITEQ):
    def __init__(self, settings, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0, diag=1.,
        q=(1./2.,1./2.), global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param diag: strength of "diagonal" interaction on effective square lattice,
                     (diag*J1) S_r.S_{r+(1,1)}. Default ``diag=1.`` reproduces triangular lattice.
        :param q: pitch vector in units of 2pi
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type diag: float
        :type q: tuple(float, float)
        :type global_args: GLOBALARGS
        
        See :class:`J1J2J4`.
        """
        super().__init__(phys_dim=phys_dim, j1=j1, j2=j2, j4=j4, jchi=jchi,\
            diag=diag, global_args=global_args)
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.dtype=torch.complex128 if settings.default_dtype=='complex128' else torch.float64
        self.device='cpu' if not hasattr(settings, 'device') else settings.device

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        self.R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        self.Rinv= self.R.t().conj()
        # self.SS_rot= torch.einsum('ixay,xj,yb->ijab',self.SS,self.R,self.R)
        # self.SS_rot2= torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R)

    def energy_1x3(self,state,env,q=None,compressed=-1,looped=False,\
        ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
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
            C^s = (R^-1)^{ss'}A^{s'}

        where the unitary R is a rotation around spin y-axis by :math:`2\pi * q`, i.e.
        :math:`R = exp(-i 2\pi * q\sigma^y)`.

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
        energy_nn_diag=0.
        energy_nnn=0.
        energy_p=0.
        energy_chi=0.

        if q is None: 
            assert hasattr(state,'q'), "No q-vector available"
            q= state.q

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        self.R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        self.Rinv= self.R.t().conj()

        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
            # t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
            t=((0, 1), (0, 1), (0, 1), (0, 1)),
            D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense_2s={i: l for i,l in enumerate(_tmp_t.get_legs())}

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
                    + self.j4*energy_p/num_sites + self.jchi*energy_chi/(3*num_sites)
        else:
            for coord in state.sites.keys():
                #
                # A--B
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env)
                import pdb; pdb.set_trace;
                tmp_rdm_2x1= tmp_rdm_2x1.to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R,self.R),
                    tmp_rdm_2x1)
                
                #
                # A
                # |
                # C
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.Rinv,self.Rinv),
                    tmp_rdm_1x2)
                #
                # B      C(1,-1)
                #       /
                # A(0,0) B
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)\
                    .to_dense(legs=_lss_dense_2s,reverse=True)
                energy_nn_diag+= torch.einsum('ijab,abij',
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.R@self.R,self.R@self.R),
                    tmp_rdm_2x2_NNN_1n1)

                num_sites= len(state.sites)
                energy_per_site= self.j1*(energy_nn + self.diag*energy_nn_diag)/num_sites

        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site

    def eval_obs(self,state,env,q=None):
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
        if q is None: 
            assert hasattr(state,'q'), "No q-vector available"
            q= state.q
        
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        self.R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        self.Rinv= self.R.t().conj()

        _tmp_t= yastn.ones(config=state.engine, s=(-1, 1),
                # t=((-1, 1), (-1, 1)),
                t=((0, 1), (0, 1)),
                D=((1, 1), (1, 1)))
        _lss_dense_1s={i: l for i,l in enumerate(_tmp_t.get_legs())}

        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
                # t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
                t=((0, 1), (0, 1), (0, 1), (0, 1)),
                D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense_2s={i: l for i,l in enumerate(_tmp_t.get_legs())}

        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            # single-site
            #
            # magnetization vector <\vec{S}> for B, and C sublattice is obtained simply
            # by applying appropriate rotation, i.e. <\vec{S}>_B= R <\vec{S}>_A
            #                                        <\vec{S}>_C= R^2 <\vec{S}>_A   
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env).to_dense(legs=_lss_dense_1s,reverse=True)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

            # two-site
            for coord,site in state.sites.items():
                # s0
                # s1
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                # s0 s1
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env).to_dense(legs=_lss_dense_2s,reverse=True)
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env)\
                    .to_dense(legs=_lss_dense_2s,reverse=True)
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
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.Rinv,self.Rinv))
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
                    torch.einsum('ixay,xj,yb->ijab',self.SS,self.Rinv,self.Rinv))
                
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