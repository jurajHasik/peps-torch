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

class J1J2LAMBDA_C4V_BIPARTITE():
    def __init__(self, j1=1.0, j2=0, j3=0, hz_stag= 0.0, delta_zz=1.0, lmbd=0, \
        global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j3: next-to-next nearest-neighbour interaction
        :param hz_stag: staggered magnetic field
        :param delta_zz: easy-axis (nearest-neighbour) anisotropy
        :param lmbd: chiral 4-site (plaquette) interaction
        :type lmbd: float
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type j3: float
        :type hz_stag: float
        :type detla_zz: float
        :type global_args: GLOBALARGS
        
        Build Spin-1/2 :math:`J_1-J_2-J_3-\lambda` Hamiltonian

        .. math:: 

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
              + J_3\sum_{<<<i,j>>>} \mathbf{S}_i.\mathbf{S}_j + i\lambda \sum_p P_p - P^{-1}_p
        
        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`), and 
        the last sum runs over pairs of sites `i,j` which are next-to-next nearest-neighbours 
        (denoted as `<<<.,.>>>`). Running over all plaquettes `p`, the chiral term P permutes
        spins on the plaquette in clockwise order and its inverse P^{-1} in anti-clockwise order.
        """
        
        self.dtype=global_args.torch_dtype
        assert torch.rand(1, dtype=self.dtype).is_complex(), "Invalid dtype: J1-J2-Lambda "\
            +" requires complex numbers"
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        self.j3=j3
        self.lmbd= lmbd
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
        chiral_term= chiral_term.permute(0,1,3,2, 4,5,7,6)
        chiral_term= torch.einsum('xj,yk,ixylauvd,ub,vc->ijklabcd',rot_op,rot_op,chiral_term,\
            rot_op,rot_op)
        self.chiral_term= chiral_term.contiguous()
        self.hp_chiral= self.lmbd*chiral_term    

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

        Analogous to :meth:`models.j1j2.J1J2_C4V_BIPARTITE.energy_1x1`.
        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,sym_pos_def=False,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp + self.hp_chiral)
        if abs(self.j3)>0:
            rdm3x1= rdm_c4v.rdm3x1(state,env_c4v,sym_pos_def=True,\
                force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
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
            4. (optionally) :math:`\langle P - P^{-1} \rangle`
    
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

            if abs(self.lmbd)>0:
                rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,force_cpu=force_cpu,\
                    verbosity=cfg.ctm_args.verbosity_rdm)
                obs[f"ChiralT"]= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.chiral_term)

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
        if abs(self.lmbd)>0: obs_labels += [f"ChiralT"]
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
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Identical to :meth:`models.j1j2.J1J2_C4V_BIPARTITE.eval_corrf_SS`.
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
        
        Identical to :meth:`models.j1j2.J1J2_C4V_BIPARTITE.eval_corrf_DD_H`.
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
        
        Identical to :meth:`models.j1j2.J1J2_C4V_BIPARTITE.eval_corrf_DD_V`.
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