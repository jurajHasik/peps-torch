import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm, rdm_looped
from ctm.generic import corrf
from math import sqrt, pi
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

def eval_nn_per_site(coord,state,env,R,Rinv,op_nn,op_nn_diag,
    force_cpu=False,unroll=False,checkpoint_unrolled=False,checkpoint_on_device=False):
    if not unroll: unroll= {}

    # O(X^3 D^4 s^2)
    energy_nn, energy_nn_diag= 0., 0.
    #
    # A(0,0)--RA(1,0) (for 120deg A--B)
    tmp_rdm_2x1= rdm.rdm2x1(coord,state,env,force_cpu=force_cpu,
        unroll=unroll.get('rdm2x1',False),checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device)
    energy_nn+= torch.einsum('ijab,abij',
        torch.einsum('ixay,xj,yb->ijab',op_nn,R,R),
        tmp_rdm_2x1)
    
    #
    # A(0,0)                 A
    # |                      |
    # R^-1A(0,1) (for 120deg C)
    tmp_rdm_1x2= rdm.rdm1x2(coord,state,env,force_cpu=force_cpu,
        unroll=unroll.get('rdm1x2',False),checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device)
    energy_nn+= torch.einsum('ijab,abij',
        torch.einsum('ixay,xj,yb->ijab',op_nn,Rinv,Rinv),
        tmp_rdm_1x2)

    #
    # RA     R^2C(1,-1)              B C
    #       /                         /
    # A(0,0)   RA        (for 120deg A B)
    tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2(coord,state,env,open_sites=[1,2],force_cpu=force_cpu,\
        unroll=unroll.get('rdm2x2',False),checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device)
    energy_nn_diag+= torch.einsum('ijab,abij',
        torch.einsum('xjyb,xi,ya->ijab',op_nn_diag,R@R,R@R),
        tmp_rdm_2x2_NNN_1n1)

    return energy_nn, energy_nn_diag

def eval_nnn_per_site_semimanual(coord,state,env,R,Rinv,op_nnn,unroll=False,
    checkpoint_unrolled=False,checkpoint_on_device=False,force_cpu=False,verbosity=0):
    if not unroll: unroll= {}

    # O(X^3 D^6 s^2)
    energy_nnn= 0.
    # RA R^2A R^3A                  B  C  A     x  x s2
    # A  RA   R^2A => 120deg order  A  B  C <=> s3 x x
    tmp_rdm_2x3= rdm_looped.rdm2x3_loop_oe_semimanual(coord,state,env,\
            open_sites=[2,3], unroll=unroll.get('rdm2x3_loop_oe_semimanual',False), 
            checkpoint_unrolled=checkpoint_unrolled, 
            checkpoint_on_device=checkpoint_on_device,
            force_cpu=force_cpu,verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_2x3,
        torch.einsum('jxiy,xb,ya->jbia',op_nnn,R@R@R,R@R@R)) # A--A nnn

    #
    # R^2A R^3A     x  s3                 C A
    # RA   R^2A     x  x                  B C
    # A      RA <=> s2 x  => 120deg order A B
    tmp_rdm_3x2= rdm_looped.rdm3x2_loop_oe_semimanual(coord,state,env,\
            open_sites=[2,3], unroll=unroll.get('rdm3x2_loop_oe_semimanual',False), 
            checkpoint_unrolled=checkpoint_unrolled,
            checkpoint_on_device=checkpoint_on_device,
            force_cpu=force_cpu,verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_3x2,
        torch.einsum('jxiy,xb,ya->jbia',op_nnn,R@R@R,R@R@R)) # A--A nnn

    # 
    # A    RA     s0 x                 A B
    # R^-1A A <=> x s3 => 120deg order C A
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[0,3],force_cpu=force_cpu,
        unroll=unroll.get('rdm2x2',False),checkpoint_unrolled=checkpoint_unrolled, 
        checkpoint_on_device=checkpoint_on_device,
        verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_2x2,op_nnn) # A--A nnn

    return energy_nnn

def eval_nnn_per_site(coord,state,env,R,Rinv,op_nnn,unroll=False,
    checkpoint_unrolled=False,checkpoint_on_device=False,force_cpu=False,verbosity=0):
    if not unroll: unroll= {}

    # O(X^3 D^6 s^2)
    energy_nnn= 0.
    # RA R^2A R^3A                  B  C  A     x  x s2
    # A  RA   R^2A => 120deg order  A  B  C <=> s3 x x
    tmp_rdm_2x3= rdm_looped.rdm2x3_loop_oe(coord,state,env,\
            open_sites=[2,3], unroll=unroll.get('rdm2x3_loop_oe',False), 
            checkpoint_unrolled=checkpoint_unrolled, 
            checkpoint_on_device=checkpoint_on_device,
            force_cpu=force_cpu,verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_2x3,
        torch.einsum('jxiy,xb,ya->jbia',op_nnn,R@R@R,R@R@R)) # A--A nnn

    #
    # R^2A R^3A     x  s3                 C A
    # RA   R^2A     x  x                  B C
    # A      RA <=> s2 x  => 120deg order A B
    tmp_rdm_3x2= rdm_looped.rdm3x2_loop_oe(coord,state,env,\
            open_sites=[2,3], unroll=unroll.get('rdm3x2_loop_oe',False), 
            checkpoint_unrolled=checkpoint_unrolled,
            checkpoint_on_device=checkpoint_on_device,
            force_cpu=force_cpu,verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_3x2,
        torch.einsum('jxiy,xb,ya->jbia',op_nnn,R@R@R,R@R@R)) # A--A nnn

    # 
    # A    RA     s0 x                 A B
    # R^-1A A <=> x s3 => 120deg order C A
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[0,3],force_cpu=force_cpu,
        unroll=unroll.get('rdm2x2',False),checkpoint_unrolled=checkpoint_unrolled, 
        checkpoint_on_device=checkpoint_on_device,
        verbosity=verbosity)
    energy_nnn+= torch.einsum('iajb,jbia',tmp_rdm_2x2,op_nnn) # A--A nnn

    return energy_nnn

def eval_nn_and_chirality_per_site(coord,state,env,R,Rinv,
    op_nn,op_nn_diag,op_chi,
    unroll=False,checkpoint_unrolled=False,
    checkpoint_on_device=False,
    force_cpu=False,verbosity=0):
    if not unroll: unroll= {}

    # O(X^3 D^4 s^[2 to 4]) 
    energy_nn, energy_nn_diag, energy_chi= 0.,0.,0.
    # A    RA     x  s1                     A B    
    # R^-1A A <=> s2 s3 => for 120deg order C A
    #                               B  x B
    # where we evaluate for   C--A, A, C x and chirality with anti-clockwise order
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[1,2,3],unroll=unroll.get('rdm2x2_123',False),\
        checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,
        force_cpu=force_cpu,verbosity=verbosity)
    tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[10,12,4, 11,13,5],\
                    R, [0,10], R, [1,11], Rinv, [2,12], Rinv, [3,13], [0,2,4, 1,3,5])
    energy_nn+= torch.einsum('ijab,nabnij',op_nn,tmp_rdm_2x2)\
        + torch.einsum('ijab,anbinj',op_nn,tmp_rdm_2x2)
    energy_nn_diag+= torch.einsum('ijab,abnijn',op_nn_diag,tmp_rdm_2x2)
    energy_chi+= torch.einsum('ijkabc,abcijk',op_chi,tmp_rdm_2x2)

    #
    # A B     s0 s1                         A  x B
    # C A <=> s2 x, where we evaluate A--B, C, C x and chirality anti-clockwise
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[0,1,2],unroll=unroll.get('rdm2x2_012',False),\
        checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,
        force_cpu=force_cpu,verbosity=verbosity)
    tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[0,10,12, 1,11,13],\
                    R, [2,10], R, [3,11], Rinv, [4,12], Rinv, [5,13], [0,2,4, 1,3,5])
    energy_nn+= torch.einsum('ijab,abnijn',op_nn,tmp_rdm_2x2)\
        + torch.einsum('ijab,anbinj',op_nn,tmp_rdm_2x2)
    energy_nn_diag+= torch.einsum('ijab,nabnij',op_nn_diag,tmp_rdm_2x2)
    energy_chi+= torch.einsum('ijkabc,cbakji',op_chi,tmp_rdm_2x2)

    return energy_nn/2, energy_nn_diag/2, energy_chi

def eval_j1j2j4jX_per_site_legacy(coord,state,env,R,Rinv,\
    op_nn,op_nnn,op_chi,op_p,compressed=-1,unroll=False,
    checkpoint_unrolled=False,checkpoint_on_device=False,force_cpu=False,\
    ctm_args=cfg.ctm_args,global_args=cfg.global_args):

    # O(X^3 D^4 s^[2 to s^6])
    energy_nn, energy_nnn, energy_chi, energy_p= 0.,0.,0.,0.
    # RA R^2A R^3A                  B  C  A     x  s3 s2
    # A  RA   R^2A => 120deg order  A  B  C <=> s0 s1 x
    tmp_rdm_2x3= None
    if compressed>0:
        tmp_rdm_2x3= rdm.rdm2x3_trglringex_compressed(coord,state,env,compressed,\
            ctm_args=ctm_args,global_args=global_args) 
    elif type(unroll)==dict and unroll=={}:
        tmp_rdm_2x3= rdm_looped.rdm2x3_loop_trglringex_manual(coord,state,env,\
            checkpoint_unrolled=checkpoint_unrolled)
    else:
        # x  s0 s1              x  s3 s2
        # s2 s3 x  => (permute) s0 s1 x
        tmp_rdm_2x3= rdm_looped.rdm2x3_loop_oe(coord, state, env, open_sites=[1,2,3,4], 
            unroll=unroll.get('rdm2x3_loop_oe',False),\
            sym_pos_def=False, force_cpu=force_cpu, checkpoint_unrolled=checkpoint_unrolled,
            checkpoint_on_device=checkpoint_on_device)
        tmp_rdm_2x3= tmp_rdm_2x3.permute(2,3,1,0, 6,7,5,4).contiguous()
    tmp_rdm_2x3= torch.einsum(tmp_rdm_2x3,[0,10,12,14,1,11,13,15],\
        R, [2,10], R, [3,11], R@R@R,[4,12],R@R@R,[5,13],\
        R@R, [6,14], R@R, [7,15], [0,2,4,6,1,3,5,7])
    energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,op_nn)
    energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x3,op_nnn) # A--A nnn
    energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x3,op_p)
    #
    # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
    energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_2x3,op_chi)
    energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_2x3,op_chi)

    #
    # R^2A R^3A     x  s2     x k                 C A
    # RA   R^2A     s3 s1     l j                 B C
    # A      RA <=> s0 x  <=> i x => 120deg order A B
    tmp_rdm_3x2= None
    if compressed>0:
        tmp_rdm_3x2= rdm.rdm3x2_trglringex_compressed(coord,state,env,compressed,\
            ctm_args=ctm_args,global_args=global_args) 
    elif type(unroll)==dict and unroll=={}:
        tmp_rdm_3x2= rdm_looped.rdm3x2_loop_trglringex_manual(coord,state,env,\
            checkpoint_unrolled=checkpoint_unrolled)
    else:
        # x  s2               x  s2
        # s0 s3               s3 s1
        # s1  x  => (permute) s0  x
        tmp_rdm_3x2= rdm_looped.rdm3x2_loop_oe(coord, state, env, open_sites=[1,2,3,4], 
            unroll=unroll.get('rdm3x2_loop_oe',False),\
            sym_pos_def=False, force_cpu=force_cpu, checkpoint_unrolled=checkpoint_unrolled,
            checkpoint_on_device=checkpoint_on_device)
        tmp_rdm_3x2= tmp_rdm_3x2.permute(1,3,2,0, 5,7,6,4).contiguous()
    tmp_rdm_3x2= torch.einsum(tmp_rdm_3x2,[0,10,12,14, 1,11,13,15],\
        R@R,[2,10],R@R,[3,11], R@R@R,[4,12],R@R@R,[5,13], R,[6,14],R,[7,15], [0,2,4,6,1,3,5,7])
    energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,op_nn)
    energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_3x2,op_nnn) # A--A nnn
    energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_3x2,op_p)
    energy_chi+= torch.einsum('ijclabcd,abdijl',tmp_rdm_3x2,op_chi)
    energy_chi+= torch.einsum('ajklabcd,bcdjkl',tmp_rdm_3x2,op_chi)

    # A    RA     s0 s1                 s0 s1     i j                 A B   
    # R^-1A A <=> s2 s3 => (permute) => s3 s2 <=> l k => 120def order C A
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[0,1,2,3],unroll=unroll.get('rdm2x2',False),\
        checkpoint_unrolled=checkpoint_unrolled,checkpoint_on_device=checkpoint_on_device,force_cpu=force_cpu)
    tmp_rdm_2x2= tmp_rdm_2x2.permute(0,1,3,2, 4,5,7,6).contiguous()
    tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[0,10,4,12,1,11,5,13],\
        R, [2,10], R, [3,11],\
        Rinv, [6,12], Rinv, [7,13], [0,2,4,6,1,3,5,7])
    energy_nn+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,op_nn)
    energy_nnn+= torch.einsum('ibkdabcd,acik',tmp_rdm_2x2,op_nnn) # A--A nnn
    energy_p+= torch.einsum('ijklabcd,abcdijkl',tmp_rdm_2x2,op_p)
    #
    # anti-clockwise, i.e. s0,s1,s3 and s1,s2,s3
    energy_chi+= torch.einsum('ijclabcd,adbilj',tmp_rdm_2x2,op_chi)
    energy_chi+= torch.einsum('ajklabcd,bdcjlk',tmp_rdm_2x2,op_chi)

    return energy_nn/4, energy_nnn, energy_chi/3, energy_p

def eval_obs_chirality(coord,state,env,R,Rinv,op_chi,looped=False,\
        checkpoint_unrolled=False,ctm_args=cfg.ctm_args,global_args=cfg.global_args):
    r"""
    Expectations of scalar chirality on two non-equivalent triangles
    """
    # A    RA     x  s1                     A B    
    # R^-1A A <=> s2 s3 => for 120deg order C A
    #                               B  x B
    # where we evaluate for   C--A, A, C x and chirality with anti-clockwise order
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[1,2,3],unroll=[2] if looped else [],\
        checkpoint_unrolled=checkpoint_unrolled,force_cpu=force_cpu)
    tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[10,12,4, 11,13,5],\
                    R, [0,10], R, [1,11], Rinv, [2,12], Rinv, [3,13], [0,2,4, 1,3,5])
    chi_123= torch.einsum('ijkabc,abcijk',op_chi,tmp_rdm_2x2)

    #
    # A B     s0 s1                         A  x B
    # C A <=> s2 x, where we evaluate A--B, C, C x and chirality anti-clockwise
    tmp_rdm_2x2= rdm.rdm2x2(coord,state,env,open_sites=[0,1,2],unroll=[1] if looped else [],\
        checkpoint_unrolled=checkpoint_unrolled,force_cpu=force_cpu)
    tmp_rdm_2x2= torch.einsum(tmp_rdm_2x2,[0,10,12, 1,11,13],\
                    R, [2,10], R, [3,11], Rinv, [4,12], Rinv, [5,13], [0,2,4, 1,3,5])
    chi_021= torch.einsum('ijkabc,cbakji',op_chi,tmp_rdm_2x2)
    return chi_123, chi_021


class J1J2J4_1SITEQ():
    def __init__(self, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0, diag=1.,
        q=None, global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param diag: strength of "diagonal" interaction on effective square lattice,
                     (diag*J1) S_r.S_{r+(1,1)}. Default ``diag=1.`` reproduces triangular lattice.
        :param q: pitch vector in units of 2pi
        :type q: tuple(float, float)
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type diag: float
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
        self.diag= diag
        self.q= q
        self.unroll= {
            'j4': {
                'rdm2x3_loop_oe': [47,48,106,107,108,109,102,103,104,105],
                'rdm3x2_loop_oe': [83,84,102,103,104,105,106,107,108,109],
                'rdm2x2': True
            },
            'j2': {
                # reduces peak-mem by a factor 2 * 2^2 * D^2 (for two halves of 2x3 [3x2] patch, each with one open site)
                'rdm2x3_loop_oe_semimanual': [47,48,106,107,104,105], # alternatively [47,48]
                'rdm2x3_loop_oe': [47,48,106,107,104,105],
                'rdm3x2_loop_oe_semimanual': [83,84,104,105,106,107], # alternatively [83,84]
                'rdm3x2_loop_oe': [83,84,104,105,106,107],
                # reduces peak-mem by a factor 2 * 2^2 (for two halves of 2x2 patch, each with one open site)
                'rdm2x2': True
                },
            'jchi': {
                # reduces peak-mem by a factor 2^4 (for two halves of 2x2 patch, one with two open sites)
                'rdm2x2_123': True, # alternatively [102,103]
                'rdm2x2_012': True, # alternatively [100,101]
                },
            'j1': {
                # reduces peak-mem by a factor 2 * 2^2 (for two halves of 2x2 patch, each with one open site)
                'rdm2x2': True 
            }
        }

        self.SS, self.SSSS, self.h_p, self.h_p_and_nnn, self.h_nn_only, self.h_chi= self.get_h()
        self.obs_ops= self.get_obs_ops()

        if not (self.q is None):
            s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
            self.R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
            self.Rinv= self.R.t().conj()

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
        h_nn_only= SSId + SSId.permute(0,3,2,1, 4,7,6,5).contiguous()\
         + SSId.permute(2,3,0,1, 6,7,4,5).contiguous()\
         + SSId.permute(2,0,1,3, 6,4,5,7).contiguous()

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

    def energy_per_site(self,state,env,q=None,compressed=-1,unroll=False,\
        force_cpu=False,ctm_args=cfg.ctm_args,global_args=cfg.global_args):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_1S_Q
        :type env: ENV
        :param q: pitch vector in units of 2pi. By default, pitch vector is read from state.
        :type q: tuple(float, float)
        :param force_cpu: compute on CPU
        :type force_cpu: bool
        :param compressed: if ``compressed`` > 0, use projectors to compress :math:`\chi\times D^2`
                           space to size ``compressed`` 
        :type compressed: int
        :param unroll: use index unrolling when constructing large reduced density matrices
        :type unroll: bool
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        :return: energy per site
        :rtype: float

        We assume single-site iPEPS which tiles the lattice with a (2piq,2piq) spiral 
        pattern::
            
            R^-1A--A-----RA---R^2A--R^3A               A B C A B
            |    / |    / | /    |  / |
            R^-2A--R^-1A--A-----RA--R^2A               C A B C A
            |    / |    / |  /   |  / |
            R^-3A--R^-2A--R^-1A--A----RA               B C A B C
            |    / |    / |    / |  / |
            R^-4A--R^-3A--R^-2A--R^-1A--A => 120degree A B C A B

        All tensors are obtained from tensor A by applying a unitary R on 
        the physical index.
        The unitary R is a rotation around spin y-axis by :math:`2\pi * q`, i.e.
        :math:`R = exp(-i 2\pi * q\sigma^y)`.

        TODO plaquette
        """
        assert not (abs(self.j4)>0 and self.diag!=1),"J4!=0 and diag!=1 are not suppored" 

        energy_nn=0.
        energy_nn_diag=0.
        energy_nnn=0.
        energy_p=0.
        energy_chi=0.

        if q is None: 
            if not (self.q is None):
                q= self.q
            else:
                assert hasattr(state,'q'), "No q-vector available"
                q= state.q

        if type(unroll)==bool:
            unroll= self.unroll if unroll else {}
        
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        Rinv= R.t().conj()

        if abs(self.j4)>0:
            for coord in state.sites.keys():
                _nn,_nnn,_chi,_p=eval_j1j2j4jX_per_site_legacy(coord,state,env,R,Rinv,\
                    self.h_nn_only,self.SS,self.h_chi,self.h_p,\
                    compressed=compressed,unroll=unroll.get('j4',{}),\
                    checkpoint_unrolled=ctm_args.fwd_checkpoint_loop_rdm,
                    checkpoint_on_device=global_args.offload_to_gpu,\
                    force_cpu=force_cpu,ctm_args=ctm_args,global_args=global_args)
                energy_nn+=_nn
                energy_nnn+=_nnn
                energy_chi+=_chi
                energy_p+=_p
        else:
            if abs(self.j2)>0:
                for coord in state.sites.keys():
                    _nnn= eval_nnn_per_site_semimanual(coord,state,env,R,Rinv,self.SS,unroll=unroll.get('j2',{}),
                        checkpoint_unrolled=ctm_args.fwd_checkpoint_loop_rdm,
                        checkpoint_on_device=global_args.offload_to_gpu,force_cpu=force_cpu,\
                        verbosity=ctm_args.verbosity_rdm)
                    energy_nnn+= _nnn
            if abs(self.jchi)>0:
                for coord in state.sites.keys():
                    _nn,_nn_diag,_chi= eval_nn_and_chirality_per_site(coord,state,env,R,Rinv,
                        self.SS,self.SS,self.h_chi,
                        unroll=unroll.get('jchi',{}),checkpoint_unrolled=ctm_args.fwd_checkpoint_loop_rdm,
                        checkpoint_on_device=global_args.offload_to_gpu,
                        force_cpu=force_cpu,
                        verbosity=ctm_args.verbosity_rdm)
                    energy_nn+= _nn
                    energy_nn_diag+= _nn_diag
                    energy_chi+= _chi
            else:
                for coord in state.sites.keys():
                    _nn,_nn_diag= eval_nn_per_site(coord,state,env,R,Rinv,self.SS,self.SS,
                        force_cpu=force_cpu,unroll=unroll.get('j1',{}),
                        checkpoint_unrolled=ctm_args.fwd_checkpoint_loop_rdm,
                        checkpoint_on_device=global_args.offload_to_gpu)
                    energy_nn+= _nn
                    energy_nn_diag+= _nn_diag

        # import pdb; pdb.set_trace()
        energy_cell= self.j1*(energy_nn + self.diag*energy_nn_diag) + \
            self.j2*energy_nnn + self.j4*energy_p + self.jchi*energy_chi
        energy_per_site= _cast_to_real(energy_cell/len(state.sites))

        return energy_per_site

    def eval_obs(self,state,env,q=None,force_cpu=False):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_1S_Q
        :type env: ENV
        :param q: pitch vector in units of 2pi. By default, pitch vector is read from state.
        :type q: tuple(float, float)
        :param force_cpu: compute on CPU
        :type force_cpu: bool
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
            if not (self.q is None):
                q= self.q
            else:
                assert hasattr(state,'q'), "No q-vector available"
                q= state.q
        
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        Rinv= R.t().conj()

        obs= dict({"avg_m": 0.})
        with torch.no_grad():
            # single-site
            #
            # magnetization vector <\vec{S}> for B, and C sublattice is obtained simply
            # by applying appropriate rotation, i.e. <\vec{S}>_B= R <\vec{S}>_A
            #                                        <\vec{S}>_C= R^2 <\vec{S}>_A   
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env,force_cpu=force_cpu)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]
            obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())

            # two-site
            for coord,site in state.sites.items():
                # s0
                # s1
                tmp_rdm_1x2= rdm.rdm1x2(coord,state,env,force_cpu=force_cpu)
                # s0 s1
                tmp_rdm_2x1= rdm.rdm2x1(coord,state,env,force_cpu=force_cpu)
                tmp_rdm_2x2_NNN_1n1= rdm.rdm2x2_NNN_1n1(coord,state,env,force_cpu=force_cpu)
                #
                # A--B
                SS2x1= torch.einsum('ijab,abij',tmp_rdm_2x1,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,R,R))
                obs[f"SS2x1AB{coord}"]= _cast_to_real(SS2x1)

                #
                # A
                # |
                # C
                SS1x2= torch.einsum('ijab,abij',tmp_rdm_1x2,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,Rinv,Rinv))
                obs[f"SS1x2AC{coord}"]= _cast_to_real(SS1x2)

                #
                # B C
                #  /
                # A B
                SS2x2_NNN_1n1= torch.einsum('ijab,abij',tmp_rdm_2x2_NNN_1n1,
                    torch.einsum('ixay,xj,yb->ijab',self.SS,R@R,R@R))
                
                obs[f"SS2x2_NNN_1n1{coord}"]= _cast_to_real(SS2x2_NNN_1n1)
        
        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1AB{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2AC{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS2x2_NNN_1n1{coord}" for coord in state.sites.keys()]
        obs_labels += ["q_x","q_y"]
        obs["q_x"], obs["q_y"]= q[0], q[1]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,coord,direction,state,env,dist,q=None,canonical=False,conj_s=True,rl_0=None):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_1S_Q
        :type env: ENV
        :param q: pitch vector in units of 2pi. By default, pitch vector is read from state.
        :type q: tuple(float, float)
        :param dist: maximal distance of correlator
        :type dist: int
        :param canonical: decompose correlations wrt. to vector of spontaneous magnetization
                          into longitudinal and transverse parts
        :type canonical: bool 
        :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
        :type rl_0: tuple(function(tuple(int,int))->torch.Tensor, function(tuple(int,int))->torch.Tensor)
        :param conj_s: apply spin-rotation to spin operators to obtain corresponding 
                       (2pi*q,2pi*q)-ordered corr. function
        :type conj: bool 
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        s3 = su2.SU2(self.phys_dim+1, dtype=self.dtype, device=self.device)
        S_zxiy= s2.S() if state.site(coord).is_complex() else s2.S().real
        # pass to real, i.e. iS^y= 0.5(S^+ - S^-)
        S_zxiy[2,:,:]= 0.5*(s2.SP()-s2.SM())

        if q is None: 
            if not (self.q is None):
                q= self.q
            else:
                assert hasattr(state,'q'), "No q-vector available"
                q= state.q

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

        # function allowing for additional site-dependent conjugation of op.
        def conjugate_op(op):
            op_0= op.clone()
            sign_r=1. 
            if direction in [(1,0), (0,-1)]:
                sign_r= 1.
            elif direction in [(-1,0), (0,1)]:
                sign_r= -1.
            else:
                raise RuntimeError("Invalid direction "+str(direction))
            
            def _gen_op(r):
                # r=0 is nearest-neighbour
                R= torch.linalg.matrix_exp( (sign_r*(r+1)*pi*q[0])*(s2.SP()-s2.SM()) )
                return torch.einsum('xy,xj,yb->jb',op_0,R,R) if conj_s else op_0
            return _gen_op

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[0,:,:], \
            conjugate_op(S_zxiy[0,:,:]), dist, rl_0=rl_0)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[1,:,:], conjugate_op(S_zxiy[1,:,:]), dist,  rl_0=rl_0)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[2,:,:], conjugate_op(S_zxiy[2,:,:]), dist,  rl_0=rl_0)

        Sz0szR, Sx0sxR, nSy0SyR= _cast_to_real(Sz0szR), _cast_to_real(Sx0sxR), _cast_to_real(nSy0SyR)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

    def eval_corrf_SId(self,coord,direction,state,env,dist,q=None,conj_s=True,rl_0=None):
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
        :param q: pitch vector in units of 2pi. By default, pitch vector is read from state.
        :type q: tuple(float, float)
        :param conj_s: apply spin-rotation to spin operators to obtain corresponding (2pi*q,2pi*q)-order
        :type conj_s: bool
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
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id1= s2.I()
        def _gen_op(r):
            return id1

        if q is None: 
            if not (self.q is None):
                q= self.q
            else:
                assert hasattr(state,'q'), "No q-vector available"
                q= state.q

        S_zxiy= s2.S() if state.site(coord).is_complex() else s2.S().real
        # pass to real, i.e. iS^y= 0.5(S^+ - S^-)
        S_zxiy[2,:,:]= 0.5*(s2.SP()-s2.SM())

        R= torch.linalg.matrix_exp( (pi*(coord[0]*q[0]+coord[1]*q[0]))*(s2.SP()-s2.SM()) )
        S_zxiy= torch.einsum('ki,xkj->xij', R, torch.einsum('xkl,lj->xkj',S_zxiy,R)) if conj_s else S_zxiy

        Sz0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[0,:,:], _gen_op, dist, rl_0=rl_0)
        Sx0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[1,:,:], _gen_op, dist, rl_0=rl_0)
        iSy0IdR= corrf.corrf_1sO1sO(coord,direction,state,env, S_zxiy[2,:,:], _gen_op, dist, rl_0=rl_0)

        Sz0IdR, Sx0IdR, iSy0IdR= _cast_to_real(Sz0IdR), _cast_to_real(Sx0IdR), _cast_to_real(iSy0IdR)

        res= dict({"sz": Sz0IdR, "sx": Sx0IdR, "isy": iSy0IdR})
        return res


class J1J2J4(J1J2J4_1SITEQ):
    def __init__(self, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0, diag=1,\
        q=(0,0), global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param diag: strength of "diagonal" interaction on effective square lattice,
                     (diag*J1) S_r.S_{r+(1,1)}. Default ``diag=1.`` reproduces triangular lattice.
        :param q: ordering vector:
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type diag: float
        :type q: tuple(float,float)
        :type global_args: GLOBALARGS

        See :class:`J1J2J4_1SITEQ`.
        """
        super().__init__(phys_dim=phys_dim, j1=j1, j2=j2, j4=j4, jchi=jchi,\
            diag=diag, q=q, global_args=global_args)

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
        assert self.q[0]==0 and self.q[1]==0,"Ordering vector different than (0,0) not supported."

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
            return self.SS
        
        D0DR= corrf.corrf_2sOH2sOH_E1(coord, direction, state, env, self.SS, _gen_op,\
            dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res


class J1J2J4_1SITE(J1J2J4_1SITEQ):
    def __init__(self, phys_dim=2, j1=1.0, j2=0, j4=0, jchi=0,\
        q=(-1./3,-1./3), global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin irrep, i.e. 2 for spin S=1/2 
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param j4: plaquette interaction
        :param jchi: scalar chirality
        :param q: ordering vector
        :param global_args: global configuration
        :type phys_dim: int
        :type j1: float
        :type j2: float
        :type j4: float
        :type jchi: float
        :type global_args: GLOBALARGS
        
        .. note::
            120degree order is realized by four possible ordering vectors (±1/3, ±1/3).

        See :class:`J1J2J4_1SITEQ`.
        """
        super().__init__(phys_dim=phys_dim, j1=j1, j2=j2, j4=j4, jchi=jchi,
            diag=1, q=q, global_args=global_args)

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
        :return: scalar chiralities
        :rtype: dict(tuple(int,int)), scalar)
    
        Expectations of scalar chirality on two non-equivalent triangles for each site.
        """
        obs=dict()
        for coord in state.sites.keys():
            obs[f'2x2_123_{coord}'], obs[f'2x2_021_{coord}']= eval_obs_chirality(coord,state,env,\
                self.R,self.Rinv,self.h_chi,looped=looped,ctm_args=ctm_args,global_args=global_args)
        return obs