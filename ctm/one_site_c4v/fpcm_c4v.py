import time
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v.ctm_components_c4v import *
try:
    import scipy.sparse.linalg
    from scipy.sparse.linalg import LinearOperator
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")
from linalg.custom_eig import *
import logging
log = logging.getLogger(__name__)

# performs CTM move
def fpcm_MOVE_sl(a, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args,
    past_steps_data=None):
    verbosity= ctm_args.verbosity_fpcm_move
    # 0) extract raw tensors as tuple
    tensors= tuple([a,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the FPCM MOVE segment of CTM algorithm
    def fpcm_MOVE_sl_c(*tensors):
        a, C, T= tensors

        # 1) compute fixed points for U, T
        i=0
        e0=1.0e+16
        tmp_=[(T,C)]
        while e0 > ctm_args.fpcm_fpt_tol:
            if verbosity>0: log.info(f"fpcm_MOVE_sl_c iteration {i}")
            # 1.1) get C, U by solving pulling-through eq
            #      --Cp--T-- = --U--Cp-- where 0--U--1
            #            |       |                2
            Cp, U= isogauge_MPS(T, C0=tmp_[-1][1], isogauge_tol=ctm_args.fpcm_isogauge_tol,\
                verbosity=verbosity)
            Tp= fp_T(a, U, T0=tmp_[-1][0])
            
            # (optional) check symmetry error on Tp
            Tp= 0.5*(Tp + Tp.permute(1,0,2))
            
            # 1.2) check convergence 
            tmp_.append((Tp,Cp))
            if len(tmp_)>1:
                e0= torch.norm(tmp_[1][0]-tmp_[0][0])/max(Tp.size())
                if verbosity>0: log.info(f"fpcm_MOVE_sl_c iteration {i} error (p=2) {e0}"\
                    +f" (p=inf) {torch.norm(tmp_[1][0]-tmp_[0][0],float('inf'))}")
                tmp_.pop(0)
            i+=1

        # 2) compute fixed point of the corner
        nC= fp_C(a, Tp, U, verbosity=verbosity)
        
        # 4) symmetrize, normalize and assign new C,T
        nC= 0.5*(nC + nC.t())
        nC= nC/torch.max(torch.abs(nC))
        # nC= nC/nC.norm()
        Tp= Tp/torch.max(torch.abs(Tp))
        # nT= nT/nT.norm()

        return nC, Tp

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(fpcm_MOVE_sl_c,*tensors)
    else:
        new_tensors= fpcm_MOVE_sl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

def fp_C(a, T, P, C0=None, verbosity=0):
    C0= C0.view(-1) if C0 is not None else None

    # applies
    #  
    #  /--T--P--    /--
    # |   \  /      |
    # C   a^+a   => C
    # |   /  \      |         0--P--1
    #  \--T--P--    \-- where    2
    P_loc= P.permute(0,2,1).contiguous().view(P.size()[0]*P.size()[2],P.size()[1])
    def mv(v):
        B= torch.as_tensor(v,dtype=a.dtype,device=a.device)
        B= B.view(T.size()[0],T.size()[0])
        B= c2x2_sl(a, B, T)
        # absorb and truncate with P
        #
        # C2X2--1 0--P--1
        # 0
        # 0
        # P^t
        # 1->0
        B= P_loc.t() @ B @ P_loc
        B= B.view(-1)
        return B.detach().cpu().numpy()
    M_op= LinearOperator((T.size()[0]**2,T.size()[1]**2), matvec=mv)
    
    D, U= truncated_eig_arnoldi(M_op, 2, v0=C0, dtype=a.dtype, device=a.device)

    # (optional) verify that leading eigenvector is real 
    if verbosity>0: log.info(f"fp_C spec {[tuple(D[i,:].tolist()) for i in range(D.size()[0])]}")
    assert( torch.abs(torch.max(U[:,0,1]))<1.0e-14 )

    nC= U[:,0,0]
    nC= nC.view(T.size()[0],T.size()[0])
    # (optional) verify that leading eigenvector gives symmetric half-row/-column 
    #            transfer matrix
    # print(f"final nC_asymm: {torch.norm(nC-nC.t())}")
    return nC

def fp_T(a, P, T0=None, verbosity=0):
    T0= T0.view(-1) if T0 is not None else None
    
    # applies
    #
    # /----P---     /--
    # |    |        |
    # T--a^+a--  => T--         0--P--1
    # |    |        |              2
    # \----P---     \--  where 
    def mv(v):
        B= torch.as_tensor(v,dtype=a.dtype,device=a.device)
        B= B.view(P.size())

        #    1->0
        #  __P__
        # 0     2->1
        # 0
        # B--2->3
        # 1->2
        B= torch.tensordot(P, B,([0],[0]))

        # 4) double-layer tensor contraction - layer by layer
        # 4i) untangle the fused D^2 indices
        #    0
        #  __P__
        # |     1->1,2
        # |
        # B--3->4,5
        # 2->3
        B= B.view(B.size()[0],a.size()[1],a.size()[1],B.size()[2],\
            a.size()[2],a.size()[2])

        # 4ii) first layer "bra" (in principle conjugate)
        #    0
        #  __P___________
        # |         1    2->1
        # |         1 /0->4
        # B----4 2--a--4->6 
        # | |       3->5
        # |  --5->3
        # 3->2
        B= torch.tensordot(B, a,([1,4],[1,2]))

        # 4iii) second layer "ket"
        #    0
        #  __P__________
        # |    |       1
        # |    |/4 0\  | 
        # B----a---------6->3 
        # | |  |      \1
        # |  -----3 2--a--4->5
        # |    |       3->4
        # |    |
        # 2->1 5->2
        B= torch.tensordot(B, a,([1,3,4],[1,2,0]))

        # 4iv) fuse pairs of aux indices
        #    0
        #  __P_
        # |    | 
        # T----a----3\ 
        # | |  |\     ->3 
        # |  ----a--5/
        # |    | |
        # |    | |
        # 1   (2 4)->2
        B= B.permute(0,1,2,4,3,5).contiguous().view(B.size()[0],B.size()[1],\
            a.size()[3]**2,a.size()[4]**2)

        #    0
        #  __P____
        # |       |
        # |       |
        # T------aa--3->1
        # 1       2
        # 0       2
        # |___P___|
        #     1->2
        B= torch.tensordot(B,P,([1,2],[0,2]))
        B= B.permute(0,2,1).contiguous()
        B= B.view(-1)
        return B.detach().cpu().numpy()
    M_op= LinearOperator((P.numel(),P.numel()), matvec=mv)
    
    D, U= truncated_eig_arnoldi(M_op, 2, v0=T0, dtype=a.dtype, device=a.device)

    # (optional) verify that leading eigenvector is real 
    if verbosity>0: log.info(f"fp_T spec {[tuple(D[i,:].tolist()) for i in range(D.size()[0])]}")
    assert( torch.abs(torch.max(U[:,0,1]))<1.0e-14 )

    nT= U[:,0,0]
    nT= nT.view(P.size())
    # (optional) verify that leading eigenvector gives symmetric half-row/-column 
    #            transfer matrix
    # print(f"final nT_asymm {torch.norm(nT - nT.permute(1,0,2))}")
    return nT

def fp_TT(T, U=None, C2_0=None, verbosity=0):
    if U is None: U=T

    # applies
    #
    # /--T--    /--
    # B  |   => B
    # \--U--    \--
    def mv(v):
        B= torch.as_tensor(v,dtype=T.dtype,device=T.device)
        B= B.view(T.size()[0],T.size()[0])
        
        # B--1 0--T--1 
        # 0       2
        B= torch.tensordot(B,T,([1],[0]))
        
        # B-------T--1
        # |       2
        # |       2
        # \--0 0--U--1->0
        B= torch.tensordot(U,B,([0,2],[0,2]))
        B= B.view(-1)
        return B.detach().cpu().numpy()
    M_op= LinearOperator((T.size()[0]**2,T.size()[0]**2), matvec=mv)

    D, V= truncated_eig_arnoldi(M_op, 2, v0=C2_0, dtype=T.dtype, device=T.device)

    if verbosity>0: log.info(f"fp_TT spec {[tuple(D[i,:].tolist()) for i in range(D.size()[0])]}")
    assert( torch.abs(torch.max(V[:,0,1]))<1.0e-14 )

    nC2= V[:,0,0]
    nC2= nC2.view(T.size()[0],T.size()[0])

    return nC2

def polar_decomp_left(M, normalize=False):
    U, S, V= torch.svd(M, compute_uv=True)
    # 
    # M= --U--S--V^T-- = --(U--V^T)--(V--S--V^T)-- = --U'--P--                                                             --/            2
    U= U@V.t()
    if normalize: S=S/S[0]
    P= V@torch.diag(S)@V.t()
    return P, U

def pull_through(C,T):
    # polar decompostion - pulling through condition
    #
    # 0--C--1 0--T--1 = 0--CT--1 => 0--CT--1 => 0--CT--1
    #            2          2       2--/
    CT= torch.tensordot(C, T, ([1],[0]))
    CT= CT.permute(0,2,1).contiguous().view(C.size()[1]*T.size()[2],T.size()[1])
 
    # CT= --U--S--V^T-- = --(U--V^T)--(V--S--V^T)-- = --U'--P-- = --U'--P-- & 0--U'--1
    #                                                             --/            2
    P, U= polar_decomp_left(CT, normalize=True)
    U= U.view(C.size()[1],T.size()[2],T.size()[1]).permute(0,2,1).contiguous()

    return P, U

def isogauge_MPS(T, C0=None, isogauge_tol=1.0e-8, verbosity=0):
    # get the fixed-point of MPS's transfer matrix
    #
    # C^2--T--           C^2--
    # |    |   = \lambda |
    # \----T--           \----
    if C0 is not None:
        C2_0= C0@C0
        C2_0= C2_0.view(-1)
    else:
        C2_0= None

    nC2= fp_TT(T, C2_0=C2_0, verbosity=verbosity)

    # (optional) verify hermicity
    nC2_asymm_norm= torch.norm(nC2-nC2.t())
    # print(f"nC2_asymm {nC2_asymm_norm}")
    assert( nC2_asymm_norm/torch.abs(nC2).max() < 1.0e-8 )
    nC2= 0.5*(nC2+nC2.t())
    
    D, U= torch.symeig(nC2, eigenvectors=True)
    # D might have an overall minus sign
    # NOTE: torch.symeig orders eigenpairs in ascending manner
    # order in descending order by magnitude
    absD,p= torch.sort(torch.abs(D),descending=True)
    D=D[p]
    U=U[:,p]
    if D[0]<0: D=-D
    # (optional) verify positivity
    assert D.min()/D[0]>-1.0e-14, "Fixed point of transfer matrix is not positive"
    # no large negative elements, clamp non-negative numbers
    D= torch.clamp(D,min=0)

    # take square root and normalize 
    # 
    # C^2--   C--
    # |     = |
    # \----   C--
    nC= U @ torch.diag(torch.sqrt(D/D[0])) @ U.t()

    # solve initial pulling-through equation
    # --C--T-- \propto --U--C-- where 0--U--1
    #      |             |               2
    P, U= pull_through(nC,T)

    # initial error in gauging (Frobenius or spectral norm ?)
    e0= torch.norm(nC-P)/max(nC.size())
    if verbosity>0: 
        log.info(f"isogauge_CT init gauging error (p=2) {e0} (p=inf) {torch.norm(nC-P,float('inf'))}")

    # is error above tolerance ? Iteratively refine
    while e0 > isogauge_tol:
        # get fixed point of mixed transfer matrix
        #
        # C--T--           C--
        # |  |   = \lambda |
        # \--U--           \--
        nC= fp_TT(T, U=U, C2_0=nC, verbosity=verbosity)  
        
        # left polar decomposition nC = QP and nC<-P where 
        # P is positive and Hermitian and Q is unitary 
        nC, Q= polar_decomp_left(nC, normalize=True)

        # (optional) verify nC is positive and Hermitian
        # solve pulling-through equation 
        # --nC--T-- \propto --U--nC--
        #       |                 |
        P, U= pull_through(nC, T)
        e0= torch.norm(nC-P)
        if verbosity>0: log.info(f"isogauge_CT iterative improvement gauging error (p=2) {e0}"\
            +f" (p=inf) {torch.norm(nC-P,float('inf'))}")
    return nC, U