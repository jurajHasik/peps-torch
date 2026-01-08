import context
from copy import deepcopy
import os
from cmath import isclose, pi
import torch
import numpy as np
import pytest
from ipeps.ipeps import IPEPS, read_ipeps
import groups.su2 as su2
from ctm.generic.env import ENV, init_env, ctmrg_conv_specC
from ctm.generic import ctmrg
from ctm.generic import rdm
from ctm.generic import rdm_looped
from models import spin_triangular
import config as cfg
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"{__name__}.log", mode="w")
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice", \
    choices=["1SITE", "1SITE_NOROT", "1STRIV", "1SPG", "1SITEQ"])
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--compressed_rdms", type=int, default=-1, help="use compressed RDMs for 2x3 and 3x2 patches"\
        +" with chi lower that chi x D^2")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
args, unknown_args = parser.parse_known_args()


def trace_norm(r1,r2):
    return 0.5*torch.linalg.eigvalsh(r1-r2).abs().sum()


class TestRdms_basic_TRGL_D3_1SITE():
    tol= 1.0e-4
    tol_high= 1.0e-8
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-trgl_1site"
    ANSATZE= [("1SITE","trglC_j20.1_j40_D3ch27_r0_LS_1SITE_iD3n_C4X4cS_ptol8_state.json"),
        ("1ISTE","trglC_j20.1_j40_jX0.1_D3ch49_r0_LS_1SITE_iD3j201n2_C4X4cS_ptol8_state.json")]

    @pytest.fixture
    def setUp(self):
        args.bond_dim=3
        args.chi=9
        args.CTMARGS_projector_svd_reltol=1.0e-12
        args.GLOBALARGS_dtype= "complex128"
        cfg.configure(args)
        cfg.print_config()
        torch.set_num_threads(args.omp_cores)
        torch.manual_seed(args.seed)

    @pytest.mark.parametrize("ansatz", ANSATZE)
    def test_rdms2x3_loop_trgl_1site(self,setUp,ansatz):
        args.tiling= ansatz[0]
        args.instate= self.DIR_PATH+"/../test-input/"+ansatz[1]
        args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"

        state= read_ipeps(args.instate)

        def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
            _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
            return _conv_check, history

        env = ENV(args.chi, state)
        init_env(state, env)

        env, *ctm_log= ctmrg.run(state, env, conv_check=ctmrg_conv_specC_loc)

        # x  s4 s3
        # s0 s1 x
        R2x3_ref= rdm.rdm2x3_trglringex((0,0), state, env, sym_pos_def=False, verbosity=0)
        
        # inject more precise projector_svd_reltol
        loc_ctm_args= deepcopy(cfg.ctm_args)
        loc_ctm_args.projector_svd_reltol= 1.0e-14
        R2x3_compressed0= rdm.rdm2x3_trglringex_compressed((0,0),state,env,\
            compressed_chi=(env.chi*args.bond_dim**2),sym_pos_def=False,\
            ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)

        assert trace_norm(R2x3_compressed0,R2x3_ref) < self.tol, \
            f"rdm.rdm2x3_trglringex_compressed does not match reference: {trace_norm(R2x3_compressed0,R2x3_ref)}"

        R2x3_loop= rdm_looped.rdm2x3_loop_trglringex_manual((0,0), state, env, sym_pos_def=False, checkpoint_unrolled=False, \
            verbosity=0)
        assert trace_norm(R2x3_loop,R2x3_ref) < self.tol_high, \
            f"rdm_looped.rdm2x3_loop_trglringex_manual does not match reference: {trace_norm(R2x3_loop,R2x3_ref)}"

        # s0 s1 s2
        # s3 s4 s5
        R2x3_loop_eo= rdm_looped.rdm2x3_loop_oe((0,0), state, env, open_sites=[1,2,3,4], unroll=True,\
            sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
            checkpoint_on_device=False,verbosity=0)
        R2x3_loop_eo= R2x3_loop_eo.permute(2,3,1,0, 6,7,5,4).contiguous()  # permute to match R2x3_ref
        assert trace_norm(R2x3_loop_eo,R2x3_ref) < self.tol_high, \
            f"rdm_looped.rdm2x3_loop_oe does not match reference: {trace_norm(R2x3_loop_eo,R2x3_ref)}"


        loc_ctm_args= deepcopy(cfg.ctm_args)
        loc_ctm_args.projector_svd_reltol= 1.0e-14
        R2x3_loop_eo_compressed0= rdm_looped.rdm2x3_loop_trglringex_compressed((0,0),state,env,open_sites=[0,1,2,3],
            compressed_chi=(env.chi*args.bond_dim**2), sym_pos_def=False,\
            unroll=True, checkpoint_unrolled=False, checkpoint_on_device=False,\
            force_cpu=False, dtype=None,\
            ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)
        assert trace_norm(R2x3_loop_eo_compressed0,R2x3_compressed0) < self.tol_high, \
            f"rdm_looped.rdm2x3_loop_trglringex_compressed does not match compressed reference"+\
                f" rdm.rdm2x3_trglringex_compressed: {trace_norm(R2x3_loop_eo_compressed0,R2x3_compressed0)}"
        # assert trace_norm(R2x3_loop_eo_compressed0,R2x3_ref) < self.tol_high, \
        #     f"rdm_looped.rdm2x3_loop_trglringex_compressed does not match reference: {trace_norm(R2x3_loop_eo_compressed0,R2x3_ref)}"


    @pytest.mark.parametrize("ansatz", ANSATZE)
    def test_rdms3x2_loop_trgl_1site(self,setUp,ansatz):
        args.tiling= ansatz[0]
        args.instate= self.DIR_PATH+"/../test-input/"+ansatz[1]
        args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"

        state= read_ipeps(args.instate)

        def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
            _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
            return _conv_check, history

        env = ENV(args.chi, state)
        init_env(state, env)

        env, *ctm_log= ctmrg.run(state, env, conv_check=ctmrg_conv_specC_loc)

        # x  s2
        # s3 s1
        # s0 x
        R3x2_ref= rdm.rdm3x2_trglringex((0,0), state, env, sym_pos_def=False, verbosity=0)
        
        # inject more precise projector_svd_reltol
        loc_ctm_args= deepcopy(cfg.ctm_args)
        loc_ctm_args.projector_svd_reltol= 1.0e-14
        R3x2_compressed0= rdm.rdm3x2_trglringex_compressed((0,0),state,env,\
            compressed_chi=(env.chi*args.bond_dim**2),sym_pos_def=False,\
            ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)

        assert trace_norm(R3x2_compressed0,R3x2_ref) < self.tol, \
            f"rdm.rdm3x2_trglringex_compressed does not match reference: {trace_norm(R3x2_compressed0,R3x2_ref)}"

        R3x2_loop= rdm_looped.rdm3x2_loop_trglringex_manual((0,0), state, env, sym_pos_def=False, checkpoint_unrolled=False, \
            verbosity=0)
        assert trace_norm(R3x2_loop,R3x2_ref) < self.tol_high, \
            f"rdm_looped.rdm3x2_loop_trglringex_manual does not match reference: {trace_norm(R3x2_loop,R3x2_ref)}"

        # s0 s3
        # s1 s4
        # s2 s5
        R3x2_loop_eo= rdm_looped.rdm3x2_loop_oe((0,0), state, env, open_sites=[1,2,3,4], unroll=True,\
            sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
            checkpoint_on_device=False,verbosity=0)
        R3x2_loop_eo= R3x2_loop_eo.permute(1,3,2,0, 5,7,6,4).contiguous()  # permute to match R3x2_ref
        assert trace_norm(R3x2_loop_eo,R3x2_ref) < self.tol_high, \
            f"rdm_looped.rdm3x2_loop_oe does not match reference: {trace_norm(R3x2_loop_eo,R3x2_ref)}"


        loc_ctm_args= deepcopy(cfg.ctm_args)
        loc_ctm_args.projector_svd_reltol= 1.0e-14
        R3x2_loop_eo_compressed0= rdm_looped.rdm3x2_loop_trglringex_compressed((0,0),state,env,open_sites=[0,1,2,3],
            compressed_chi=(env.chi*args.bond_dim**2), sym_pos_def=False,\
            unroll=True, checkpoint_unrolled=False, checkpoint_on_device=False,\
            force_cpu=False, dtype=None,\
            ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)
        assert trace_norm(R3x2_loop_eo_compressed0,R3x2_compressed0) < self.tol_high, \
            f"rdm_looped.rdm3x2_loop_trglringex_compressed does not match compressed reference"+\
                f" rdm.rdm3x2_trglringex_compressed: {trace_norm(R3x2_loop_eo_compressed0,R3x2_compressed0)}"
        # assert trace_norm(R3x2_loop_eo_compressed0,R2x3_ref) < self.tol_high, \
        #     f"rdm_looped.rdm3x2_loop_trglringex_compressed does not match reference: {trace_norm(R3x2_loop_eo_compressed0,R3x2_ref)}"


class TestRdms_precision_J1_TRGL_D5_1SITE_real():
    tol= 1.0e-4
    tol_high= 1.0e-8
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d5-trgl_1site"
    ANSATZE= [("1SITE","trgl_j20_j40_D5ch150_r9_LS_1SITE_c7D5r9_C4X2_ptol12_state.json"),]

    @pytest.fixture
    def setUp(self, config_kwargs):
        args.j1, args.j2, args.j4, args.jchi= 1,0,0,0
        args.bond_dim=5
        args.chi=50
        # args.CTMARGS_projector_svd_method="PROPACK"
        args.CTMARGS_projector_svd_reltol=1.0e-12
        args.GLOBALARGS_dtype= "float64"
        args.tiling= self.ANSATZE[0][0]
        args.instate= self.DIR_PATH+"/../test-input/"+self.ANSATZE[0][1]
        args.out_prefix=self.OUT_PRFX+f"_{self.ANSATZE[0][0]}"
        cfg.configure(args)
        cfg.print_config()
        torch.set_num_threads(config_kwargs['omp_cores'])
        torch.manual_seed(args.seed)

        self.state= read_ipeps(args.instate)

        def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
            _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
            print(f"ctmrg_conv_specC: {len(history['conv_crit'])} {history['conv_crit'][-1]}")
            return _conv_check, history

        self.env = ENV(args.chi, self.state)
        init_env(self.state, self.env)
        self.env, *ctm_log= ctmrg.run(self.state, self.env, conv_check=ctmrg_conv_specC_loc)

        self.model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)

    
    @pytest.mark.slow
    @pytest.mark.parametrize("compressed_chi",[[25,50,100,200,400,800]])#400,800]]) # max is D^2 \chi
    def test_rdms_2x3_loop_trgl_1site(self,setUp,compressed_chi):

        q= self.model.q
        s2 = su2.SU2(self.model.phys_dim, dtype=self.model.dtype, device=self.model.device)
        R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        Rinv= R.t().conj()

        # s0 s1 s2
        # s3 s4 s5
        R2x3_loop_eo= rdm_looped.rdm2x3_loop_oe((0,0), self.state, self.env, open_sites=[2,3], 
            unroll=True,\
            sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
            checkpoint_on_device=False,verbosity=0)
        SS_ref= torch.einsum('iajb,jbia',R2x3_loop_eo,
                    torch.einsum('jxiy,xb,ya->jbia',self.model.SS,R@R@R,R@R@R)) # A--A nnn
        
        print(f"rdm.rdm2x3_loop_oe chi={self.env.chi} SS_ref={SS_ref}")
        print("rdm_looped.rdm2x3_loop_trglringex_compressed chi_c delta_SS_nnn delta_rho_nnn dtype")
        for dtype in [torch.float32, torch.float64]:
            for c_chi in compressed_chi:
                # x  s3 s2
                # s0 s1 x
                loc_ctm_args= deepcopy(cfg.ctm_args)
                loc_ctm_args.projector_svd_reltol= 1.0e-14
                loc_ctm_args.projector_full_matrices= False
                loc_ctm_args.verbosity_projectors= 1
                R2x3_loop_eo_c= rdm_looped.rdm2x3_loop_trglringex_compressed((0,0),self.state,self.env,open_sites=[0,2],
                    compressed_chi=c_chi, sym_pos_def=False,\
                    unroll=True, checkpoint_unrolled=False, checkpoint_on_device=False,\
                    force_cpu=False, dtype=dtype,\
                    ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)
                R2x3_loop_eo_c= R2x3_loop_eo_c.permute(1,0,3,2).contiguous()  # permute to match R2x3_ref

                SS_c= torch.einsum('iajb,jbia',R2x3_loop_eo_c,
                            torch.einsum('jxiy,xb,ya->jbia',self.model.SS,R@R@R,R@R@R)) # A--A nnn
                
                print(f"{c_chi} {SS_c} {SS_c-SS_ref} {trace_norm(R2x3_loop_eo_c,R2x3_loop_eo)} {dtype}")
            
        
    @pytest.mark.slow
    @pytest.mark.parametrize("compressed_chi",[[25,50,100,200,400,800]])#400,800]]) # max is D^2 \chi
    def test_rdms_3x2_loop_trgl_1site(self,setUp,compressed_chi):

        q= self.model.q
        s2 = su2.SU2(self.model.phys_dim, dtype=self.model.dtype, device=self.model.device)
        R= torch.linalg.matrix_exp( (pi*q[0])*(s2.SP()-s2.SM()) )
        Rinv= R.t().conj()

        # s0 s3
        # s1 s4  
        # s2 s5
        R3x2_loop_eo= rdm_looped.rdm3x2_loop_oe((0,0), self.state, self.env, open_sites=[2,3], 
            unroll=True,\
            sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
            checkpoint_on_device=False,verbosity=0)
        SS_ref= torch.einsum('iajb,jbia',R3x2_loop_eo,
                    torch.einsum('jxiy,xb,ya->jbia',self.model.SS,R@R@R,R@R@R)) # A--A nnn
        
        print(f"rdm.rdm3x2_loop_oe chi={self.env.chi} SS_ref={SS_ref}")
        print("rdm_looped.rdm3x2_loop_trglringex_compressed chi_c delta_SS_nnn delta_rho_nnn")
        for c_chi in compressed_chi:
            # x  s2
            # s3 s1
            # s0 x
            loc_ctm_args= deepcopy(cfg.ctm_args)
            loc_ctm_args.projector_svd_reltol= 1.0e-14
            loc_ctm_args.projector_full_matrices= False
            loc_ctm_args.verbosity_projectors= 1
            R3x2_loop_eo_c= rdm_looped.rdm3x2_loop_trglringex_compressed((0,0),self.state,self.env,open_sites=[0,2],
                compressed_chi=c_chi, sym_pos_def=False,\
                unroll=True, checkpoint_unrolled=False, checkpoint_on_device=False,\
                force_cpu=False, dtype=None,\
                ctm_args=loc_ctm_args,global_args=cfg.global_args,verbosity=0)
            R3x2_loop_eo_c= R3x2_loop_eo_c.permute(1,0,3,2).contiguous()  # permute to match R2x3_ref

            SS_c= torch.einsum('iajb,jbia',R3x2_loop_eo_c,
                        torch.einsum('jxiy,xb,ya->jbia',self.model.SS,R@R@R,R@R@R)) # A--A nnn
            
            print(f"{c_chi} {SS_c} {SS_c-SS_ref} {trace_norm(R3x2_loop_eo_c,R3x2_loop_eo)}")