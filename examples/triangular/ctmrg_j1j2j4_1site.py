import os
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ipeps.ipeps_1s_Q import *
from ipeps.ipeps_trgl_pg import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import spin_triangular
# from optim.ad_optim import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim_sgd_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--diag", type=float, default=1., help="diagonal strength")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j4", type=float, default=0., help="plaquette coupling")
parser.add_argument("--q", type=float, default=1., help="pitch vector")
parser.add_argument("--jchi", type=float, default=0., help="scalar chirality")
parser.add_argument("--tiling", default="1SITE", help="tiling of the lattice", \
    choices=["1SITE", "1SITE_NOROT", "1STRIV", "1SPG", "1SITEQ"])
parser.add_argument("--corrf_canonical", action='store_true', help="align spin operators" \
    + " with the vector of spontaneous magnetization")
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--compressed_rdms", type=int, default=-1, help="use compressed RDMs for 2x3 and 3x2 patches"\
        +" with chi lower that chi x D^2")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--profile_mode",action='store_true')
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    # 2) select model and the "energy" function     
    if args.tiling in ["1SITE", "1STRIV", "1SPG"]:
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["1SITE_NOROT"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["1SITEQ"]:
        model= spin_triangular.J1J2J4_1SITEQ(j1=args.j1, j2=args.j2, j4=args.j4, diag=args.diag,\
            q=None)
        
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +["1SITE", "1SITE_NOROT", "1STRIV", "1SPG", "1SITEQ"])
    energy_f=model.energy_per_site
    eval_obs_f= model.eval_obs

    if args.instate!=None:
        if args.tiling in ["1STRIV"]:
            state= read_ipeps_trgl_1s_ttphys_pg(args.instate)
        elif args.tiling in ["1SPG"]:
            state= read_ipeps_trgl_1s_tbt_pg(args.instate)
        elif args.tiling in ["1SITEQ"]:
            state= read_ipeps_1s_q(args.instate)
        else:
            state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            # TODO some ansatze have only instance methods
            state = extend_bond_dim(state, args.bond_dim)
        if args.tiling in ["1STRIV","1SPG"]:
            state= state.add_noise(args.instate_noise)
        else:
            state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling in ["1SITE", "1SITE_NOROT"]:
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TTPHYS_PG()
        elif args.tiling == "1SPG":
            state= IPEPS_TRGL_1S_TBT_PG()
        elif args.tiling in ["1SITEQ"]:
            state= IPEPS_1S_Q()
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
            +" the model")
        if args.tiling in ["1SITEQ"]:
            model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4, diag=args.diag,\
                q=None)
        else:
            model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)

    print(state)
    
    # gauge, operates only on IPEPS base and its sites tensors
    if args.gauge:
        state_g= IPEPS_WEIGHTED(state=state).gauge()
        state= state_g.absorb_weights()


    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env, compressed=args.compressed_rdms, unroll=args.loop_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        history.append([e_curr.item()]+obs_values)
        print(", ".join([f"{len(history)}"]+[f"{e_curr}"]*2+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol):
            return True, history
        return False, history

    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        e_curr= energy_f(state, env, compressed=args.compressed_rdms, unroll=args.loop_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}",\
            f"{e_curr}"]+[f"{v}" for v in obs_values]))
        return _conv_check, history

    def ctmrg_conv_specC_loc_profile(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}"]))
        return _conv_check, history

    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)
    
    loss0= energy_f(state, ctm_env_init, compressed=args.compressed_rdms, unroll=args.loop_rdms)
    obs_values, obs_labels = eval_obs_f(state,ctm_env_init)
    print(", ".join(["epoch","conv_crit","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    if args.profile_mode:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/mnist'),
            record_shapes=True,
            #profile_memory=True,
            with_stack=True,
            #with_flops=True,
            #with_modules=True
        )

        prof.start()
        ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_specC_loc_profile)
        prof.step()
        prof.stop()
    else:
        ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init, compressed=args.compressed_rdms, unroll=args.loop_rdms)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    # print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        log.info(f"spectrum C[{c_loc}]")
        log.info(f"{s}")

    # chirality
    # obs= model.eval_obs_chirality(state, ctm_env_init, compressed=args.compressed_rdms,\
    #     unroll=args.loop_rdms)
    # print("\n\n")
    # for label,val in obs.items():
    #     print(f"{label} {val}")

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    dir_to_ind={ (0,-1): 1, (-1,0): 2, (0,1):3, (1,0):4 }
    evecs, evecs_maps=dict(), dict()
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        # direction gives direction of "growth" of the channel
        # i.e. for (1,0) the channel grows to right corresponding to solution of eq.
        #
        #  -- --T--           --
        # E-- --A-- = lambda E--  <=>  solving for *left* eigenvector E(TAT) = lambda E
        #  -- --T--           --    
        l, evecs_left= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init, eigenvectors=True)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

        # compute right eigenvector (TAT)E = lambda E (by reversing the direction of growth), 
        # since A = P^-1 diag(lambda) P i.e. P[:,0] and P^-1[0,:] are not simply related.
        l, evecs_right= transferops.get_Top_spec(1, sdp[0], (-sdp[1][0],-sdp[1][1]), \
            state, ctm_env_init, eigenvectors=True)

        evecs[sdp]= (evecs_left[:,0].view(ctm_env_init.chi,\
                state.site(sdp[0]).size(dir_to_ind[(-sdp[1][0],-sdp[1][1])])**2,ctm_env_init.chi).clone(),
                evecs_right[:,0].view(ctm_env_init.chi,\
                    state.site(sdp[0]).size(dir_to_ind[sdp[1]])**2,ctm_env_init.chi).clone())
        if not state.site(sdp[0]).is_complex():
            assert evecs_left[:,0].imag.abs().max()<1.0e-14,"Leading eigenvector is not real"
            assert evecs_right[:,0].imag.abs().max()<1.0e-14,"Leading eigenvector is not real"
            evecs[sdp]= (evecs[sdp][0].real, evecs[sdp][1].real) 

        # Pass in tuple of leading eigenvector-generating functions. First element gives
        # the left eigenvector, second gives right eigenvector
        evecs_maps[sdp]= (lambda x: evecs[sdp][0], lambda x: evecs[sdp][1])

    # ----- S(0).S(r) -----
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r, \
            canonical=args.corrf_canonical, rl_0=evecs_maps[sdp])
        print(f"\n\nSRSRt[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()])\
            +f" canonical {args.corrf_canonical}")
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r, \
            canonical=args.corrf_canonical, conj_s=False, rl_0=evecs_maps[sdp])
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()])\
            +f" canonical {args.corrf_canonical}")
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ----- S(0).Id(r) -----
    for sdp in site_dir_list:
        corrSId= model.eval_corrf_SId(*sdp, state, ctm_env_init, 0, rl_0=evecs_maps[sdp])
        print(f"\n\nSId[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSId.keys()]))
        for i in range(len(next(iter(corrSId.values())))):
            print(f"{i} "+" ".join([f"{corrSId[label][i]}" for label in corrSId.keys()]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg_TRGL_D3_1SITE(unittest.TestCase):
    tol=1.0e-4
    tol_high= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-trgl_1site"
    ANSATZE= [
        ("1SITE","trglC_j20.1_j40_D3ch27_r0_LS_1SITE_iD3n_C4X4cS_ptol8_state.json",
        (0.1, 1.0),
        """
        -0.5076644938218757, 0.19418241891004798, 0.19418241891004798, (0.08523919965348262+0j), 
        (0.17447369027909715-0.00014851459444411427j), (0.17447369027909715+0.00014851459444411427j), 
        -0.18338522376670024, -0.18374716555353687, -0.17249021501170017
        """,
        tol),
        ("1SITEQ","trgl_diag0.9_q3.0_D3ch49_r2_LS_1SITEQ_c1RND_C4X4cS_ptol12_state.json",
            (0,0.9),
            """
            -0.5333282148759652, 0.28196304154692114, 0.28196304154692114, -0.27525964333112335, 
            -0.06111698251397508, -0.06111698251397508, -0.2358303657738277, -0.23583731839097916, 
            -0.13459918927093553, 3.5095252870487315e-07
            """,
            tol_high
        )]

    def setUp(self):
        args.j1= 1.0
        args.bond_dim=3
        args.chi=27
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_d3_trgl_1site(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                args.tiling= ansatz[0]
                args.instate= self.DIR_PATH+"/../../test-input/"+ansatz[1]
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
                args.j2, args.diag= ansatz[2]
                tol= ansatz[4]

                # i) run ctmrg and compute observables
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                final_obs=None
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs

                # compare with the reference
                ref_data= ansatz[3]
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [complex(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    isclose(val,ref_val, rel_tol=tol, abs_tol=tol)

    def tearDown(self):
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log"]:
                if os.path.isfile(f): os.remove(f)

class Test_j1j2energy_TRGL_D3_1SITE(unittest.TestCase):
    tol= 1.0e-6

    def test_j1j2_energy_impl_d3_trgl_1site(self):
        from cmath import isclose

        torch.manual_seed(1)
        D,X= 3,27
        cfg.global_args.dtype= "complex128"
        cfg.global_args.torch_dtype= torch.complex128
        state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
        env= ENV(X, state)
        init_random(env)

        model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=1.0, j2=0.0, global_args=cfg.global_args)
        energy_nn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
            ctm_args=cfg.ctm_args,global_args=cfg.global_args)

        model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=0, j2=1.0, global_args=cfg.global_args)
        energy_nnn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
            ctm_args=cfg.ctm_args,global_args=cfg.global_args)

        nn_h_v,nn_diag= spin_triangular.eval_nn_per_site((0,0),state,env,model.R,model.Rinv,model.SS,model.SS)
        nnn= spin_triangular.eval_nnn_per_site((0,0),state,env,model.R,model.Rinv,model.SS,unroll=False,
            checkpoint_unrolled=False)

        assert isclose(nn_h_v+nn_diag,energy_nn_manual, rel_tol=self.tol, abs_tol=self.tol)
        assert isclose(nnn,energy_nnn_manual, rel_tol=self.tol, abs_tol=self.tol)

class TestCtmrg_j1j2jXenergy_TRGL_D3_1SITE(unittest.TestCase):
    tol= 1.0e-4
    tol_high= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-trgl_1site"
    ANSATZE= [("1SITE","trglC_j20.1_j40_D3ch27_r0_LS_1SITE_iD3n_C4X4cS_ptol8_state.json"),
        ("1ISTE","trglC_j20.1_j40_jX0.1_D3ch49_r0_LS_1SITE_iD3j201n2_C4X4cS_ptol8_state.json")]

    def setUp(self):
        args.j1= 1.0
        args.j2= 0.1
        args.jchi= 0.1
        args.bond_dim=3
        args.chi=49
        args.GLOBALARGS_dtype= "complex128"

    def test_j1j2jX_energy_impl_d3_trgl_1site(self):
        from cmath import isclose
        import numpy as np

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                args.tiling= ansatz[0]
                args.instate= self.DIR_PATH+"/../../test-input/"+ansatz[1]
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"

                state= read_ipeps(args.instate)

                def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
                    _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
                    return _conv_check, history

                env = ENV(args.chi, state)
                init_env(state, env)

                env, *ctm_log= ctmrg.run(state, env, conv_check=ctmrg_conv_specC_loc)

                model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=1., j2=0, global_args=cfg.global_args)
                energy_nn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=0, j2=1., global_args=cfg.global_args)
                energy_nnn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                nn_h_v,nn_diag= spin_triangular.eval_nn_per_site((0,0),state,env,model.R,model.Rinv,model.SS,model.SS)
                nnn= spin_triangular.eval_nnn_per_site((0,0),state,env,model.R,model.Rinv,model.SS,unroll=False,
                    checkpoint_unrolled=False)

                assert isclose(nn_h_v+nn_diag,energy_nn_manual, rel_tol=self.tol_high, abs_tol=self.tol_high)
                assert isclose(nnn,energy_nnn_manual, rel_tol=self.tol_high, abs_tol=self.tol_high)

                model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=1., j2=1.0e-14, global_args=cfg.global_args)
                energy_nn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                model= spin_triangular.J1J2J4_1SITE(phys_dim=2, j1=0, j2=0, jchi=1., global_args=cfg.global_args)
                energy_chi_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                nn_h_v,nn_diag,chi= spin_triangular.eval_nn_and_chirality_per_site((0,0),state,env,model.R,model.Rinv,
                    model.SS,model.SS,model.h_chi,unroll=False,checkpoint_unrolled=False)

                assert isclose(nn_h_v+nn_diag,energy_nn_manual, rel_tol=self.tol, abs_tol=self.tol)
                assert isclose(chi,energy_chi_manual, rel_tol=self.tol, abs_tol=self.tol)