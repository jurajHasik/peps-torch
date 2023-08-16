import os
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
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
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j4", type=float, default=0., help="plaquette coupling")
parser.add_argument("--jchi", type=float, default=0., help="scalar chirality")
parser.add_argument("--diag", type=float, default=1, help="diagonal strength")
parser.add_argument("--tiling", default="3SITE", help="tiling of the lattice", \
    choices=["1SITE", "1SITE_NOROT", "1STRIV", "1SPG", "2SITE", "3SITE", "4SITE"])
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--corrf_DD", action='store_true', help="compute horizontal dimer-dimer correlation function")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--gauge", action='store_true', help="put into quasi-canonical form")
parser.add_argument("--compressed_rdms", type=int, default=-1, help="use compressed RDMs for 2x3 and 3x2 patches"\
        +" with chi lower that chi x D^2")
parser.add_argument("--loop_rdms", action='store_true', help="loop over central aux index in rdm2x3 and rdm3x2")
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling in ["1SITE", "1STRIV", "1SPG"]:
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling in ["1SITE_NOROT"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        lattice_to_site=None
    elif args.tiling == "2SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, diag=args.diag)
        def lattice_to_site(coord):
            vx = coord[0] % 2
            vy = coord[1]
            return (vx, 0)
    elif args.tiling == "3SITE":
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi)
        def lattice_to_site(coord):
            vx = coord[0] % 3
            vy = coord[1]
            return ((vx - vy) % 3, 0)
    elif args.tiling in ["4SITE", "4SITE_T"]:
        model= spin_triangular.J1J2J4(j1=args.j1, j2=args.j2, j4=args.j4, jchi=args.jchi, diag=args.diag)
        if args.tiling=="4SITE":
            def lattice_to_site(coord):
                vx = coord[0] % 2
                vy = ( coord[1] + ((coord[0]%4)//2) ) % 2
                return (vx, vy)
        elif args.tiling=="4SITE_T":
            def lattice_to_site(coord):
                vx = coord[0] % 2
                vy = coord[1] % 2
                return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE, 2SITE, 3SITE, 4SITE, 4SITE_T")

    if args.instate!=None:
        if args.tiling in ["1STRIV"]:
            state= read_ipeps_trgl_1s_ttphys_pg(args.instate)
        elif args.tiling in ["1SPG"]:
            state= read_ipeps_trgl_1s_tbt_pg(args.instate)
        else:
            state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling in ["1SITE", "1SITE_NOROT"]:
            state= IPEPS(dict(), lX=1, lY=1)
        elif args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TTPHYS_PG()
        elif args.tiling == "1SPG":
            state= IPEPS_TRGL_1S_TBT_PG()
        elif args.tiling == "2SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=2, lY=1)
        elif args.tiling == "3SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=3, lY=3)
        elif args.tiling == "4SITE":
            state= IPEPS(dict(), vertexToSite=lattice_to_site, lX=4, lY=2)
        state.load_checkpoint(args.opt_resume)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4)

    print(state)
    
    # gauge, operates only IPEPS base and its sites tensors
    if args.gauge:
        state_g= IPEPS_WEIGHTED(state=state).gauge()
        state= state_g.absorb_weights()

    # 2) select the "energy" function 
    energy_f=model.energy_per_site
    eval_obs_f= model.eval_obs

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env, compressed=args.compressed_rdms, \
            unroll=args.loop_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        history.append([e_curr.item()]+obs_values)
        print(", ".join([f"{len(history)}"]+[f"{e_curr}"]*2+[f"{v}" for v in obs_values]))

        if (len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol):
            return True, history
        return False, history

    def ctmrg_conv_specC_loc(state, env, history, ctm_args=cfg.ctm_args):
        _conv_check, history= ctmrg_conv_specC(state, env, history, ctm_args=ctm_args)
        e_curr= energy_f(state, env, compressed=args.compressed_rdms,\
            unroll=args.loop_rdms)
        obs_values, obs_labels = eval_obs_f(state,env)
        print(", ".join([f"{len(history['diffs'])}",f"{history['conv_crit'][-1]}",\
            f"{e_curr}"]+[f"{v}" for v in obs_values]))
        return _conv_check, history

    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC_loc
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    
    loss0= energy_f(state, ctm_env_init, compressed=args.compressed_rdms,\
        unroll=args.loop_rdms)
    obs_values, obs_labels = eval_obs_f(state,ctm_env_init)
    print(", ".join(["epoch","conv_crit","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_f)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init, compressed=args.compressed_rdms,\
        unroll=args.loop_rdms)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    # print("\n")
    # for c_loc,c_ten in ctm_env_init.C.items(): 
    #     u,s,v= torch.svd(c_ten, compute_uv=False)
    #     print(f"spectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")

    # chirality
    # obs= model.eval_obs_chirality(state, ctm_env_init, compressed=args.compressed_rdms,\
    #     unroll=args.loop_rdms)
    # print("\n\n")
    # for label,val in obs.items():
    #     print(f"{label} {val}")

    # ----- S(0).S(r) -----
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        corrSS= model.eval_corrf_SS(*sdp, state, ctm_env_init, args.corrf_r)
        print(f"\n\nSS[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrSS.keys()]))
        for i in range(args.corrf_r):
            print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # ----- horizontal dimer-dimer (S(0).S(x))(S(rx).S(rx+x)) -----
    if args.corrf_DD:
        for sdp in [((0,0), (1,0))]:
            corrDD= model.eval_corrf_DD_H(*sdp, state, ctm_env_init, args.corrf_r)
            print(f"\n\nDD[{sdp[0]},{sdp[1]}] r "+" ".join([label for label in corrDD.keys()]))
            for i in range(args.corrf_r):
                print(f"{i} "+" ".join([f"{corrDD[label][i]}" for label in corrDD.keys()])) 

    # transfer operator spectrum
    site_dir_list=[((0,0), (1,0)),((0,0), (0,1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_spec(args.top_n, *sdp, state, ctm_env_init)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg_TRGL_D3_2SITE(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-trgl_2site"
    ANSATZE= [
        ("4SITE",(0,0.2),
        "trgl_j20_j40.2_D2ch18_r0_4SITE_iD1j408n_state.json",
        """
        -0.4285699726740929, 0.31621010409386163, 0.3156632411737321, 0.3167680236637072, 
        0.3159352048119832, 0.3164739467260239, -0.05096502541476977, -0.3115218259011359, 
        -0.3115218259011359, -0.051148531957431714, -0.3126112737800903, -0.3126112737800903, 
        0.051162944045398645, 0.3117649864821211, 0.3117649864821211, 0.050605949320557675, 
        0.31240165948616755, 0.31240165948616755, 0.14612501689854762, -0.4056401319041736, 
        0.1467662366349674, -0.4048845166531687, -0.2075142124009132, -0.20847272287628027, 
        -0.20827045879520084, -0.2076929659422449, -0.3994675501487469, 0.15268205127772405, 
        -0.3989443401985397, 0.15295405462022651
        """
        ),
        # ("2SITE",(0.1,0),
        # "trglC_j20.1_j40_D3ch27_r0_LS_2SITE_iRND_C4X4cS_ptol8_state.json",
        # """
        # -0.5009862860339886, 0.19684623746739846, 0.19684652115010953, 0.1968459537846874, 
        # (0.06767173380753749+0j), (0.061493668336264865-0.17432044655441817j), (0.061493668336264865+0.17432044655441817j), 
        # (-0.06768089236019678+0j), (-0.06149150841720994+0.17431701214542125j), (-0.06149150841720994-0.17431701214542125j), 
        # -0.29891582297393593, -0.29888741633838806, 0.09265360104449452, 0.09264535663046618, 
        # -0.30300708787231867, -0.3026830410396812
        # """
        # ),
        ("2SITE",(0.1,0),
            "trglC_j20.1_j40_D2ch24_r0_LS_2SITE_iRND_C4X4cS_ptol8_state.json",
            """
            -0.48596171247057873, 0.29683089351341685, 0.29683055665163766, 0.29683123037519604, 
            (-0.2411977616244775+0j), (0.1729627044936747-0.003990238551334363j), (0.1729627044936747+0.003990238551334363j), 
            (0.24128045475046545+0j), (-0.17284749294252105+0.004033071403588617j), (-0.17284749294252105-0.004033071403588617j), 
            -0.3028797521586471, -0.3024980740453929, 0.15096258937728615, 0.15097450456756265, -0.33258467879881726, 
            -0.33287426360790684, 1.7319479184152442e-14
            """
        ),
        # ("4SITE",(0,1.0),
        # "trglC_j20_j41.0_jchi0_D3ch27_r0_LS_4SITE_c4RNDn2_C4X4cS_ptol8_state.json",
        # """
        #  -0.5301114707097865, 0.2830186205720097, 0.2813041259117088, 0.2845940989015287, 0.2856075188804097, 
        #  0.28056873859439174, (0.064409627622395+0j), (0.27383067624897484-0.00041457149447410134j), 
        #  (0.27383067624897484+0.00041457149447410134j), (0.07113462868430598+0j), (0.27556022980968153-0.00047484603243146557j), 
        #  (0.27556022980968153+0.00047484603243146557j), (-0.07342859630370174+0j), (-0.2760068873239239+0.00030697825715491603j), 
        #  (-0.2760068873239239-0.00030697825715491603j), (-0.06626942007757877+0j), (-0.272630028325782+0.000220667425893222j), 
        #  (-0.272630028325782-0.000220667425893222j), 0.038480907327594675, -0.26579427293657343, 0.03776861880749622, 
        #  -0.3102408206123273, -0.06699006502243406, -0.07981284520822408, -0.07627001502257819, -0.07911528502065476, 
        #  -0.26694116570300264, 0.050130636793616394, -0.308434886816075, 0.04206050612170027, 1.91034529528622e-07
        # """
        # ),
        # ("4SITE_T",(0,0.15),
        # "trglC_j40_jchi0.15_D2ch24_r0_LS_4SITE-T_iRND_C4X4cS_ptol8_state.json",
        # """
        # -0.5040932396346516, 0.05917211046943889, 0.05911282849600335, 0.059225334801918286, 0.05911889722927268, 
        # 0.05923138135056125, (-0.0008116723907531931+0j), (0.0558529551666207-0.019342054696527274j), 
        # (0.0558529551666207+0.019342054696527274j), (0.0007997148571636525+0j), (-0.05591256371908876+0.019513737645704483j),
        # (-0.05591256371908876-0.019513737645704483j), (0.0008625498107922291+0j), (-0.05587048775030575+0.019307734615229784j), 
        # (-0.05587048775030575-0.019307734615229784j), (-0.0008981568735706436+0j), (0.05591974764250341-0.019507221086684998j), 
        # (0.05591974764250341+0.019507221086684998j), -0.1856424821015755, -0.13152900369834097, -0.13152528966951377, 
        # -0.1856043732712988, -0.18581280831086608, -0.13147556648106623, -0.1314715789028623, -0.18586500247883728, 
        # -0.1630215234904121, -0.1768414704307892, -0.177011664770302, -0.16313177805750956, -5.079896392423677e-09
        # """
        # )
        ("4SITE",(0,0.8),
            "trgl_j20_j40.8_D2ch18_r0_4SITE_iD1j408n_state.json",
            """
            -0.4504559458647691, 0.3327944490578174, 0.33410079205406085, 0.3313594878933706, 0.3342294750500981, 
            0.3314880412337402, -0.1357247099868382, 0.30529025917991465, 0.30529025917991465, 0.13426449334702684, 
            -0.30293919529044605, -0.30293919529044605, 0.13644472544782135, -0.30511010945187117, -0.30511010945187117, 
            -0.1359160113409509, 0.30234278450485363, 0.30234278450485363, -0.35573161918523405, 0.07769600135248506, 
            -0.35517283586938314, 0.07768565739629232, -0.07788900536205548, -0.08435831053525183, -0.07789157383261866, 
            -0.08403875938738742, 0.09541323992291861, -0.34919723473679676, 0.09554199812235141, -0.349580454695963, 
            3.673061854669868e-12
            """
        )
        ]

    def setUp(self):
        args.j1= 1.0
        args.chi=27
        args.GLOBALARGS_dtype= "complex128"

    def test_ctmrg_d3_trgl_2site(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                args.tiling= ansatz[0]
                args.instate= self.DIR_PATH+"/../../test-input/"+ansatz[2]
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
                args.j2, args.j4= ansatz[1]
                
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
                    assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log"]:
                if os.path.isfile(f): os.remove(f)

class TestCtmrg_j1j2jXenergy_TRGL_D3_2SITE(unittest.TestCase):
    tol= 1.0e-4
    tol_high= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-ctmrg_d3-trgl_2site"
    ANSATZE= [
        ("2SITE","trglC_j20.1_j40_D2ch24_r0_LS_2SITE_iRND_C4X4cS_ptol8_state.json"),
        # ("2SITE","trglC_j20.1_j40_D3ch27_r0_LS_2SITE_iRND_C4X4cS_ptol8_state.json"),
        ]

    def setUp(self):
        args.j1= 1.0
        args.j2= 0.1
        args.bond_dim=2
        args.chi=24
        args.GLOBALARGS_dtype= "complex128"
        cfg.configure(args)

    def test_j1j2jX_energy_impl_d3_trgl_2site(self):
        from cmath import isclose
        import numpy as np

        I= torch.eye(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

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

                model= spin_triangular.J1J2J4(j1=1., j2=0, global_args=cfg.global_args)
                energy_nn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                model= spin_triangular.J1J2J4(j1=0, j2=1., global_args=cfg.global_args)
                energy_nnn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                nn_h_v,nn_diag,nnn=0.,0.,0.
                for coord in state.sites.keys():
                    _nn_h_v,_nn_diag= spin_triangular.eval_nn_per_site(coord,state,env,model.R,model.Rinv,model.SS,model.SS)
                    _nnn= spin_triangular.eval_nnn_per_site(coord,state,env,model.R,model.Rinv,model.SS,unroll=False,
                        checkpoint_unrolled=False)
                    nn_h_v+=_nn_h_v
                    nn_diag+=_nn_diag
                    nnn+=_nnn
                nn=(nn_h_v+nn_diag)/len(state.sites)
                nnn=nnn/len(state.sites)

                assert isclose(nn,energy_nn_manual, rel_tol=self.tol_high, abs_tol=self.tol_high)
                assert isclose(nnn,energy_nnn_manual, rel_tol=self.tol_high, abs_tol=self.tol_high)

                model= spin_triangular.J1J2J4(j1=1., j2=1.0e-14, global_args=cfg.global_args)
                energy_nn_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                model= spin_triangular.J1J2J4(j1=0, j2=0, jchi=1., global_args=cfg.global_args)
                energy_chi_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                nn_h_v,nn_diag,chi=0.,0.,0.
                for coord in state.sites.keys():
                    _nn_h_v,_nn_diag,_chi= spin_triangular.eval_nn_and_chirality_per_site(coord,state,env,model.R,model.Rinv,
                        model.SS,model.SS,model.h_chi,unroll=False,checkpoint_unrolled=False)
                    nn_h_v+=_nn_h_v
                    nn_diag+=_nn_diag
                    chi+=_chi
                nn=(nn_h_v+nn_diag)/len(state.sites)
                chi=chi/len(state.sites)

                assert isclose(nn,energy_nn_manual, rel_tol=self.tol, abs_tol=self.tol)
                assert isclose(chi,energy_chi_manual, rel_tol=self.tol, abs_tol=self.tol)

                model= spin_triangular.J1J2J4(j1=0, j2=0, jchi=0, j4=1, global_args=cfg.global_args)
                energy_j4_manual= model.energy_per_site(state,env,compressed=-1,unroll=False,\
                    ctm_args=cfg.ctm_args,global_args=cfg.global_args)

                j4_term=0
                for coord in state.sites.keys():
                    _,_,_,_j4_term= spin_triangular.eval_j1j2j4jX_per_site_legacy(coord,state,env,model.R,model.Rinv,
                        model.h_nn_only,model.SS,model.h_chi,model.h_p,unroll=False,checkpoint_unrolled=False)
                    j4_term+=_j4_term
                j4_term= j4_term/len(state.sites)

                assert isclose(j4_term,energy_j4_manual, rel_tol=self.tol, abs_tol=self.tol)