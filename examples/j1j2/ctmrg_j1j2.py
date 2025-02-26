import os
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from models import j1j2
import unittest

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--j3", type=float, default=0, help="next-to-next nearest-neighbour coupling")
parser.add_argument("--lmbd", type=float, default=0, help="chiral plaquette interaction")
parser.add_argument("--hz_stag", type=float, default=0, help="staggered mag. field")
parser.add_argument("--h_uni", nargs=3, type=float, default=[0,0,0], help="uniform mag. field with components in directions h^z, h^x, h^y")
parser.add_argument("--delta_zz", type=float, default=1, help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["BIPARTITE", "1SITE", "2SITE", "4SITE", "8SITE"])
# additional observables-related arguments
parser.add_argument("--corrf_r", type=int, default=1, help="maximal correlation function distance")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
    
    model = j1j2.J1J2(j1=args.j1, j2=args.j2, j3=args.j3, lmbd=args.lmbd,
        hz_stag=args.hz_stag, h_uni=args.h_uni, delta_zz=args.delta_zz)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "1SITE":
        def lattice_to_site(coord):
            return (0, 0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
    elif args.tiling == "4SITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 2) % 2
            return (vx, vy)
    elif args.tiling == "8SITE":
        def lattice_to_site(coord):
            shift_x = coord[0] + 2*(coord[1] // 2)
            vx = shift_x % 4
            vy = coord[1] % 2
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)

        # normalization of initial random tensors
        A = A/torch.max(torch.abs(A))
        sites = {(0,0): A}
        if args.tiling in ["BIPARTITE", "2SITE", "4SITE", "8SITE"]:
            B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            sites[(1,0)]= B/torch.max(torch.abs(B))
        if args.tiling in ["4SITE", "8SITE"]:
            C= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            D= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            sites[(0,1)]= C/torch.max(torch.abs(C))
            sites[(1,1)]= D/torch.max(torch.abs(D))
        if args.tiling == "8SITE":
            E= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            F= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            G= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            H= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            sites[(2,0)] = E/torch.max(torch.abs(E))
            sites[(3,0)] = F/torch.max(torch.abs(F))
            sites[(2,1)] = G/torch.max(torch.abs(G))
            sites[(3,1)] = H/torch.max(torch.abs(H))
        state = IPEPS(sites, vertexToSite=lattice_to_site)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.torch_dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.torch_dtype} and reinitializing "\
        +" the model")
        model= j1j2.J1J2(j1=args.j1, j2=args.j2, j3=args.j3, lmbd=args.lmbd,
            hz_stag=args.hz_stag, h_y=args.hy, delta_zz=args.delta_zz)

    print(state)

    # 2) select the "energy" function 
    if args.tiling == "BIPARTITE" or args.tiling == "2SITE":
        energy_f= model.energy_2x2_2site
        eval_obs_f= model.eval_obs
    elif args.tiling == "1SITE":
        energy_f= model.energy_2x2_1site_BP
        eval_obs_f= model.eval_obs_1site_BP
    elif args.tiling == "4SITE":
        energy_f= model.energy_2x2_4site
        eval_obs_f= model.eval_obs
    elif args.tiling == "8SITE":
        energy_f= model.energy_2x2_8site
        eval_obs_f= model.eval_obs
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE")

    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history=[]
            e_curr = energy_f(state, env)
            obs_values, obs_labels = eval_obs_f(state, env)
            history.append([e_curr.item()]+obs_values)
            print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

            if len(history) > 1 and abs(history[-1][0]-history[-2][0]) < ctm_args.ctm_conv_tol:
                return True, history
        return False, history

    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)
    print(ctm_env_init)

    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)

    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))

    ctm_env_init, *ctm_log= ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy)

    # 6) compute final observables
    e_curr0 = energy_f(state, ctm_env_init)
    obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # 7) ----- additional observables ---------------------------------------------
    # TODO correct corrf_SS for 1site ansatz - include rotation on B-sublattice
    corrSS= model.eval_corrf_SS((0,0), (1,0), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(1,0)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    corrSS= model.eval_corrf_SS((0,0), (0,1), state, ctm_env_init, args.corrf_r)
    print("\n\nSS[(0,0),(0,1)] r "+" ".join([label for label in corrSS.keys()]))
    for i in range(args.corrf_r):
        print(f"{i} "+" ".join([f"{corrSS[label][i]}" for label in corrSS.keys()]))

    # environment diagnostics
    print("\n")
    for c_loc,c_ten in ctm_env_init.C.items(): 
        u,s,v= torch.svd(c_ten, compute_uv=False)
        print(f"spectrum C[{c_loc}]")
        for i in range(args.chi):
            print(f"{i} {s[i]}")

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


class TestCtmrgBasic(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.CTMARGS_ctm_max_iter=2

    # basic tests
    def test_ctmrg_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_ctmrg_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ctmrg_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()


class TestCtmrg_States(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_CTMRGJ1J2_"
    ANSATZE= [
        ("BIPARTITE",{"j3": 0.125, "h_uni": [3.9,0,0]},
        "BIPARTITE_j2_0_j3_1250_h_39000_D_3_chi_32_seed_100_state.json",
        """
        -1.3896897615463615, 0.4884474386344192, 0.48844697363007333, 0.4884479036387651, 
        -0.46200561021924863, 0.1585284270227813, 0.1585284270227813, -0.4620060817268178, 
        -0.15852991836412875, -0.15852991836412875, 0.1751621217404098, 0.17516618332251627, 
        0.17516347390256323, 0.17516132836311246
        """
        ),
        ("2SITE",{"j2": 0.55},
        "gesdd-D2-chi50-j20.55-run0-iRND2x1_state.json",
        """
        -0.4434603770143078, 0.3184895704619597, 0.31842030538406385, 0.31855883553985553, 
        -0.26397659399034457, 0.17806697814624955, 0.1780669781462514, 0.26446699770693394, 
        -0.17758642635176156, -0.1775864263517598
        """
        )
        ]

    def setUp(self):
        args.j1, args.j2, args.j3, args.h_uni= 1.0, 0, 0, [0,0,0]
        args.CTMARGS_ctm_max_iter=100
        args.chi=32

    def test_ctmrgj1j2_states(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                self.setUp()
                args.tiling= ansatz[0]
                args.instate= self.DIR_PATH+"/../../test-input/"+ansatz[2]
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
                for k in ansatz[1]: args.__setattr__(k,ansatz[1][k])
                
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
                fobs_tokens= [float(x) for x in final_obs[len("FINAL"):].split(",")]
                ref_tokens= [float(x) for x in ref_data.split(",")]
                for val,ref_val in zip(fobs_tokens, ref_tokens):
                    assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.instate=None
        for ansatz in self.ANSATZE:
            out_prefix=self.OUT_PRFX+f"_{ansatz[0]}"
            for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
                out_prefix+".log"]:
                if os.path.isfile(f): os.remove(f)