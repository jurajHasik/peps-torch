import os
import context
import argparse
import numpy as np
import torch
import config as cfg
import yastn.yastn as yastn
from ipeps.ipeps_kagome_abelian import *
from ipeps.ipess_kagome_abelian import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from models import spin_half_kagome
from optim.ad_optim_lbfgs_mod import optimize_state
import scipy.io as io
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument("--yast_backend", type=str, default='torch',
    help="YAST backend", choices=['torch','torch_cpp'])
parser.add_argument("--theta", type=float, default=0, help="angle [<value> x pi] parametrizing the chiral Hamiltonian")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbor exchange coupling")
parser.add_argument("--JD", type=float, default=0, help="two-spin DM interaction")
parser.add_argument("--j1sq", type=float, default=0, help="nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--j2", type=float, default=0, help="next-nearest-neighbor exchange coupling")
parser.add_argument("--j2sq", type=float, default=0, help="next-nearest-neighbor biquadratic exchange coupling")
parser.add_argument("--jtrip", type=float, default=0, help="(SxS).S")
parser.add_argument("--jperm", type=complex, default=0+0j, help="triangle permutation")
parser.add_argument("--h", type=float, default=0, help="magnetic field")
parser.add_argument("--ansatz", type=str, default=None, help="choice of the tensor ansatz",choices=["IPEPS", "IPESS","IPESS_PG","A_2,B"])
parser.add_argument("--no_sym_up_dn", action='store_false', dest='sym_up_dn',help="same trivalent tensors for up and down triangles")
parser.add_argument("--no_sym_bond_S", action='store_false', dest='sym_bond_S',help="same bond site tensors")
parser.add_argument("--disp_corre_len", action='store_true', dest='disp_corre_len',help="display correlation length during optimization")
parser.add_argument("--CTM_check", type=str, default='Partial_energy', help="method to check CTM convergence",choices=["Energy", "SingularValue", "Partial_energy"])
parser.add_argument("--force_cpu", action='store_true', dest='force_cpu', help="force RDM contractions on CPU")
# initial state can be selected as D=3 RVB state by passing
# --ipeps_init_type RVB
parser.add_argument("--itebd", action='store_true', dest='do_itebd', help="do itebd as initial state")
parser.add_argument("--itebd_tol", type=float, default=1e-12, help="itebd truncation tol")
parser.add_argument("--no_keep_multiplets", action='store_false', dest='keep_multiplets',help="keep multiplets when performing svd")
parser.add_argument("--SU_ctm_obs_freq", type=int, default=0)
parser.add_argument("--SU_schedule", type=str, default="[[0.5,10],[0.1,5],[0.01,1]]")
args, unknown_args = parser.parse_known_args()

@torch.no_grad()
def main():
    cfg.configure(args)
    cfg.print_config()
    from yastn.yastn.sym import sym_U1
    if args.yast_backend=='torch':
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend
    settings_U1= yastn.make_config(backend=backend, sym=sym_U1, \
        default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)
    torch.set_num_threads(args.omp_cores)
    settings_U1.backend.random_seed(args.seed)

    # 0) initialize model
    if not args.theta is None:
        args.jtrip= args.j1*math.sin(args.theta*math.pi)
        args.j1= args.j1*math.cos(args.theta*math.pi)
    model= spin_half_kagome.S_HALF_KAGOME(j1=args.j1, JD=args.JD,\
        j2=args.j2, jtrip=args.jtrip, jperm=args.jperm)

    # 1) initialize the ipess/ipeps
    if args.ansatz in ["IPESS","IPESS_PG","A_2,B"]:
        ansatz_pgs= None
        if args.ansatz=="A_2,B": ansatz_pgs= IPESS_KAGOME_PG.PG_A2_B

        # 1.1) reading from instate file
        if args.instate!=None:
            if args.ansatz=="IPESS":
                state= read_ipess_kagome_generic(args.instate, settings_U1)
            #
            # TODO allow PGs
            #
            # elif args.ansatz in ["IPESS_PG","A_2,B"]:
            #     state= read_ipess_kagome_pg(args.instate)
            #
            # possibly symmetrize by PG
            # if ansatz_pgs!=None:
            #     if type(state)==IPESS_KAGOME_GENERIC:
            #         state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
            #             SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            #     elif type(state)==IPESS_KAGOME_PG:
            #         if state.pgs==None or state.pgs==dict():
            #             state= to_PG_symmetric(state, SYM_UP_DOWN=args.sym_up_dn,\
            #                 SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            #         elif state.pgs==ansatz_pgs:
            #             # nothing to do here
            #             pass
            #         elif state.pgs!=ansatz_pgs:
            #             raise RuntimeError("instate has incompatible PG symmetry with "+args.ansatz)
            state= state.add_noise(args.instate_noise)

        # 1.2) reading from checkpoint file
        elif args.opt_resume is not None:
            T_u= yastn.Tensor(config=settings_U1, s=(-1,-1,-1))
            T_d= yastn.Tensor(config=settings_U1, s=(-1,-1,-1))
            B_c= yastn.Tensor(config=settings_U1, s=(-1,1,1))
            B_a= yastn.Tensor(config=settings_U1, s=(-1,1,1))
            B_b= yastn.Tensor(config=settings_U1, s=(-1,1,1))
            if args.ansatz in ["IPESS"]:
                state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                    'B_b': B_b, 'B_c': B_c})
            #
            # TODO allow PGs
            #
            # elif args.ansatz in ["IPESS_PG", "A_2,B"]:
            #     state= IPESS_KAGOME_PG(T_u, B_c, T_d=T_d, B_a=B_a, B_b=B_b,\
            #         SYM_UP_DOWN=args.sym_up_dn,SYM_BOND_S=args.sym_bond_S, pgs=ansatz_pgs)
            state.load_checkpoint(args.opt_resume)
    if args.instate==None and args.opt_resume==None:
        args.ansatz="IPESS"
        if args.ipeps_init_type=='RANDOM':

            #su(2) sectors
            if args.bond_dim==3:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 1, 1), (1, 1, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1, 1), (1, 1, 1), (1, 1, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1, 1), (1, 1, 1), (1, 1, 1)))
            if args.bond_dim==6:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 2, 1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 2, 1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 1)))
            if args.bond_dim==8:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 2, 2, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 2, 2, 1), (1, 2, 2, 2, 1), (1, 2, 2, 2, 1)))
            if args.bond_dim==9:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 3, 2, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 2, 3, 2, 1), (1, 2, 3, 2, 1), (1, 2, 3, 2, 1)))

            #non-su(2) sectors
            elif args.bond_dim==4:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 1), (1, 2, 1), (1, 2, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 2, 1), (1, 2, 1), (1, 2, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 2, 1), (1, 2, 1), (1, 2, 1)))
            elif args.bond_dim==5:
                B_c = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2,-1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))
                B_b = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2,-1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))
                B_a = yastn.rand(config=settings_U1, s=(-1, 1, 1), n=0,
                    t=((-1, 1), (-2, -1, 0, 1, 2), (-2,-1, 0, 1, 2)),
                    D=((1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))

                T_u = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))
                T_d = yastn.rand(config=settings_U1, s=(-1, -1, -1), n=0,
                    t=((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)),
                    D=((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))
            state= IPESS_KAGOME_GENERIC_ABELIAN(settings_U1, {'T_u': T_u, 'B_a': B_a,\
                'T_d': T_d,'B_b': B_b, 'B_c': B_c})

        elif args.ipeps_init_type=="RVB":
            unit_block= np.ones((1,1,1), dtype=cfg.global_args.dtype)
            B_c= yastn.Tensor(settings_U1, s=(-1, 1, 1), n=0)
            B_c.set_block(ts=(1,1,0), Ds= unit_block.shape, val= unit_block)
            B_c.set_block(ts=(1,0,1), Ds= unit_block.shape, val= unit_block)
            B_c.set_block(ts=(-1,-1,0), Ds= unit_block.shape, val= unit_block)
            B_c.set_block(ts=(-1,0,-1), Ds= unit_block.shape, val= unit_block)
            B_b=B_c.copy()
            B_a=B_c.copy()

            unit_block= np.ones((1,1,1), dtype=cfg.global_args.dtype)
            T_u= yastn.Tensor(settings_U1, s=(-1, -1, -1), n=0)
            T_u.set_block(ts=(1,-1,0), Ds= unit_block.shape, val= unit_block)
            T_u.set_block(ts=(-1,1,0), Ds= unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(0,1,-1), Ds= unit_block.shape, val= unit_block)
            T_u.set_block(ts=(0,-1,1), Ds= unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(-1,0,1), Ds= unit_block.shape, val= unit_block)
            T_u.set_block(ts=(1,0,-1), Ds= unit_block.shape, val= -1*unit_block)
            T_u.set_block(ts=(0,0,0), Ds= unit_block.shape, val= unit_block)
            T_d=T_u.copy()
            state= IPESS_KAGOME_GENERIC_ABELIAN(settings_U1, {'T_u': T_u, 'B_a': B_a,\
                'T_d': T_d,'B_b': B_b, 'B_c': B_c})
            state= state.add_noise(args.instate_noise)

    # 2) (optional) perform iTEBD on top of the initial state
    if args.do_itebd:
        print("-"*20 + " iTEBD initialization "+"-"*20)
        from examples.abelian.SU_kagome_spin_half_u1 import main as itebd_main
        state.write_to_file(args.out_prefix+"_state.json")
        args.instate= args.out_prefix+"_state.json"
        itebd_main(args=args)
        state= read_ipess_kagome_generic(args.out_prefix+"_state.json", settings_U1)

    # 3) define auxilliary functions
    def energy_f(state, env, force_cpu=False, fail_on_check=False,\
        warn_on_check=True):
        #print(env)
        e_dn,n_dn = model.energy_triangle_dn(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        e_up,n_up = model.energy_triangle_up(state, env, force_cpu=force_cpu,\
            fail_on_check=fail_on_check, warn_on_check=warn_on_check)
        # e_nnn = model.energy_nnn(state, env)
        return (sum(e_up.values()) + sum(e_dn.values()))/(3*len(state.sites)) #+ e_nnn) / 3
    def energy_f_complex(state, env, force_cpu=False):
        e_dn,n_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        e_up,n_up = model.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
        # e_nnn = model.energy_nnn(state, env)
        return (sum(e_up.values()) + sum(e_dn.values()))/(3*len(state.sites)) #+ e_nnn) / 3
    def dn_energy_f_NoCheck(state, env, force_cpu=False):
        e_dn,n_dn = model.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
        return sum(e_dn.values())/len(state.sites)

    @torch.no_grad()
    def print_corner_spectra(env):
        spectra = []
        for c_loc,c_ten in env.C.items():
            c_ten=c_ten.to_dense()
            #print(torch.Tensor.size(c_ten))
            #print(c_ten)
            u,s,v= torch.svd(c_ten, compute_uv=False)
            # none, none, none, s=yastn.svd(c_ten, axes=((0),(1)), untruncated_S=True)
            # print(s)
            # s=s.to_dense()
            # s=torch.diag(s)
            # print(s)
            if c_loc[1] == (-1, -1):
                label = 'LU'
            if c_loc[1] == (-1, 1):
                label = 'LD'
            if c_loc[1] == (1, -1):
                label = 'RU'
            if c_loc[1] == (1, 1):
                label = 'RD'
            spectra.append([label, s])
        return spectra

    if args.CTM_check=="Energy":
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            e_curr = energy_f_complex(state, env, force_cpu=ctm_args.conv_check_cpu)
            history.append(e_curr.item())
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="Partial_energy":
        def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
            if not history:
                history = []
            if len(history)>8:
                e_curr = dn_energy_f_NoCheck(state, env, force_cpu=ctm_args.conv_check_cpu)
                history.append(e_curr.item())
            else:
                history.append(len(history)+1)
            #print(history)
            if (len(history) > 1 and abs(history[-1] - history[-2]) < ctm_args.ctm_conv_tol*2) \
                    or len(history) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history), "history": history})
                return True, history
            return False, history
    elif args.CTM_check=="SingularValue":
        ctmrg_conv_f= ctmrg_conv_specC

    print(state)
    # convert to dense
    state_d= state.to_dense()
    print(state_d)

    # 4) compute initial environment and observables
    # ctm_env_init= ENV_ABELIAN(args.chi, state=state, init=True)
    ctm_env_init = ENV(args.chi, state_d)
    init_env(state_d, ctm_env_init)
    ctm_env_init, history, t_ctm, t_conv_check = ctmrg.run(state_d, ctm_env_init, \
        conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)

    loss0 = energy_f(state_d, ctm_env_init, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state_d,ctm_env_init,force_cpu=args.force_cpu,\
        disp_corre_len=args.disp_corre_len)
    print(", ".join(["epoch",f"loss"]+[label for label in obs_labels]))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))


    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        state_d= state.to_dense()

        # build on-site tensors
        if args.ansatz in ["IPESS", "IPESS_PG", "A_2,B"]:
            if args.ansatz in ["IPESS"]:
                state_sym= state_d
                state_sym.sites= state_sym.build_onsite_tensors()
            #
            # TODO allow PGs
            #
            # elif args.ansatz in ["IPESS_PG", "A_2,B"]:
            #     # symmetrization and implicit rebuild of on-site tensors
            #     state_sym= to_PG_symmetric(state, state.pgs)
        else:
            A= state_d.sites[(0,0)]
            A= A/A.abs().max()
            state_sym= IPEPS_KAGOME({(0,0): A}, lX=1, lY=1)

        # 1) re-build precomputed double-layer on-site tensors
        #    Some objects, in this case open-double layer tensors, are pre-computed
        # state_sym.sync_precomputed()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)
        # compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        loss = energy_f(state_sym, ctm_env_out, force_cpu=cfg.ctm_args.conv_check_cpu)

        return loss, ctm_env_out, history, t_ctm, t_conv_check

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state_d= state.to_dense()
        state_sym= state_d
        if args.ansatz in ["A_2,B"]:
            state_sym= to_PG_symmetric(state, state.pgs)
        elif args.ansatz in ["IPESS"]:
            state_sym.sites= state_sym.build_onsite_tensors()
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
        if opt_context["line_search"]:
            print(", ".join([f"{epoch}",f"{loss}"]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state_sym.sites.items()]))
        else:
            obs_values, obs_labels = model.eval_obs(state_sym,ctm_env,force_cpu=args.force_cpu,\
                disp_corre_len=args.disp_corre_len)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]), end="")
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state_sym.sites.items()]))
            print(" "+", ".join([f"{t.norm()}" for c,t in state_sym.sites.items()]) )

    # 4) optimize
    optimize_state(state, ctm_env_init, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    if args.ansatz=="IPESS":
        state= read_ipess_kagome_generic(outputstatefile, settings_U1)
    #
    # TODO allow PGs
    #
    # elif args.ansatz in ["IPESS_PG","A_2,B"]:
    #     state= read_ipess_kagome_pg(outputstatefile)
    state_d= state.to_dense()
    ctm_env = ENV(args.chi, state_d)
    init_env(state_d, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state_d, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state_d, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state_d,ctm_env)
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{opt_energy}"]+[f"{v}" for v in obs_values]))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestOptim_RVB(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt_u1_RVB"

    def setUp(self):
        args.theta=0.2
        args.j1=1.0
        args.bond_dim=3
        args.chi=27
        args.opt_max_iter= 5
        args.seed=1
        args.instate_noise=0.02
        args.out_prefix=self.OUT_PRFX
        args.GLOBALARGS_dtype= "complex128"
        args.ipeps_init_type="RVB"

    def test_basic_opt_rvb(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        with patch('sys.stdout', new = StringIO()) as tmp_out:
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        final_obs=None
        final_opt_line=None
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                final_opt_line= l.rstrip()
            if "epoch, loss," in l and not OPT_OBS_DONE:
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs
        assert final_opt_line

        # compare with the reference
        ref_data="""
        -0.3180413202889958, (-0.47713120439278667+3.639623534397078e-15j), (-0.4769927564742007+0j),
        (0.0026971891280391123+1.5021174805076658e-17j), (2.953928227115468e-16+3.389897989359694e-17j),
        (3.4201498958040513e-16-1.6468002706981575e-18j), (0.0027083902082258096-3.4755919552534e-16j),
        (-2.362926078045056e-16+1.5088297192245105e-16j), (-9.96011820377404e-16-1.3731797741619337e-16j),
        (-0.005405578567283862+2.1355452589085543e-16j), (-1.0376523478434083e-16-4.722156860876956e-16j),
        (4.2601775076768687e-16-2.782213398973379e-16j), (7.274829192412387e-06+8.10298987492667e-20j),
        (7.335377520013444e-06-1.882651843879459e-18j), (2.9220279647078648e-05-2.3087715362043214e-18j),
        (-0.15907030195628583+1.1943046817711557e-15j), (-0.16060958285806728+6.565741657327078e-16j),
        (-0.15744582552237282+1.820532263827809e-15j), (-0.1574280729028653-2.710505431213761e-20j),
        (-0.16058888687471531+0j), (-0.15897179122755842+3.3881317890172014e-21j)
        """
        # compare final observables from optimization and the observables from the
        # final state
        final_opt_line_t= [complex(x) for x in final_opt_line.split(",")[1:7]]
        fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")[:6]]
        for val0,val1 in zip(final_opt_line_t, fobs_tokens):
            assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

        # compare final observables from final state against expected reference
        # drop first token, corresponding to iteration step
        ref_tokens= [complex(x) for x in ref_data.split(",")[:6]]
        for val,ref_val in zip(fobs_tokens, ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)