import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from ipeps.ipeps_c4v_thermal import *
from ipeps.ipeps_c4v_thermal import _build_elem_t
from linalg.custom_eig import truncated_eig_sym
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v_thermal import entropy, rdm1x1_sl, rdm2x1_sl, rdm2x2
# from ctm.one_site_c4v import transferops_c4v
from optim.ad_optim_lbfgs_mod import optimize_state
from models import j1j2
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--u1_class", type=str, default="B")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--j3", type=float, default=0., help="next-to-next nearest-neighbour coupling")
parser.add_argument("--beta", type=float, default=0., help="inverse temperature")
parser.add_argument("--hz_stag", type=float, default=0., help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1., help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args= parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE_THERMAL(j1=args.j1, j2=args.j2, j3=args.j3, \
        hz_stag=args.hz_stag, delta_zz=args.delta_zz, beta=args.beta)
    
    energy_f= model.energy_1x1
    def approx_S(state, ctm_env, ad_decomp_reg=cfg.ctm_args.ad_decomp_reg, force_cpu=False):
        return entropy( rdm2x2(state, ctm_env), ad_decomp_reg )/4.

    

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_thermal_lc(args.instate, vertexToSite=None)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"

        # TODO extending from smaller bond-dim to higher bond-dim is 
        # currently not possible

        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [4]:
            elem_t= _build_elem_t()
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(elem_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_C4V_THERMAL_LC(elem_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type in ['INFT','RANDOM']:
        if args.bond_dim in [4]:
             elem_t= _build_elem_t()
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        if args.ipeps_init_type=='INFT':
            A= torch.zeros(len(elem_t), dtype=cfg.global_args.torch_dtype, \
                device=cfg.global_args.device)
            A[4]= 1.
        if args.ipeps_init_type=='RANDOM':
            A= torch.rand(len(elem_t), dtype=cfg.global_args.torch_dtype, \
                device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))
        coeffs = {(0,0): A}
        state = IPEPS_C4V_THERMAL_LC(elem_t, coeffs)
        state.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        # unfuse ancilla+physical
        dims= state.site().size()
        _tmp_site= state.site().view( 2,2, *dims[1:])
        _tmp_state= IPEPS_C4V_THERMAL(_tmp_site)
        rdm2x1= rdm2x1_sl(_tmp_state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"]=rdm2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return True, history
        elif len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return False, history
        return False, history

    @torch.no_grad()
    def ctmrg_conv_f2(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        U, specC, V= torch.svd(env.get_C(), compute_uv=False)
        dist= float('inf')
        if len(history["log"]) > 1:
            dist= torch.dist(specC, history["specC"], p=2).item()
        history["specC"]=specC
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return True, history
        elif len(history['log']) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log'],
                "final_multiplets": compute_multiplets(ctm_env)})
            return False, history
        return False, history

    def renyi2_site(a, D1, D2):
        # a rank-6 on-site tensor with indices [apuldr]
        # 0) build projectors
        #
        #    ||/p0       
        #  ==a*a== -> ==a*a== = P D P^H
        # p1/||          
        #
        tmp_M= torch.einsum('apuldr,apufdh->lfrh',a,a.conj()).contiguous().view(\
            a.size(3)**2,a.size(5)**2)
        D, P= truncated_eig_sym(tmp_M, D1, keep_multiplets=True,\
                verbosity=cfg.ctm_args.verbosity_projectors)
        #
        # 1) build 1st layer
        #
        #      |
        #      P
        #      ||/p0        |/p0
        # --P==a*a==P-- = --l1--
        #   p1/||        p1/|
        #      P
        #      |
        #
        l1= torch.einsum('apuldr,asefgh->psuelfdgrh',a,a.conj()).contiguous().view(\
            a.size(1), a.size(1),a.size(2)**2,a.size(3)**2,a.size(4)**2,a.size(5)**2)
        l1= torch.einsum('psxywz,xu,yl,wd,zr->psuldr',l1, P, P, P, P).contiguous()
        #
        # 2) build 2nd layer
        #
        #   p0 
        #   |/        ||  
        # --l1-- => ==l2==
        #  /|         ||
        #   p1
        #   |/
        # --l1--
        #  /p0
        #
        l2= torch.einsum('psuldr,spxywz->uxlydwrz',l1,l1).contiguous().view(
            (D1**2,)*4)

        # prepare initial environment tensors
        #
        # C--, --T--
        # |      |
        #
        C_0= torch.einsum('psuldr,spulwz->dwrz',l1,l1).contiguous().view(
            (D1**2,)*2)
        T_0= torch.einsum('psuldr,spuywz->lydwrz',l1,l1).contiguous().view(
            (D1**2,)*3)

        #
        # 3) (optional) truncate auxiliary bonds
        if D2<D1**2:
            tmp_M= l2.view( (D1,D1,D1**2)*2 )
            tmp_M= torch.einsum('ij,l,ij,r->lr').contiguous()
            D, P= truncated_eig_sym(tmp_M, D2, keep_multiplets=True,\
                verbosity=cfg.ctm_args.verbosity_projectors)
            l2= torch.einsum('xywz,xu,yl,wd,zr->uldr',l2, P, P, P, P).contiguous()
            C_0= torch.einsum('wz,wd,zr',C_0,P,P).contiguous()
            T_0= torch.einsum('ywz,yl,wd,zr',T_0,P,P,P).contiguous()

        return l2, C_0, T_0

    def approx_renyi2(state,D1,D2):
        # get intial renyi2 on-site tensor and env tensors
        l2, C0, T0= renyi2_site(state.site(),D1,D2)
        state_r2= IPEPS_C4V(l2)
        env_r2= ENV_C4V(args.chi, state_r2)
        init_env(None, env_r2, (C0,T0))

        # run CTM
        env_r2, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_r2, env_r2, \
            conv_check=ctmrg_conv_f2)

        # parition function per site <=> Tr(\rho^2)/N
        #
        # C--T--C            / C--C
        # |  |  |   C--C    /  |  |   C--T--C
        # T--A--T * |  |   /   T--T * |  |  |
        # |  |  |   C--C  /    |  |   C--T--C
        # C--T--C        /     C--C

        C= env_r2.get_C()
        T= env_r2.get_T()
        A= state_r2.site()

        # closed C^4
        C4= torch.einsum('ij,jk,kl,li',C,C,C,C)

        # closed rdm1x1
        CTC = torch.tensordot(C,T,([0],[0]))
        #   C--0
        # A |
        # | T--2->1
        # | 1
        #   0
        #   C--1->2
        CTC = torch.tensordot(CTC,C,([1],[0]))
        rdm = torch.tensordot(CTC,T,([2],[0]))
        rdm = torch.tensordot(rdm,state_r2.site(),([1,3],[1,2]))
        rdm = torch.tensordot(T,rdm,([1,2],[0,2]))
        rdm = torch.tensordot(rdm,CTC,([0,1,2],[2,0,1]))

        # closed CTCCTC
        #   C--0 2--C
        # A |       |
        # | T--1 1--T |
        #   |       | V
        #   C--2 0--C
        CTC= torch.tensordot(CTC,CTC,([0,1,2],[2,1,0]))

        renyi2= -torch.log((rdm / CTC) * (C4 / CTC))
        return renyi2


    # 0) fuse ancilla+physical index and create regular c4v ipeps with rank-5 on-site
    #    tensor
    state_fused= state.to_fused_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)

    e0 = energy_f(state, ctm_env, force_cpu=True)
    S0 = approx_S(state, ctm_env)
    r2_0 = approx_renyi2(state, args.bond_dim, args.bond_dim**2)
    # loss0 = e0 - 1./args.beta * S0
    loss0 = e0 - 1./args.beta * r2_0
    obs_values, obs_labels = model.eval_obs(state, ctm_env,force_cpu=True)
    print(", ".join(["epoch","loss","e0","S0","r2_0"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}",f"{e0}",f"{S0}",f"{r2_0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build on-site tensors
        state.sites= state.build_onsite_tensors()
        state_fused= state.to_fused_ipeps_c4v()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_fused, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_fused, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        e0 = energy_f(state, ctm_env_out, force_cpu=True)
        # S0 = approx_S(state, ctm_env_out)
        # loss0 = e0 - 1./args.beta * S0
        
        loc_ctm_args= copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run_dl(state_fused, ctm_env_out, \
            ctm_args=loc_ctm_args)
        e1 = energy_f(state, ctm_env_out, force_cpu=True)
        # S1 = approx_S(state, ctm_env_out)
        # loss1 = e1 - 1./args.beta * S1

        r2_0 = approx_renyi2(state, args.bond_dim, args.bond_dim**2)
        loss= torch.max(e0,e1) - 1./args.beta * r2_0
        # loss= torch.max(loss0,loss1)

        return loss, ctm_env_out, history, t_ctm, t_obs

    def _to_json(l):
        re=[l[i,0].item() for i in range(l.size()[0])]
        im=[l[i,1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        ctm_args= opt_context["ctm_args"]
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        e0 = energy_f(state, ctm_env, force_cpu=True)
        S0 = approx_S( state, ctm_env )
        r2_0 = approx_renyi2(state, args.bond_dim, args.bond_dim**2)
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
        print(", ".join([f"{epoch}",f"{loss}",f"{e0}", f"{S0}", f"{r2_0}"]+[f"{v}" for v in obs_values]))

        if (not opt_context["line_search"]) and args.top_freq>0 and epoch%args.top_freq==0:
            coord_dir_pairs=[((0,0), (1,0))]
            for c,d in coord_dir_pairs:
                # transfer operator spectrum
                print(f"TOP spectrum(T)[{c},{d}] ",end="")
                l= transferops_c4v.get_Top_spec_c4v(args.top_n, state, ctm_env)
                print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_thermal_lc(outputstatefile)
    state_fused= state.to_fused_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run_dl(state_fused, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env,force_cpu=True)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=True)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=3
        args.chi=18
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. ARNOLDISVD is not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_SYMEIG_LS(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()

    def test_opt_SYMEIG_LS_SYMARP(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        args.OPTARGS_line_search_svd_method="SYMARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_LS_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_LS_SYMARP_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        args.OPTARGS_line_search_svd_method="SYMARP"
        main()
