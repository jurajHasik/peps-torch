import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from ipeps.ipeps_c4v_thermal import *
from linalg.custom_eig import truncated_eig_sym
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v_thermal import entropy, rdm1x1_sl, rdm2x1_sl, rdm2x2
from ctm.one_site_c4v import transferops_c4v
from optim.ad_optim_lbfgs_mod import optimize_state
from models import ising
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--l2d", type=int, default=1)
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--hz", type=float, default=0., help="longitudinal field")
parser.add_argument("--q", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--beta", type=float, default=0., help="inverse temperature")
parser.add_argument("--layers", type=int, default=1)
parser.add_argument('--layers_Ds', nargs="+", type=int)
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--logz", action='store_true')
parser.add_argument("--mode", type=str, default="dl")
args, unknown_args= parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # 0) initialize model
    model = ising.ISING_C4V(hx=args.hx, hz=args.hz, q=args.q)
    assert args.q==0,"plaquette term is not supported"
    energy_f= model.energy_1x1_nn_thermal
    eval_obs_f= model.eval_obs_thermal

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_c4v_thermal_ttn(args.instate)
        # state.add_noise(args.instate_noise)
        state.seed_site= model.ipepo_trotter_suzuki(args.beta/(2**args.layers))
        if len(state.isometries)< args.layers:
            assert len(args.layers_Ds)==args.layers,"Incompatible number of layers"
            assert state.iso_Ds == args.layers_Ds[:len(state.isometries)],\
                "Dimensions of existing layers do not match"
            # add more isometry layers besides the ones included in instate
            Ds_out= [state.seed_site.size(2)]+list(args.layers_Ds)
            new_iso= [ torch.rand([Ds_out[i]]*4,\
                dtype=model.dtype, device=model.device) \
                for i in range(len(state.isometries),args.layers) ]
            state.extend_layers(new_iso)
    elif args.opt_resume is not None:
        if args.bond_dim in [4]:
            elem_t= _build_elem_t()
        else:
            raise ValueError("Unsupported --bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(elem_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_C4V_THERMAL_LC(elem_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type in ['RANDOM','SVD','ID','CONST']:
        assert args.layers>0,"number of layers must be larger than 0"
        # 1-layers: tower of 2 ipepo's & single non-equivalent isometry
        # 2-layers: tower of 2**2 ipepo's & two non-equivalent isometries
        # 3-layers: tower of 2**3 ipepo's & three non-equivalent isometries
        # ...
        from linalg.hosvd import hosvd
        A= model.ipepo_trotter_suzuki(args.beta/(2**args.layers))
        if args.ipeps_init_type=='RANDOM':
            if args.layers>1:
                assert len(args.layers_Ds)==args.layers,\
                    "reduced Ds of isometries must be provided"
            # output dimensions of isometries, starting with aux bond dim of ipepo site
            Ds_out= [A.size(2)] + args.layers_Ds
            isometries=[ torch.rand([Ds_out[i]]*4,\
                dtype=model.dtype, device=model.device) \
                for i in range(args.layers) ]
        if args.ipeps_init_type=='SVD':
            assert len(args.layers_Ds)==args.layers,\
                "reduced Ds of isometries must be provided"
            redD_iso= [A.size(0)] + args.layers_Ds
            isometries=[]
            B= A.clone()
            for i in range(args.layers):
                #
                #   pbc
                #  --A-- = --E--
                #  --A--
                #   pbc
                # 
                E= torch.einsum('spuldr,psxyzw->uxlydzrw',B,B).contiguous()\
                    .view([B.size(5)**2]+[B.size(5)**6])
                U,S,Vh= torch.linalg.svd(E,full_matrices=False)

                # init_iso= U[:,:redD_iso[i]].reshape(B.size(5),B.size(5),redD_iso[i]).contiguous()
                parent_iso= U @ torch.diag(S) @ Vh[:,:S.size(0)]
                isometries.append( parent_iso.view([redD_iso[i]]*4) )
                init_iso= U[:,:redD_iso[i+1]].view([redD_iso[i]]*2+[redD_iso[i+1]])
                #   
                #            |/
                #     /--tmp_A--\
                # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
                #     \x       y/
                #
                #              a
                #            |/
                #     /--tmp_A--\
                # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
                #     \x   b   y/
                #
                tmp_B_lr= torch.einsum('mxl,nyr,spambn->spalxbry',init_iso,init_iso,B)
                #
                #             a   u
                #              \ /  
                #               U
                #             |/
                #      x--tmp_A--y
                #            /|  
                #        b\ /
                #          U
                #         / 
                #        d
                #
                tmp_B_ud= torch.einsum('amu,bnd,spmxny->spauxbdy',init_iso,init_iso,B)
                B= torch.einsum('skalxbry,kpauxbdy->spuldr',tmp_B_lr,tmp_B_ud).contiguous()
        elif args.ipeps_init_type=='ID':
            redD_iso= [A.size(2)] + [ A.size(0)**(2**i) for i in range(1,args.layers) ]
            isometries=[]
            for i in range(1,args.layers):
                isometries.append( 
                    torch.eye(redD_iso[i],dtype=model.dtype, device=model.device)\
                        .view(redD_iso[i-1],redD_iso[i-1],redD_iso[i])
                )
                isometries[-1] += args.instate_noise*(torch.rand_like(isometries[-1])-1.0)
        elif args.ipeps_init_type=='CONST':
            isometries=[]
            A0= A.clone()
            for i in range(1,args.layers):
                A= torch.einsum("sxuldr,xpefgh->spuelfdgrh",A,A0).contiguous()
                A= A.view([model.phys_dim]*2 + [A0.size(5)**(i+1)]*4)
        state = IPEPS_C4V_THERMAL_TTN(A,iso_Ds=args.layers_Ds,isometries=isometries)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    # initial normalization
    print(f"norm(seed_site) {state.seed_site.abs().max().item()}")
    norm0= state.site().norm()
    # state.seed_site= state.seed_site/torch.sqrt(norm0)
    state.metadata= { "origin": str(cfg.main_args)+str(cfg.global_args)
        +str(cfg.peps_args)+str(cfg.ctm_args)+str(cfg.opt_args),
        "sqrt(norm0)": torch.sqrt(norm0).item() }

    @torch.no_grad()
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
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
    def ctmrg_conv_Cspec(state, env, history, ctm_args=cfg.ctm_args):
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

    ctmrg_conv_f= ctmrg_conv_Cspec

    # def renyi2_site(a, D1, D2):
    #     # a rank-6 on-site tensor with indices [apuldr]
    #     # 0) build projectors
    #     #
    #     #    ||/p0       
    #     #  ==a*a== -> ==a*a== = P D P^H
    #     # p1/||          
    #     #
    #     tmp_M= torch.einsum('apuldr,apufdh->lfrh',a,a.conj()).contiguous().view(\
    #         a.size(3)**2,a.size(5)**2)
    #     D, P= truncated_eig_sym(tmp_M, D1, keep_multiplets=True,\
    #             verbosity=cfg.ctm_args.verbosity_projectors)
    #     # import pdb; pdb.set_trace()
    #     #
    #     # 1) build 1st layer
    #     #
    #     #      |
    #     #      P
    #     #      ||/p0        |/p0
    #     # --P==a*a==P-- = --l1--
    #     #   p1/||        p1/|
    #     #      P
    #     #      |
    #     #
    #     l1= torch.einsum('apuldr,asefgh->psuelfdgrh',a,a.conj()).contiguous().view(\
    #         a.size(1), a.size(1),a.size(2)**2,a.size(3)**2,a.size(4)**2,a.size(5)**2)
    #     l1= torch.einsum('psxywz,xu,yl,wd,zr->psuldr',l1, P, P, P, P).contiguous()
    #     #
    #     # 2) build 2nd layer
    #     #
    #     #   p0 
    #     #   |/        ||  
    #     # --l1-- => ==l2==
    #     #  /|         ||
    #     #   p1
    #     #   |/
    #     # --l1--
    #     #  /p0
    #     #
    #     l2= torch.einsum('psuldr,spxywz->uxlydwrz',l1,l1).contiguous().view(
    #         (D1**2,)*4)

    #     # prepare initial environment tensors
    #     #
    #     # C--, --T--
    #     # |      |
    #     #
    #     C_0= torch.einsum('psuldr,spulwz->dwrz',l1,l1).contiguous().view(
    #         (D1**2,)*2)
    #     T_0= torch.einsum('psuldr,spuywz->lydwrz',l1,l1).contiguous().view(
    #         (D1**2,)*3)

    #     #
    #     # 3) (optional) truncate auxiliary bonds
    #     if D2<D1**2:
    #         tmp_M= l2.view( (D1,D1,D1**2)*2 )
    #         tmp_M= torch.einsum('ijlijr->lr',tmp_M).contiguous()
    #         D, P= truncated_eig_sym(tmp_M, D2, keep_multiplets=True,\
    #             verbosity=cfg.ctm_args.verbosity_projectors)
    #         l2= torch.einsum('xywz,xu,yl,wd,zr->uldr',l2, P, P, P, P).contiguous()
    #         C_0= torch.einsum('wz,wd,zr',C_0,P,P).contiguous()
    #         T_0= torch.einsum('ywz,yl,wd,zr',T_0,P,P,P).contiguous()

    #     return l2, C_0, T_0

    def get_z_per_site(A, C, T):
        A_sl_scale= A.abs().max()
        if len(A.size())==5:
            # assume it is single-layer tensor with physica+ancilla fused
            auxD= A.size(4)
            A= torch.einsum('suldr,sefgh->uelfdgrh',A,A).contiguous()
            A= A.view([auxD**2]*4)
        elif len(A.size())==6:
            # assume it is single-layer tensor with physica+ancilla unfused
            if args.mode=="sl":
                auxD= A.size(5)
                A= torch.einsum('asuldr,asefgh->uelfdgrh',A,A).contiguous()
                A= A.view([auxD**2]*4)
                A_sl_scale= A_sl_scale**2
            elif args.mode=="dl":
                A= torch.einsum('ssuldr->uldr',A).contiguous()

        # C--T--C            / C--C
        # |  |  |   C--C    /  |  |   C--T--C
        # T--A--T * |  |   /   T--T * |  |  |
        # |  |  |   C--C  /    |  |   C--T--C
        # C--T--C        /     C--C

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
        # rdm = torch.tensordot(rdm,A,([1,3],[1,2]))
        # rdm = torch.tensordot(rdm,A/(A_sl_scale**2),([1,3],[1,2]))
        rdm = torch.tensordot(rdm,A/A_sl_scale,([1,3],[1,2]))
        rdm = torch.tensordot(T,rdm,([1,2],[0,2]))
        rdm = torch.tensordot(rdm,CTC,([0,1,2],[2,0,1]))

        # closed CTCCTC
        #   C--0 2--C
        # A |       |
        # | T--1 1--T |
        #   |       | V
        #   C--2 0--C
        CTC= torch.tensordot(CTC,CTC,([0,1,2],[2,1,0]))
        
        log.info(f"get_z_per_site rdm {rdm.item()} CTC {CTC.item()} C4 {C4.item()}")

        # z_per_site= (rdm/CTC)*(CTC/C4)
        logz_per_site= torch.log(A_sl_scale) + torch.log(rdm) + torch.log(C4)\
            - 2*torch.log(CTC)
        return logz_per_site
        # return z_per_site

    # def approx_renyi2(state,env,D1,D2):
    #     # get intial renyi2 on-site tensor and env tensors
    #     # state.site() rank-6 tensor [apuldr]
    #     a= state.site()
    #     # l2 - rank-4 tensor [uldr]
    #     l2, C0, T0= renyi2_site(a,D1,D2)
    #     state_r2= IPEPS_C4V(l2)
    #     env_r2= ENV_C4V(args.chi, state_r2)
    #     init_env(None, env_r2, (C0,T0))

    #     # run CTM
    #     env_r2, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_r2, env_r2, \
    #         conv_check=ctmrg_conv_f2)
    #     print(history)

    #     # renyi-2 = -log( (Tr \rho^2)/Z^2 ) = -log ( (Tr \rho^2) / (Tr \rho)^2 )
    #     #

    #     rho2_per_site= get_z_per_site( state_r2.site(), env_r2.get_C(), env_r2.get_T() )
        
    #     # double-layer
    #     l1= torch.einsum('apuldr,apxywz->uxlydwrz',a,a.conj())\
    #         .contiguous().view(a.size(2)**2,a.size(3)**2,a.size(4)**2,a.size(5)**2)
    #     rho1_per_site= get_z_per_site( l1, env.get_C(), env.get_T() )

    #     renyi2= -torch.log(rho2_per_site/(rho1_per_site**2))
    #     return renyi2


    # 0) fuse ancilla+physical index and create regular c4v ipeps with rank-5 on-site
    #    tensor
    state_fused= state.to_fused_ipeps_c4v()
    # state_fused= state.to_nophys_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)

    # e0 = energy_f(state, ctm_env, force_cpu=True)
    # S0 = approx_S(state, ctm_env)
    # r2_0 = approx_renyi2(state, ctm_env, args.l2d, args.bond_dim**2)
    # loss0 = e0 - 1./args.beta * S0
    # loss0 = e0 - 1./args.beta * r2_0
    S0= r2_0= 0
    e0= energy_f(state, ctm_env, args.mode, force_cpu=True)
    z0= get_z_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    loss0= -torch.log(z0) if args.logz else -z0 
    obs_values, obs_labels = eval_obs_f(state, ctm_env, args.mode, force_cpu=True)
    print("\n\n",end="")
    print(", ".join(["beta","epoch","loss","e0","z0","S0","r2_0"]+obs_labels+["norm(A)"]))
    print(", ".join([f"{args.beta}",f"{-1}",f"{loss0}",f"{e0}",f"{z0}",f"{S0}",f"{r2_0}"]\
        +[f"{v}" for v in obs_values]+[f"{state.site().norm()}"]))
    if args.logz:
        state.seed_site= state.seed_site *0.6 #torch.pow(z0,-0.25)

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build on-site tensors
        # new_iso= state.symmetrize_isometries()
        # state= IPEPS_C4V_THERMAL_TTN(state.seed_site, new_iso)
        state.update_()
        state_fused= state.to_fused_ipeps_c4v()
        # state_fused= s_state.to_nophys_ipeps_c4v()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_fused, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run_dl(state_fused, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        # e0 = energy_f(state, ctm_env_out, force_cpu=True)
        # S0 = approx_S(state, ctm_env_out)
        # loss0 = e0 - 1./args.beta * S0

        # r2_0 = approx_renyi2(state, ctm_env_out, args.l2d, args.bond_dim**2)
        # loss= torch.max(e0,e1) - 1./args.beta * r2_0
        # loss= torch.max(loss0,loss1)
        # z= get_z_per_site(state.site(), ctm_env_out.get_C(), ctm_env_out.get_T())
        logz= get_z_per_site(state.site(), ctm_env_out.get_C(), ctm_env_out.get_T())
        # loss= -torch.log(z) if args.logz else -z
        loss= -logz

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
        e0 = energy_f(state, ctm_env, args.mode, force_cpu=True)
        z0= get_z_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
        # S0 = approx_S( state, ctm_env )
        # r2_0 = approx_renyi2(state, ctm_env, args.l2d, args.bond_dim**2)
        S0=r2_0= 0
        obs_values, obs_labels = eval_obs_f(state,ctm_env,args.mode,force_cpu=True)
        print(f"{args.beta}, "+", ".join([f"{epoch}",f"{loss}",f"{e0}", f"{z0}", f"{S0}", f"{r2_0}"]\
            +[f"{v}" for v in obs_values]+[f"{state.site().norm()}"]))

        if (not opt_context["line_search"]) and args.top_freq>0 and epoch%args.top_freq==0:
            coord_dir_pairs=[((0,0), (1,0))]
            for c,d in coord_dir_pairs:
                # transfer operator spectrum
                print(f"TOP spectrum(T)[{c},{d}] ",end="")
                state_fused= state.to_fused_ipeps_c4v()
                l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_fused, ctm_env)
                print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v_thermal_ttn(outputstatefile)
    state_fused= state.to_fused_ipeps_c4v()
    ctm_env = ENV_C4V(args.chi, state_fused)
    init_env(state_fused, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run_dl(state_fused, ctm_env, conv_check=ctmrg_conv_f)
    e0 = energy_f(state,ctm_env,args.mode,force_cpu=True)
    z0= get_z_per_site(state.site(), ctm_env.get_C(), ctm_env.get_T())
    obs_values, obs_labels = eval_obs_f(state, ctm_env, args.mode, force_cpu=True)
    loss0, S0, r2_0= -torch.log(z0) if args.logz else -z0, 0, 0
    print("\n\n",end="")
    print(", ".join(["beta","epoch","loss","e0","z0","S0","r2_0"]+obs_labels))
    print(", ".join([f"{args.beta}","FINAL",f"{loss0}",f"{e0}",f"{z0}",f"{S0}",f"{r2_0}"]\
        +[f"{v}" for v in obs_values]))

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
