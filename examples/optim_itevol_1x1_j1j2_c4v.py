import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import *
import groups.su2 as su2
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import *
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
from optim.itevol_optim_bfgs import itevol_plaquette_step
# from optim.itevol_optim_sgd import itevol_plaquette_step
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("-top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("-top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state= extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
        state.sites[(0,0)]= state.site()/state.site().norm()
    elif args.opt_resume is not None:
        state= IPEPS_C4V(torch.tensor(0.))
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        # A= make_c4v_symm(A)
        # A= A/torch.max(torch.abs(A))
        A= A/A.norm()
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
        #     if not history:
        #         history=[]
        #     e_curr = energy_f(state, env)
        #     history.append(e_curr.item())
        #     if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
        #         or len(history) >= ctm_args.ctm_max_iter:
        #         log.info({"history_length": len(history), "history": history})
        #         return True, history
        # return False, history
            if not history:
                history=dict({"log": []})
            rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
            dist= float('inf')
            if len(history["log"]) > 0:
                dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
            history["rdm"]=rdm2x1
            history["log"].append(dist)
            if dist<ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
                log.info({"history_length": len(history['log']), "history": history['log']})
                return True, history
        return False, history

    state_sym= to_ipeps_c4v(state)
    ctm_env= ENV_C4V(args.chi, state_sym)
    init_env(state_sym, ctm_env)
    
    ctm_env, *ctm_log = ctmrg_c4v.run(state_sym, ctm_env, conv_check=ctmrg_conv_energy)
    loss= energy_f(state_sym, ctm_env)
    obs_values, obs_labels= model.eval_obs(state_sym,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def _to_json(l):
        re=[l[i,0].item() for i in range(l.size()[0])]
        im=[l[i,1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    def obs_fn(state, ctm_env, epoch):
        state_sym= to_ipeps_c4v(state, normalize=True)
        loss= energy_f(state_sym, ctm_env)
        obs_values, obs_labels = model.eval_obs(state_sym, ctm_env)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]+\
            [f"{torch.max(torch.abs(state.site((0,0))))}"]))

        with torch.no_grad():
            if args.top_freq>0 and epoch%args.top_freq==0:
                coord_dir_pairs=[((0,0), (1,0))]
                for c,d in coord_dir_pairs:
                    # transfer operator spectrum
                    print(f"TOP spectrum(T)[{c},{d}] ",end="")
                    l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
                    print("TOP "+json.dumps(_to_json(l)))

    def post_proc(state, opt_context):
        symm, max_err= verify_c4v_symm_A1(state.site())
        # print(f"post_proc {symm} {max_err}")
        if not symm:
            # force symmetrization outside of autograd
            with torch.no_grad():
                symm_site= make_c4v_symm(state.site())
                # we **cannot** simply normalize the on-site tensors, as the LBFGS
                # takes into account the scale
                # symm_site= symm_site/torch.max(torch.abs(symm_site))
                state.sites[(0,0)].copy_(symm_site)

    # build 4-site Trotter gate
    s2 = su2.SU2(model.phys_dim, dtype=model.dtype, device=model.device)
    id2= torch.eye(model.phys_dim**2,dtype=model.dtype,device=model.device)
    id2= id2.view(tuple([model.phys_dim]*4)).contiguous()
    h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',model.h2,id2)
    hp= 0.5*model.j1*(h2x2_SS + h2x2_SS.permute(0,2,1,3,4,6,5,7) \
            + h2x2_SS.permute(2,3,0,1,6,7,4,5) + h2x2_SS.permute(2,0,3,1,6,4,7,5)) \
            + model.j2*(h2x2_SS.permute(0,2,3,1,4,6,7,5) + h2x2_SS.permute(2,0,1,3,6,4,5,7))
    hp= hp.contiguous()

    tau=args.itevol_step
    tg= hp.view(model.phys_dim**4,model.phys_dim**4)
    tg_d, tg_u= torch.symeig(tg, eigenvectors=True)
    tg= tg_u @ torch.diag(torch.exp(-tau * tg_d)) @ tg_u.t()
    tg2= tg @ tg
    tg= tg.view(tuple([model.phys_dim]*8))
    tg2= tg2.view(tuple([model.phys_dim]*8))

    rot_op= s2.BP_rot()
    hp_rot= torch.einsum('ijklabcd,im,ln,ay,dv->mjknybcv',hp,rot_op,rot_op,rot_op,rot_op)
    tg= torch.einsum('ijklabcd,im,ln,ay,dv->mjknybcv',tg,rot_op,rot_op,rot_op,rot_op)
    tg2= torch.einsum('ijklabcd,im,ln,ay,dv->mjknybcv',tg2,rot_op,rot_op,rot_op,rot_op)

    # TODO optimize
    for epoch in range(args.opt_max_iter):
        state_sym= to_ipeps_c4v(state, normalize=True)

        # possibly re-initialize the environment
        if cfg.opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env)

        # 1) compute environment by CTMRG
        ctm_env, *ctm_log= ctmrg_c4v.run(state_sym, ctm_env, 
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)

        # 2a) prepare imag-time evol step - build mixed reduced density matrix
        #    without on-site tensors only at single vertex
        #   _______________________
        #  |a_a_a___________a______|   <=> C--T--T--C              /
        #   0 1 2       |  |  |  |         |  |  |  |       A = --a--
        #  s0 s1 s2     6  7  8  9         T--A--A--T            /|s0
        #                                  |  |  |  |           
        #  s0' s1' s2' 10 11 12 13         T--A-- --T             |/s0'
        #   3 4 5_______|  |  |  |_        |  |  |  |           --a-- 
        #  |a_a_a__________a_______|       C--T--T--C  where     /
        rdm2x2_x1= rdm2x2_x1site(state_sym, ctm_env)
        # test_rdm2x2_x1site(rdm2x2_x1, state_sym.site().size()[0], state_sym.site().size()[1])
        spd_2x2_x1, cnum= rdm2x2_x1_symm_posdef_normalize(rdm2x2_x1, state_sym.site().size()[0], \
            state_sym.site().size()[1],pos_def=False)
        # print(f"cnum {cnum}")
        # test_rdm2x2_x1_vs_rdm2x2(hp_rot, rdm2x2_x1, spd_2x2_x1, state_sym, ctm_env)

        # 2b) insert gate to create effective single-site reduced density matrix
        #   _____________________
        #  |_____________________|  
        #   0 1 2       6 7 8 9->2 3 4 5
        #   0_1_2__3->0
        #  |__4site__|
        #   4 5 6  7->1
        #   3 4 5______10_11_12_13->6 7 8 9
        #  |_____________________|
        #
        # <state_0|UU|state_0>
        norm0UU_x1= torch.tensordot(tg2,spd_2x2_x1,([0,1,2,4,5,6],[0,1,2,3,4,5]))
        norm0UU_x1= torch.einsum('ijmnopqrst,imnop->jqrst',norm0UU_x1,state_sym.site())
        norm0UU_x1= torch.einsum('jqrst,jqrst',norm0UU_x1,state_sym.site())

        rdm1U0_x1= torch.tensordot(tg,spd_2x2_x1,([0,1,2,4,5,6],[0,1,2,3,4,5]))
        # <state_1_x1|U|state_0>
        rdm1U0_x1_ket= torch.einsum('ijmnopqrst,jqrst->imnop',rdm1U0_x1,state_sym.site())
        # <state0|U|state_1_x1>
        rdm0U1_x1_bra= torch.einsum('ijmnopqrst,imnop->jqrst',rdm1U0_x1,state_sym.site())

        # <state_1_x1|state_1_x1>
        # rdm1_x1= torch.einsum('ijkijkmnopqrst->mnopqrst',spd_2x2_x1)
        rdm1_x1= torch.einsum('ijkijkmnopqrst->mnopqrst',rdm2x2_x1)
        # test symmetry and positive definitness
        rdm1_x1= rdm1_x1.view([args.bond_dim**4]*2)
        rdm1_x1_symm= 0.5*(rdm1_x1+rdm1_x1.t())
        rdm1_x1_asymm= 0.5*(rdm1_x1-rdm1_x1.t())
        print(f"rdm1_x1_symm {rdm1_x1_symm.norm()} rdm1_x1_asymm {rdm1_x1_asymm.norm()}")
        D, U= torch.symeig(rdm1_x1_symm)
        D_orig, U_orig= torch.eig(rdm1_x1)
        D_orig_re, perm= torch.sort(D_orig[:,0],descending=False)
        # print first five negative evs:
        print(f"rdm1_x1 D_symm    low {D[0:4].tolist()}")
        print(f"rdm1_x1 D_orig_re low {D_orig_re[0:4].tolist()}")
        # print first five positive evs:
        print(f"rdm1_x1 D_symm    top {D[-5:-1].tolist()}")
        print(f"rdm1_x1 D_orig_re top {D_orig_re[-5:-1].tolist()}")
        rdm1_x1= rdm1_x1.view([args.bond_dim]*8)

        #
        ardm1x1= aux_rdm1x1(state_sym, ctm_env)
        ardm1x1= ardm1x1.view([args.bond_dim**4]*2)
        ardm1x1_symm= 0.5*(ardm1x1+ardm1x1.t())
        ardm1x1_asymm= 0.5*(ardm1x1-ardm1x1.t())
        print(f"ardm1x1_symm {ardm1x1_symm.norm()} ardm1x1_asymm {ardm1x1_asymm.norm()}")
        D, U= torch.symeig(ardm1x1_symm,eigenvectors=True)
        D_orig, U_orig= torch.eig(ardm1x1)
        D_orig_re, perm= torch.sort(D_orig[:,0],descending=False)
        # print first five negative evs:
        print(f"ardm1x1 D_symm    low {D[0:4].tolist()}")
        print(f"ardm1x1 D_orig_re low {D_orig_re[0:4].tolist()}")
        # print first five positive evs:
        print(f"ardm1x1 D_symm    top {D[-5:-1].tolist()}")
        print(f"ardm1x1 D_orig_re top {D_orig_re[-5:-1].tolist()}")
        ardm1x1= ardm1x1.view([args.bond_dim]*8)
        # analyze C4v symmetry of eigenvectors
        for i in range(-5,0,1):
            evec= U[:,i]
            # normalized ? 
            print(f"U[:,{i}] norm {evec.norm()}")
            evec= evec[None,:].view([1]+[args.bond_dim]*4)
            evec_sym= make_c4v_symm_A1(evec) 
            print(f"make_c4v_symm_A1(U[-1,:]) norm {evec_sym.norm()}")


        state_trial= to_ipeps_c4v(state_sym, normalize=True)
        def loss_fn(state, opt_context):
            state_sym= to_ipeps_c4v(state, normalize=False)
            # state_sym= state

            # 1) compute norm <state_1|state_1>
            norm1_x1= torch.einsum('mnopqrst,iqrst->mnopi',rdm1_x1,state_sym.site())
            # exact grad
            g= norm1_x1.permute(4,0,1,2,3).contiguous() - rdm1U0_x1_ket
            norm1_x1= torch.einsum('mnopi,imnop',norm1_x1,state_sym.site())
            
            # 2) overlaps <state_1|U|state_0>, <state0|U|state_1>
            overlap1U0_x1= torch.einsum('imnop,imnop',rdm1U0_x1_ket,state_sym.site())
            overlap0U1_x1= torch.einsum('jqrst,jqrst',rdm0U1_x1_bra,state_sym.site())
            
            # loss= 1. - (overlap1U0_x1+overlap0U1_x1)*torch.rsqrt(norm1_x1*norm0UU_x1) + 1.
            loss= norm1_x1 - overlap1U0_x1 - overlap0U1_x1 + norm0UU_x1
            # loss= torch.einsum('imnop,imnop',g,g)
            # print(f"{dist} {loss} {norm1_x1}")

            return loss, g

        itevol_plaquette_step(state_trial, loss_fn)

        # compute observable
        # o= fidelity_rdm2x2(prdm, state_sym)
        # print(f"{n} {o} {o/n}")
        # symm, max_err= verify_c4v_symm_A1(state_sym.site())
        # print(f"post_proc {symm} {max_err}")
        state_trial= to_ipeps_c4v(state_trial, normalize=True)
        init_env(state_trial, ctm_env)

        # 3) compute environment of evolved state by CTMRG
        ctm_env, *ctm_log= ctmrg_c4v.run(state_trial, ctm_env, 
            conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)

        # 4) compute observables
        obs_fn(state_trial, ctm_env, epoch)

        # 5) check convergence
        state= to_ipeps_c4v(state_trial, normalize=True)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_energy)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()



class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3

    # basic tests
    def test_opt_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()
