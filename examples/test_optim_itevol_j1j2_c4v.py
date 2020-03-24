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
# from optim.itevol_optim_bfgs import itevol_plaquette_step
from optim.itevol_optim_sgd import itevol_plaquette_step
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
    # id4= torch.eye(model.phys_dim**4,dtype=model.dtype,device=model.device)
    # id4= id4.view(tuple([model.phys_dim]*8)).contiguous()
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

        # 
        rdm2x2aux= aux_rdm2x2(state_sym, ctm_env)
        # reshape into matrix
        rdm2x2aux= rdm2x2aux.view([args.bond_dim**8]*2)
        rdm2x2aux_symm= 0.5*(rdm2x2aux+rdm2x2aux.t())
        rdm2x2aux_asymm= 0.5*(rdm2x2aux-rdm2x2aux.t())
        print(f"rdm2x2aux_symm {rdm2x2aux_symm.norm()} rdm2x2aux_asymm {rdm2x2aux_asymm.norm()}")
        D, U= torch.symeig(rdm2x2aux_symm)
        print(D)

        # 2) prepare imag-time evol step
        prdm= partial_rdm2x2(state_sym, ctm_env)
        # normalization
        n= fidelity_rdm2x2(prdm, state_sym)
        assert n > 0, "norm given by 2x2 rdm is not positive"
        # apply 4-site operator operator
        # 
        #    0__1__2___3
        #    |____op___|
        #    4  5  6   7
        #  __2__5__8__11__       __0__1__2__3___
        # |_____prdm______| =>  |_____prdm______| 
        # 0 1 3 4 6 7 9  10     4 5 6 7 8 9 10 11
        # test
        energy0U= torch.tensordot(hp_rot,prdm,([4,5,6,7],[2,5,8,11]))
        energy0U= energy0U.permute(4,5,0,6,7,1,8,9,2,10,11,3).contiguous()
        energy0U= fidelity_rdm2x2(energy0U, state_sym)
        print(f"energyU0: {energy0U/n} {energy0U} {n}")
        # U|state_0>_ENV
        state0U= torch.tensordot(tg,prdm,([4,5,6,7],[2,5,8,11]))
        state0U= state0U.permute(4,5,0,6,7,1,8,9,2,10,11,3).contiguous()
        # <state_0|UU|state_0>_ENV
        n0UU= torch.tensordot(tg2,prdm,([4,5,6,7],[2,5,8,11]))
        n0UU= n0UU.permute(4,5,0,6,7,1,8,9,2,10,11,3).contiguous()
        n0UU= fidelity_rdm2x2(n0UU, state_sym)

        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        # A= make_c4v_symm(A)
        # A= A/torch.max(torch.abs(A))
        A= state_sym.site() + A*0.02
        A= A/A.norm()
        state_trial= IPEPS_C4V(A)
        state_trial= to_ipeps_c4v(state_trial, normalize=True)
        def loss_fn(state, opt_context):
            state_sym= to_ipeps_c4v(state, normalize=False)
            # state_sym= state
            # print(state_sym.site().norm())

            # a) compute overlap inserting 2x2 "bra" 
            overlap1U0= fidelity_rdm2x2(state0U, state_sym)
            n1= partial_rdm2x2(state_sym, ctm_env)
            n1= fidelity_rdm2x2(n1, state_sym)

            # loss= 1 - 2*overlap1U0/(torch.sqrt(n1)*torch.sqrt(n0UU)) + 1
            loss= n1 - 2*overlap1U0 + n0UU
            g= None
            print(f"{loss} {n1} {overlap1U0} {n0UU} {state_sym.site().norm()}")
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

        # 3) check convergence

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
