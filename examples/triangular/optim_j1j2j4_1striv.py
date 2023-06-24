import os
import context
import torch
import argparse
import copy
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
parser.add_argument("--tiling", default="1STRIV", help="tiling of the lattice", \
    choices=["1SITEQ","1STRIV","1SPG"])
parser.add_argument("--ctm_conv_crit", default="CSPEC", help="ctm convergence criterion", \
    choices=["CSPEC", "ENERGY"])
parser.add_argument("--test_env_sensitivity", action='store_true', help="compare loss with higher chi env")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--force_cpu", action='store_true', help="evaluate energy on cpu")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # initialize an ipeps and model
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    #
    # 2) select the "energy" function 
    if args.tiling in ["1STRIV","1SPG"]:
        model= spin_triangular.J1J2J4_1SITE(j1=args.j1, j2=args.j2, j4=args.j4)
        lattice_to_site=None
        read_state_f= read_ipeps_trgl_1s_ttphys_pg
        if args.tiling in ["1SPG"]:
            read_state_f= write_ipeps_trgl_1s_tbt_pg
    elif args.tiling in ["1SITEQ"]:
        model= spin_triangular.J1J2J4_1SITEQ(j1=args.j1, j2=args.j2, j4=args.j4, diag=args.diag,\
            q=None)
        # def energy_f(state,env,q= force_cpu=args.force_cpu):
        #     model.energy_per_site(state,env,force_cpu=force_cpu)
        read_state_f= lambda instate: read_ipeps_1s_q(instate,q=(args.q,args.q))
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITEQ, 1STRIV, 1SPG")
    energy_f=model.energy_per_site
    eval_obs_f= model.eval_obs

    if args.instate!=None:
        state = read_state_f(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = state.extend_bond_dim(args.bond_dim)
        if args.tiling in ["1STRIV","1SPG"]:
            state= state.add_noise(args.instate_noise)
        else:
            state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.tiling == "1STRIV":
            state= IPEPS_TRGL_1S_TRIVALENT()
        elif args.tiling in ["1SITEQ"]:
            state= IPEPS_1S_Q()
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        if args.tiling in ["1SITEQ"]:
            sites = {}
            A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5
            sites[(0,0)]= A/torch.max(torch.abs(A))
            state = IPEPS_1S_Q(sites, q=(1./args.q,1./args.q))
        elif args.tiling in ["1STRIV"]:
            ansatz_pgs= IPEPS_TRGL_1S_TTPHYS_PG.PG_A1
            t_aux= torch.rand([bond_dim]*3,\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            t_phys= torch.rand([bond_dim]*3 + [model.phys_dim],\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            state = IPEPS_TRGL_1S_TTPHYS_PG(t_aux=t_aux, t_phys=t_phys, pgs=ansatz_pgs,\
                pg_symmetrize=True, peps_args=cfg.peps_args, global_args=cfg.global_args)
        elif args.tiling in ["1SPG"]:
            ansatz_pgs= IPEPS_TRGL_1S_TBT_PG.PG_A1_A
            t_aux= torch.rand([bond_dim]*3,\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            t_phys= torch.rand([bond_dim]*2 + [model.phys_dim],\
                dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            state = IPEPS_TRGL_1S_TBT_PG(t_aux=t_aux, t_phys=t_phys, pgs=ansatz_pgs,\
                pg_symmetrize=True, peps_args=cfg.peps_args, global_args=cfg.global_args)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
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
            model= type(model)(j1=args.j1, j2=args.j2, j4=args.j4)

    print(state)

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr= energy_f(state, env, force_cpu=ctm_args.conv_check_cpu)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    # alternatively use ctmrg_conv_specC from ctm.generinc.env
    if args.ctm_conv_crit=="CSPEC":
        ctmrg_conv_f= ctmrg_conv_specC
    elif args.ctm_conv_crit=="ENERGY":
        ctmrg_conv_f= ctmrg_conv_energy

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        if not opt_context.get('line_search',False) and ctm_args.randomize_ctm_move_sequence:
            from itertools import permutations
            perms_ctm_moves= list(permutations([(0,-1),(-1,0),(0,1),(1,0)]))
            ctm_args.ctm_move_sequence= perms_ctm_moves[torch.randint(len(perms_ctm_moves),(1,))]

        # build state with normalized tensors
        if args.tiling in ["1STRIV", "1SPG"]:
            state_sym= to_PG_symmetric(state)
        else:
            state_sym= IPEPS_1S_Q({c: t/t.abs().max() for c,t in state.sites.items()}, q=state.q)

        # for c in state.sites.keys():
        #     with torch.no_grad():
        #         _scale= state.sites[c].abs().max()
        #     sites_n[c]= state.sites[c]/_scale
        # state_n= IPEPS(sites_n, vertexToSite=lattice_to_site, lX=state.lX, lY=state.lY)

        # possibly re-initialize the environment
        # with torch.no_grad():
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state_sym, ctm_env_in, \
             conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with the converged environment
        loss = energy_f(state_sym, ctm_env_out, force_cpu=args.force_cpu)

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
                re=[l[i,0].item() for i in range(l.size()[0])]
                im=[l[i,1].item() for i in range(l.size()[0])]
                return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state_sym= state
        if args.tiling in ["1SPG"]:
            # symmetrization and implicit rebuild of on-site tensors
            state_sym= to_PG_symmetric(state)

        if not opt_context.get("line_search",False):
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = eval_obs_f(state_sym,ctm_env)
            
            # test ENV sensitivity
            loc_ctm_args= copy.deepcopy(opt_context["ctm_args"])
            loc_ctm_args.ctm_max_iter= 1
            ctm_env_out1= ctm_env.extend(ctm_env.chi+10)
            ctm_env_out1, *ctm_log= ctmrg.run(state_sym, ctm_env_out1, \
                conv_check=ctmrg_conv_f, ctm_args=loc_ctm_args)
            loss1= energy_f(state_sym, ctm_env_out1, force_cpu=args.force_cpu)
            delta_loss= opt_context['loss_history']['loss'][-1]-opt_context['loss_history']['loss'][-2]\
                if len(opt_context['loss_history']['loss'])>1 else float('NaN')
            # if we are not linesearching, this can always happen
            # not "line_search" in opt_context.keys()
            if args.test_env_sensitivity:
                _flag_antivar= (loss1-loss)>0 and \
                    (loss1-loss)*opt_context["opt_args"].env_sens_scale>abs(delta_loss)
                opt_context["STATUS"]= "ENV_ANTIVAR" if _flag_antivar else "ENV_VAR"

            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
                + [f"{loss1-loss}"] ))
            
            log_info_string= f"env_sensitivity: {loss1-loss} loss_diff: "\
                +f"{delta_loss}"
            if args.tiling in ["1STRIV","1SPG"]:
                log_info_string += " Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.elem_tensors.items()])
            else:
                log_info_string += " Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()])
            log.info(log_info_string)

            # with torch.no_grad():
            #     if args.top_freq>0 and epoch%args.top_freq==0:
            #         coord_dir_pairs=[((0,0), (1,0)), ((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
            #         for c,d in coord_dir_pairs:
            #             # transfer operator spectrum
            #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
            #             l= transferops.get_Top_spec(args.top_n, c,d, state, ctm_env)
            #             print("TOP "+json.dumps(_to_json(l)))

    def post_proc(state, ctm_env, opt_context):
        with torch.no_grad():
            for c in state.elem_tensors.keys():
                _tmp= state.elem_tensors[c]/state.elem_tensors[c].abs().max()
                state.elem_tensors[c].copy_(_tmp)

    # optimize
    # state_g= IPEPS_WEIGHTED(state=state).gauge()
    # state= state_g.absorb_weights()
    # import pdb; pdb.set_trace()
    # state.normalize_()
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)#, post_proc=post_proc)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_state_f(outputstatefile)
    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_f)
    loss0= energy_f(state,ctm_env,force_cpu=args.force_cpu)
    obs_values, obs_labels = eval_obs_f(state,ctm_env)
    print("FINAL "+", ".join([f"{loss0}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCheckpoint_1SITEQ_Ansatze(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt-chck_trgl"
    ANSATZE= [("1SITEQ",)]

    def reset_couplings(self):
        args.j1= 1.0
        args.diag=0.9

    def setUp(self):
        self.reset_couplings()
        args.bond_dim=3
        args.chi=27
        args.seed=300
        args.opt_max_iter= 10
        args.instate_noise=0.1
        args.instate=self.DIR_PATH+"/../../test-input/D1_diag0.9_Vq_state.json"

    def test_checkpoint_ipess_ansatze(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose
        import numpy as np
        from ipeps.ipeps_1s_Q import write_ipeps_1s_q

        for ansatz in self.ANSATZE:
            with self.subTest(ansatz=ansatz):
                self.reset_couplings()
                args.opt_max_iter= 10
                args.tiling= ansatz[0]
                args.opt_resume= None
                args.out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}"
                # args.instate= args.out_prefix[len("RESULT_"):]+"_instate.json"

                # create randomized state
                # elif args.ansatz in ["IPESS"]:
                #     state= IPESS_KAGOME_GENERIC({'T_u': T_u, 'B_a': B_a, 'T_d': T_d,\
                #         'B_b': B_b, 'B_c': B_c})
                #     write_ipess_kagome_generic(state, args.instate)


                # i) run optimization and store the optimization data
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                # parse FINAL observables
                obs_opt_lines=[]
                final_obs=None
                OPT_OBS= OPT_OBS_DONE= False
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                        OPT_OBS_DONE= True
                        OPT_OBS=False
                    if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                        obs_opt_lines.append(l)
                    if "epoch, energy," in l and not OPT_OBS_DONE: 
                        OPT_OBS= True
                    if "FINAL" in l:
                        final_obs= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs
                assert len(obs_opt_lines)>0

                # compare the line of observables with lowest energy from optimization (i) 
                # and final observables evaluated from best state stored in *_state.json output file
                # drop the last column, not separated by comma
                best_e_line_index= np.argmin([ float(l.split(',')[1]) for l in obs_opt_lines ])
                opt_line_last= [complex(x) for x in obs_opt_lines[best_e_line_index].split(",")[1:-1]]
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                for val0,val1 in zip(opt_line_last, fobs_tokens):
                    assert isclose(val0,val1, rel_tol=self.tol, abs_tol=self.tol)

                # ii) run optimization for 3 steps
                # reset j1 which is otherwise set by main() if args.theta is used
                args.opt_max_iter= 3 
                self.reset_couplings()
                main()
        
                # iii) run optimization from checkpoint
                args.instate=None
                args.opt_resume= args.out_prefix+"_checkpoint.p"
                args.opt_max_iter= 7
                self.reset_couplings()
                with patch('sys.stdout', new = StringIO()) as tmp_out: 
                    main()
                tmp_out.seek(0)

                obs_opt_lines_chk=[]
                final_obs_chk=None
                OPT_OBS= OPT_OBS_DONE= False
                l= tmp_out.readline()
                while l:
                    print(l,end="")
                    if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": 
                        OPT_OBS_DONE= True
                        OPT_OBS=False
                    if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                        obs_opt_lines_chk.append(l)
                    if "checkpoint.loss" in l and not OPT_OBS_DONE: 
                        OPT_OBS= True
                    if "FINAL" in l:    
                        final_obs_chk= l.rstrip()
                        break
                    l= tmp_out.readline()
                assert final_obs_chk
                assert len(obs_opt_lines_chk)>0

                # compare initial observables from checkpointed optimization (iii) and the observables 
                # from original optimization (i) at one step after total number of steps done in (ii)
                opt_line_iii= [complex(x) for x in obs_opt_lines_chk[0].split(",")[1:]]
                # drop (last) normalization column
                opt_line_i= [complex(x) for x in obs_opt_lines[4].split(",")[1:-1]]
                fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                for val3,val1 in zip(opt_line_iii, opt_line_i):
                    assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

                # compare final observables from optimization (i) and the final observables 
                # from the checkpointed optimization (iii)
                fobs_tokens_1= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
                fobs_tokens_3= [complex(x) for x in final_obs_chk[len("FINAL"):].split(",")]
                for val3,val1 in zip(fobs_tokens_3, fobs_tokens_1):
                    assert isclose(val3,val1, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        # for ansatz in self.ANSATZE:
        #     out_prefix=self.OUT_PRFX+f"_{ansatz[0].replace(',','')}"
        #     instate= out_prefix[len("RESULT_"):]+"_instate.json"
        #     for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",\
        #         out_prefix+".log",instate]:
        #         if os.path.isfile(f): os.remove(f)