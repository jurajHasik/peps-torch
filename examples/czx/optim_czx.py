
import context
import torch
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg
from ctm.generic import transferops
from optim.ad_optim_lbfgs_mod import optimize_state
from models import czx
import logging
log = logging.getLogger(__name__)

def A_czx():
    # GS is
    #
    # |\psi> = \prod_{p\in plaquettes} (|0000> + |1111>)_p
    #
    # The state on plaquette |p> = \sum_{\{s\}} c_{\{s\} |\{s\}> is thus c_0000 = c_1111 = 1 and 0 otherwise.
    #
    # We can try and write the c_{\{s\} = Tr_aux( A A A A ) i.e. express the coefficients as 4-site periodic MPS generated by single tensor A
    # Let's try:
    #              A^0 = 1 0   and   A^1 = 0 0
    #                    0 0               0 1
    #
    # The PEPS tensor then can be generated by "grouping" four such tensors A
    #
    #      4  5
    #      |  |
    #  11--a0 a1--6  fusing virtual indices in ordered pairs as (4 5) (10 11) (8 9) (7 6)
    #  10--a3 a2--7
    #      |  |
    #      9  8
    #

    a=torch.zeros((2,2,2), dtype=torch.float64)
    a[0,0,0]=1.; a[1,1,1]=1.;
    #
    # This is "interleaved" form of input of tensor network with
    # individual operands (tensors) are followed by list of integers that label indices(/modes/legs) of
    # tensors. Optionally, last list of integers labels indices of resulting tensor
    # As usual, the repeated labels are contracted over.
    A_unfused = torch.einsum(a,[0,11,4],a,[1,5,6],a,[2,7,8],a,[3,9,10],[0,1,2,3, 4,5, 10,11, 8,9, 6,7])\
        .reshape([2,]*4+[2,]*8)

    #     4  5              4 5
    #     |  |              | |
    # 7--a0 a1--10      6--a0 a1--10   reorder indices, such that reshape preserves correct contractions when applying PBC's,
    # 6--a3 a2--11  ->  7--a3 a2--11   i.e. (4 5)(8 9)
    #     |  |              | |
    #     9  8              8 9
    Aczx= A_unfused.permute(0,1,2,3, 4,5, 7,6, 9,8, 10,11).reshape([2**4,]+[2**2,]*4)
    return Aczx

def A_zxz():    
    # convention for ordering of on-site spins
    #
    # 0 1
    # 3 2

    # Tensor for the +ZXZ ground state in MPS form - Eq. (26)
    M = torch.zeros([2, 2, 2], dtype=torch.float64)
    M[0] = torch.tensor([[0, 0], [1/np.sqrt(2), 1/np.sqrt(2)]])
    M[1] = torch.tensor([[1/np.sqrt(2), -1/np.sqrt(2)], [0, 0]])

    # PEPS tensor for the +ZXZ ground state, with the first index
    # corresponding to the physical index, with basis {|sa, sb> (4-dimensional), <rest> (12-dimensional)}
    T = torch.zeros([4*4, 4, 4, 4, 4], dtype=torch.float64)
    T[:4,:2,:2,:2,:2]= torch.einsum(M,[10,1,3],M,[20,0,2],[10,20,0,1,2,3]).reshape(4,2,2,2,2) #up right down left
    #! Note the ordering: for the up/down indices we have M[10,:,:] --> s_b
    #! and for the left/right indices we have M[i,:,:] --> s_a

    #Change of basis to the CZX original basis
    # |1>, |2>, |4>, |3>, ...
    def bin_to_state(s):
        v = torch.zeros(16, dtype=torch.float64)
        v[int(s, 2)] = 1.
        return v

    sasb = {1: (1/np.sqrt(2))*sum(bin_to_state(state) for state in ["0111", "1000"]),
            2: (1/np.sqrt(2))*sum(bin_to_state(state) for state in ["1011", "0100"]),
            4: (1/np.sqrt(2))*sum(bin_to_state(state) for state in ["1101", "0010"]),
            3: (1/np.sqrt(2))*sum(bin_to_state(state) for state in ["1110", "0001"])}

    P_sasb = torch.zeros([16, 16], dtype=torch.float64) # columns made from [<s_1_s2_s3_s4|1>,<...|2>,<...|4>,<...|3>,...]
    for s in sasb.keys():
        P_sasb[:, s-1] = sasb[s]


    # Onsite tensor for the ground state of the cZXZ model in the computational basis of the original CZX model.
    # as A = P_sasb T
    Azxz = torch.einsum(P_sasb, [0, 1], T, [1, 2, 3, 4, 5], [0, 2, 3, 4, 5]) # reshape to fit c
    Azxz = Azxz.permute(0, 1, 4, 3, 2) # u,r,d,l -> u,l,d,r
    return Azxz

# Preliminary:
# Here we specify configuration of the simulation (generally this is in the form of command line arguments)
# See for detailed description of options see https://jurajhasik.github.io/peps-torch/config.html
#
# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--gczx", type=float, default=1, help="CZX coupling")
parser.add_argument("--gzxz", type=float, default=0., help="ZXZ coupling")
parser.add_argument("--V", type=float, default=0., help="ZXZ projection")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()

    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(cfg.main_args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(cfg.global_args.device)

    model= czx.CZX(g_czx=args.gczx, g_zxz=args.gzxz, V=args.gzxz * args.V)

    # We will track convergence of CTM using spectra of CTM's corners
    # Lets modify generic conv. check to print convergence info
    #
    # It might be desirable to supress printing here
    #
    def custom_ctmrg_conv_specC(*args,**kw_args):
        verbosity= kw_args.pop('verbosity',0)
        converged, history= ctmrg_conv_specC(*args,**kw_args)
        # Optionally
        if converged and verbosity>0:
            print(f"CTM-CONV {len(history['conv_crit'])} {history['conv_crit'][-1]}")
        return converged, history

    # Loss function, which, given an iPEPS, first performs CTMRG until convergence
    # and then evaluates the energy per plaquette
    def loss_fn(state, ctm_env_in, opt_context, f_conv_ctm_opt, verbosity=0):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg.run(state, ctm_env_in, \
            conv_check=f_conv_ctm_opt, ctm_args=ctm_args)
        ctm_env_out= ctm_env_in

        # 2) evaluate loss with the converged environment
        loss= model.energy_per_site(state,ctm_env,verbosity=verbosity)

        # Add normalization of on-site tensor(s)
        #
        norm_penalty= sum( [ (1-state.site(c).norm())**2 for c in state.sites ] )
        loss = loss + 0*norm_penalty
        

        return (loss, ctm_env_out, *ctm_log)


    # We define a function, which inspects the state throughout the course of optimization
    # reporting i.e. energy and other observables of interest. We don't want to differentiate this function.
    #
    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
            norm_penalty= sum( [ (1-state.site(c).norm())**2 for c in state.sites ] )

            # Optionally
            # print(ctm_env.get_spectra())
            # transfer operator spectrum
            site_dir_list=[((0,0), (1,0)), ((0,0), (0,1))]
            for sdp in site_dir_list:
                print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
                l= transferops.get_Top_spec(5, *sdp, state, ctm_env)
                for i in range(l.size()[0]):
                    print(f"{i} {l[i,0]} {l[i,1]}")

            obs_values, obs_labels = [], []  #eval_obs_fn(state,ctm_env)
            print("OPT "+ ", ".join([f"{epoch}",f"{loss}",f"{norm_penalty.item()}",f"{loss-norm_penalty.item()}"]+[f"{v}" for v in obs_values]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

    # create CTMRG environment with environment bond dimension \chi (which governs the precision) and initialize it
    state= IPEPS(sites={(0,0): model.g_czx * A_czx() + model.g_zxz * A_zxz()})

    ctm_env= ENV(args.chi, state)
    init_env(state, ctm_env)
    def f_conv_ctm_opt(*args,**kwargs):
        return custom_ctmrg_conv_specC(*args,verbosity=1,**kwargs)
    loc_loss_fn= lambda state,env,opt_context : loss_fn(state,env,opt_context, f_conv_ctm_opt, verbosity=2)

    # converge initial environment
    # ctm_env, *ctm_log= ctmrg.run(state, ctm_env, conv_check=custom_ctmrg_conv_specC)

    # We enter optimization
    print(f"iter loss")
    optimize_state(state, ctm_env, loc_loss_fn, obs_fn=obs_fn)

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()