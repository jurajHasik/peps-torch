import torch
from args import *
import env
from env import ENV
from ipeps import *
import ctmrg
import rdm
from env import *
from models import akltS2, coupledLadders, hb
from ad_optim import optimize_state

torch.set_num_threads(args.omp_cores)

if __name__=='__main__':

    torch.set_printoptions(precision=7)

    # model = hb.HB()
    model = akltS2.AKLTS2()
    
    gs_state = read_ipeps("test-input/AKLT-S2_1x1.in")
    gs_ctm_env = ENV(args.chi, gs_state)
    init_env(gs_state, gs_ctm_env)
    ctmrg.run(gs_state, gs_ctm_env, ctm_args=CTMARGS(), global_args=GLOBALARGS())
    gs_energy = model.energy_1x1c4v(gs_state, gs_ctm_env)
    print("(exact GS) E(2x1+1x2): "+str(gs_energy))
    gs_obs= model.eval_obs(gs_state,gs_ctm_env)
    for label,val in gs_obs.items():
        print(f"{label} {val:.3E}", end=" ")
    print("")
    gs_rdm1x1 = rdm.rdm1x1((0,0),gs_state,gs_ctm_env)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
        
    def lattice_to_site(coord):
        vx = (coord[0] + abs(coord[0]) * 2) % 2
        vy = abs(coord[1])
        return ((vx + vy) % 2, 0)

    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site, peps_args=PEPSARGS(), global_args=GLOBALARGS())
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
    
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)
        B = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim), dtype=torch.float64)

        sites = {(0,0): A, (1,0): B}

        for k in sites.keys():
            sites[k] = sites[k]/torch.max(torch.abs(sites[k]))
        state = IPEPS(sites, lattice_to_site)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    ctm_env_init = ENV(args.chi, state)
    init_env(state, ctm_env_init)

    def ctmrg_conv_energy(state, env, history, ctm_args = CTMARGS()):
        with torch.no_grad():
            e_curr = model.energy_2x1_1x2(state, env)
            history.append(e_curr.item())

            if len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol:
                return True
        return False

    def loss_fn(state, ctm_env_init, ctm_args = CTMARGS(), global_args = GLOBALARGS()):
        # 0) pre-process state: normalize on-site tensors by largest elements
        # for coord,site in state.sites.items():
        #     site = site/torch.max(torch.abs(site))

        # 1) compute environment by CTMRG
        ctm_env = ctmrg.run(state, ctm_env_init, conv_check=ctmrg_conv_energy, ctm_args=ctm_args, global_args=global_args)
        loss = model.energy_2x1_1x2(state, ctm_env)
        
        # compute,print observables
        obs= model.eval_obs(state,ctm_env)
        for label,val in obs.items():
            print(f"{label} {val:.3E}", end=" ")
        print("")

        return loss, ctm_env

    # optimize
    optimize_state(state, ctm_env_init, loss_fn, verbosity=1)

    ctm_env = ENV(args.chi, state)
    init_env(state, ctm_env)
    ctm_env = ctmrg.run(state, ctm_env)
    opt_energy = model.energy_2x1_1x2(state,ctm_env)
    
    var_rdm1x1 = rdm.rdm1x1((0,0),state,ctm_env)
    print(var_rdm1x1-gs_rdm1x1)
    print(gs_rdm1x1)
    print(var_rdm1x1)

    print(f"(opt state) E(2x1+1x2): {opt_energy}")
    