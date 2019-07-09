import torch
import argparse
from args import *
from env import *
from ipeps import *
import ctmrg
from models import akltS2, coupledLadders, hb

# additional model-dependent arguments (if any)
args = parser.parse_args()
torch.set_num_threads(args.omp_cores)

if __name__=='__main__':

    state = read_ipeps(args.instate ,aux_seq=[1,0,3,2])
    state = extend_bond_dim(state, args.bond_dim)

    print(state)

    torch.set_printoptions(precision=7)
    ctm_env = ENV(args.chi,state)

    # x) Initialize env tensors C,T
    init_env(state, ctm_env)
    # if verbosity>0:
    print_env(ctm_env)

    ctm_env = ctmrg.run(state,ctm_env,ctm_args=CTMARGS(), global_args=GLOBALARGS())

    model = hb.HB()
    energy = model.energy_2x1_1x2(state,ctm_env)
    print("E(1x1c4v): "+str(energy))
    