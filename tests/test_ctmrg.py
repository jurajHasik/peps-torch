import context
import torch
import argparse
from args import *
import env
from env import ENV
import ipeps
import ctmrg
import rdm
import su2
from models import akltS2, coupledLadders, j1j2, hb

# additional model-dependent arguments (if any)
parser.add_argument("-tiling", default="DEFAULT", help="tiling of the lattice")
parser.add_argument("-verbosity", type=int, default=0, help="verbosity")
# J1-J2
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
# Coupled-Ladders
parser.add_argument("-alpha", type=float, default=0., help="inter-ladder coupling")

args = parser.parse_args()
torch.set_num_threads(args.omp_cores)

if __name__=='__main__':

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "DEFAULT":
        lattice_to_site=None
    elif args.tiling == "BIPARTITE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)
    elif args.tiling == "STRIPE":
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"DEFAULT, BIPARTITE, STRIPE")

    state = ipeps.read_ipeps(args.instate, vertexToSite=lattice_to_site)
    print(state)

    ctm_env= ENV(args.chi,state)

    # x) Initialize env tensors C,T
    env.init_env(state, ctm_env)
    if args.verbosity>0:
        env.print_env(ctm_env)

    ctm_env= ctmrg.run(state,ctm_env)
    
    model_hb= hb.HB()
    energy= model_hb.energy_2x2(state,ctm_env)
    print("E(2x2): "+str(energy))

    rdm2x2_00= rdm.rdm2x2((0,0), state, ctm_env, verbosity=1)
    sz_00= torch.einsum('ibcdabcd,ia',rdm2x2_00,model_hb.obs_ops["sz"])
    sp_00= torch.einsum('ibcdabcd,ia',rdm2x2_00,model_hb.obs_ops["sp"])
    sm_00= torch.einsum('ibcdabcd,ia',rdm2x2_00,model_hb.obs_ops["sm"])
    print(f"sz,sp,sm (0,0): {sz_00} {sp_00} {sm_00}")
    sz_11= torch.einsum('abciabcd,id',rdm2x2_00,model_hb.obs_ops["sz"])
    sp_11= torch.einsum('abciabcd,id',rdm2x2_00,model_hb.obs_ops["sp"])
    sm_11= torch.einsum('abciabcd,id',rdm2x2_00,model_hb.obs_ops["sm"])
    print(f"sz,sp,sm (1,1): {sz_11} {sp_11} {sm_11}")

    obs_ss=[]
    id2= torch.eye(4,dtype=model_hb.dtype,device=model_hb.device)
    id2= id2.view(2,2,2,2).contiguous()
    
    h2x2_nn= torch.einsum('ijab,klcd->ijklabcd',model_hb.h,id2)
    obs_ss.append(torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn))
    
    h2x2_nn_13= h2x2_nn.permute(2,0,3,1,6,4,7,5)
    obs_ss.append(torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn_13))

    h2x2_nn_02= h2x2_nn.permute(0,2,1,3,4,6,5,7)
    obs_ss.append(torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn_02))

    h2x2_nn_23= h2x2_nn.permute(2,3,0,1,6,7,4,5)
    obs_ss.append(torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn_23))

    print(f"(0,0)(1,0) (1,0)(1,1) (0,0)(0,1) (0,1)(1,1)")
    print(", ".join([f"{v}" for v in obs_ss]))

    # h2x2_nnn= torch.einsum('ijab,klcd->ikljacdb',self.h,id2)
    # h2x2_nnn= h2x2_nnn + h2x2_nnn.permute(1,0,3,2,5,4,7,6)

    # rdm2x2_00= rdm.rdm2x2((0,0),state,env)
    # rdm2x2_10= rdm.rdm2x2((1,0),state,env)
    # energy_nn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nn)
    # energy_nn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nn)
    # energy_nnn = torch.einsum('ijklabcd,ijklabcd',rdm2x2_00,h2x2_nnn)
    # energy_nnn += torch.einsum('ijklabcd,ijklabcd',rdm2x2_10,h2x2_nnn)

    obs_values, obs_labels = model_hb.eval_obs(state,ctm_env)
    print(", ".join(obs_labels))
    print(", ".join([f"{v}" for v in obs_values]))

    model = j1j2.J1J2(j2=0.5)
    energy = model.energy_2x2_2site(state,ctm_env)
    print("E(2x2_2site): "+str(energy))
    