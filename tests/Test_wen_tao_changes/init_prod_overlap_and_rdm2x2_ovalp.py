from ctm.generic.env import *
from ipeps.ipeps import *
from ctm.generic import rdm_overlap
import torch
D1=4
D2=3
d=2
chi=30
dtype=torch.double
T1=torch.rand(d,D1,D1,D1,D1)
T2=torch.rand(d,D2,D2,D2,D2)

sites_bra = {(0, 0): T1, (1, 0): T1,
                 (0, 1): T1, (1, 1): T1}
sites_ket = {(0, 0): T2, (0, 1): T2,
                 (1, 0): T2, (1, 1): T2}

# sites_bra = {(0, 0): T1,
#                  (0, 1): T1}
# sites_ket = {(0, 0): T2,
#                  (0, 1): T2}
state_bra = IPEPS(sites_bra, lX=1, lY=2)
state_ket = IPEPS(sites_ket, lX=1, lY=2)
ctm_env = ENV(chi, state_bra)
init_prod_overlap(state_bra, state_ket, ctm_env)

rdm_11=rdm_overlap.rdm2x2_overlap((0,0),state_bra,state_ket,ctm_env)
rdm_12=rdm_overlap.rdm2x2_overlap((1,0),state_bra,state_ket,ctm_env)
rdm_21 = rdm_overlap.rdm2x2_overlap((0, 1), state_bra, state_ket, ctm_env)
rdm_22 = rdm_overlap.rdm2x2_overlap((1, 1), state_bra, state_ket, ctm_env)