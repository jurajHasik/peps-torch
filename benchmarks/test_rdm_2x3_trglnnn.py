import context
import pytest
import torch
import config as cfg
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV, init_random
from ctm.generic import rdm_mc

test_dims=[(3,27), (3,54), (4,32), (4,64)]

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("open_inds",[[2,3]])
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_rdm2x3_loop_oe(dims, open_inds, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	print(f"{dims} {unroll}")
	benchmark.pedantic(rdm_mc.rdm2x3_loop_oe, args=((0,0), state, env, open_inds, unroll),\
		iterations=1, rounds=2, warmup_rounds=1)

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("open_inds",[[2,3]])
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_rdm2x3_loop_oe_semimanual(dims, open_inds, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	print(f"{dims} {unroll}")
	benchmark.pedantic(rdm_mc.rdm2x3_loop_oe_semimanual, args=((0,0), state, env, open_inds, unroll),\
		iterations=1, rounds=2, warmup_rounds=1)