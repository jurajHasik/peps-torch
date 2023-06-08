import context
import pytest
import torch
import functools
from time import perf_counter
import config as cfg
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV, init_random
from ctm.generic import rdm_mc

import logging
logging.basicConfig(filename=f"{__file__}.log", filemode='w', level=logging.INFO)

test_dims=[(3,27), (3,54), (4,32), (4,64)]

def optional_cuda_measure(tag):
	def _inner_optional_cuda_measure(f):
		@functools.wraps(f)
		def _wrap(*args,**kwargs):
			if not cfg.global_args.device=='cpu': torch.cuda.synchronize()
			t0= perf_counter()
			res= f(*args,**kwargs)
			if not cfg.global_args.device=='cpu': torch.cuda.synchronize()
			t1= perf_counter()
			logging.info(f"{tag} {t1-t0} [s]")
			return res
		return _wrap
	return _inner_optional_cuda_measure

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("open_inds",[[1,2,3,4]])
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_rdm2x3_loop_oe(dims, open_inds, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	@optional_cuda_measure(f"rdm2x3_loop_oe{dims} {open_inds} {unroll}")
	def test_f():
		rdm_mc.rdm2x3_loop_oe((0,0), state, env, open_inds, unroll)

	print(f"{dims} {unroll}")
	benchmark.pedantic(test_f, args=(),\
		iterations=1, rounds=2, warmup_rounds=1)

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("open_inds",[[1,2,3,4]])
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_rdm2x3_loop_oe_semimanual(dims, open_inds, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	@optional_cuda_measure(f"rdm2x3_loop_oe_semimanual{dims} {open_inds} {unroll}")
	def test_f():
		rdm_mc.rdm2x3_loop_oe_semimanual((0,0), state, env, open_inds, unroll)

	print(f"{dims} {unroll}")
	benchmark.pedantic(test_f, args=(),\
		iterations=1, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("dims",test_dims)
def test_profile_rdm2x3_loop_trglringex_manual(dims, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	@optional_cuda_measure(f"rdm2x3_loop_trglringex_manual_{dims}")
	def test_f():
		rdm_mc.rdm2x3_loop_trglringex_manual((0,0), state, env)
		

	benchmark.pedantic(test_f, args=(),\
		iterations=1, rounds=2, warmup_rounds=1)