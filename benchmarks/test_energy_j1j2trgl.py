import context
import pytest
import torch
import config as cfg
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV, init_random
from models.spin_triangular import J1J2J4_1SITE, eval_nn_per_site, eval_nnn_per_site, eval_nn_and_chirality_per_site

import logging
logging.basicConfig(filename=f"{__file__}.log", filemode='w', level=logging.INFO)

test_dims=[(3,27), (3,54), (4,32)]


@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_j1j2_loop_oe_semimanual(dims, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	model= J1J2J4_1SITE(phys_dim=2, j1=1.0, j2=1.0, j4=0, jchi=0, global_args=cfg.global_args)

	def test_f():
		nn_h_v,nn_diag= eval_nn_per_site((0,0),state,env,model.R,model.R@model.R,model.SS,model.SS)
		nnn= eval_nnn_per_site((0,0),state,env,None,None,model.SS,looped=unroll,use_checkpoint=False)

	benchmark.pedantic(test_f, args=(),\
		iterations=1, rounds=2, warmup_rounds=1)

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_j1j2jX_loop_oe_semimanual(dims, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	model= J1J2J4_1SITE(phys_dim=2, j1=1.0, j2=1.0, j4=0, jchi=0, global_args=cfg.global_args)

	def test_f():
		nnn= eval_nnn_per_site((0,0),state,env,None,None,model.SS,looped=unroll,use_checkpoint=False)
		nn_h_v,nn_diag,chi= eval_nn_and_chirality_per_site((0,0),state,env,\
			model.R,model.R@model.R,model.SS,model.SS,model.h_chi,looped=unroll,use_checkpoint=False)

	benchmark.pedantic(test_f, args=(),\
		iterations=1, rounds=2, warmup_rounds=1)

@pytest.mark.parametrize("dims",test_dims)
@pytest.mark.parametrize("unroll",[True,False])
def test_profile_rdm2x3_loop_manual(dims, unroll, benchmark):
	D,X= dims

	state= IPEPS({(0,0): torch.rand((2,)+(D,)*4,\
            dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)-0.5}, lX=1, lY=1)
	env= ENV(X, state)
	init_random(env)

	model= J1J2J4_1SITE(phys_dim=2, j1=1.0, j2=1.0, j4=0, jchi=0, global_args=cfg.global_args)

	benchmark.pedantic(model.energy_1x3, args=(state,env,-1,unroll,\
        cfg.ctm_args,cfg.global_args),\
		iterations=1, rounds=2, warmup_rounds=1)