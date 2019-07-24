# tn-torch


## test run for coupled ladders
Run optimization with a noiseless initialization (this will be stuck... local extremum)
command:
	python ./optim_ladders.py -bond_dim 3 -chi 9 -instate test-input/VBS_2x2_ABCD.in

Note that the input has a different convention for the indices:
	state = read_ipeps(args.instate, peps_args=PEPSARGS(), global_args=GLOBALARGS(), aux_seq=[1,0,3,2])

adding noise of absolute value 0.073
command:
	python ./optim_ladders.py -bond_dim 3 -chi 9 -instate test-input/VBS_2x2_ABCD.in -instate_noise 0.073