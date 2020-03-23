# peps-torch [![Build Status](https://travis-ci.com/jurajHasik/peps-torch.svg?branch=master)](https://travis-ci.com/jurajHasik/tn-torch) [![Documentation Status](https://readthedocs.org/projects/peps-torch/badge/?version=latest)](https://peps-torch.readthedocs.io/en/latest/?badge=latest)
### A tensor network library for two-dimensional lattice models
by Juraj Hasik, Glen Bigan Mbeng

\
peps-torch performs optimization of infinite Projected entangled-pair states (iPEPS) 
by direct energy minimization. The energy is evaluated using environments obtained 
by the corner-transfer matrix (CTM) algorithm. Afterwards, the gradients are computed by reverse-mode 
automatic differentiation (AD).

For the full documentation, continue to [peps-torch.readthedocs.io](https://peps-torch.readthedocs.io) 

#### Example with J1-J2 model on square lattice
Optimize one-site (C4v) symmetric iPEPS with bond dimension D=2
and environment dimension X=32 for J1-J2 model at J2/J1=0.3 run 

```
python examples/optim_j1j2_c4v.py -bond_dim 2 -chi 32 -seed 123 -j2 0.3 -out_prefix ex-c4v
```
Using the resulting state `ex_state.json`, compute the observables such as spin-spin 
and dimer-dimer correlations for distance up to 20 sites

```
python examples/ctmrg_j1j2_c4v.py -instate ex-c4v_state.json -chi 48 -j2 0.3 -corrf_r 20
```

To instead optimize iPEPS with 2x2 unit cell containing four distinct on-site tensors run

```
python examples/optim_j1j2.py -tiling 4SITE -bond_dim 2 -chi 32 -seed 100 -j2 0.3 -CTMARGS_fwd_checkpoint_move -OPTARGS_tolerance_grad 1.0e-8 -out_prefix ex-4site
```

The memory requirements of AD would increasly sharply if all the intermediate variables are stored.
Instead, by passing `-CTMARGS_fwd_checkpoint_move` flag we opt for recomputing them on the fly 
while saving only the intermediate tensors of CTM environment.

Compute observables and spin-spin correlation functions in horizontal and vertical direction
of the resulting state

```
python examples/ctmrg_j1j2.py -tiling 4SITE -chi 48 -j2 0.3 -instate ex-4site_state.json -corrf_r 20
```

#### Supports:
- spin systems
- arbitrary rectangular unit cells
- only real-valued tensors

#### Dependencies
- PyTorch 1.+ (see https://pytorch.org/)
- (optional) scipy 1.3.+

#### Building documentation
- PyTorch 1.+
- sphinx
- sphinx_rtd_theme


All the dependencies can be installed through ``conda`` (see https://docs.conda.io).

Afterwards, build documentation as follows:

`cd docs && make html`

The generated documentation is found at `docs/build/html/index.html`

\
\
Inspired by the pioneering work of Hai-Jun Liao, Jin-Guo Liu, Lei Wang, and Tao Xiang

https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 
