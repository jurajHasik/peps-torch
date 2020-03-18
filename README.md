# peps-torch [![Build Status](https://travis-ci.com/jurajHasik/tn-torch.svg?branch=master)](https://travis-ci.com/jurajHasik/tn-torch)
### A tensor network library for two-dimensional lattice models
by Juraj Hasik, Glen Bigan Mbeng

\
peps-torch performs optimization of infinite Projected entangled-pair states (iPEPS) 
by direct energy minimization. The gradients are computed by backpropagation 
(reverse-mode Automatic differentiation).

For the full documentation, continue to [peps-torch.readthedocs.io](https://peps-torch.readthedocs.io) 

Supports:
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
