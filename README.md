# tn-torch
### A tensor network library for two-dimensional lattice models 

tn-torch performs optimization of infinite Projected entangled-pair state (iPEPS) 
by direct energy minimization. The gradient is computed by backpropagation 
(reverse-mode Automatic differentiation).

Supports:
- spin systems
- arbitrary rectangular unit cells

#### Dependencies
- PyTorch 1.+ (see https://pytorch.org/)

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
