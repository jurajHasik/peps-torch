# peps-torch ![Test Status](https://github.com/jurajHasik/peps-torch/actions/workflows/main.yml/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/peps-torch/badge/?version=latest)](https://peps-torch.readthedocs.io/en/latest/?badge=latest)
### A tensor network library for two-dimensional lattice models
by Juraj Hasik, Glen Bigan Mbeng\
with contributions by Wei-Lin Tu, Seydou-Samba Diop, Sen Niu, Yi Xi


\
**peps-torch** performs optimization of infinite Projected entangled-pair states (iPEPS) 
by direct energy minimization. The energy is evaluated using environments obtained 
by the corner-transfer matrix (CTM) algorithm. Afterwards, the gradients are computed by reverse-mode 
automatic differentiation (AD).

#### Now supporting 
* **abelian symmetries, with implementation power by [YAST](https://gitlab.com/marekrams/yast)**

  Allows definition of abelian-symmetric iPEPS in terms of block-sparse tensors, computing their 
  symmetric environments, and their optimization with gradient-based methods.

* **complex tensors with PyTorch 1.11+**
  
  extended L-BFGS optimizer for handling complex-valued tensors.
  
* **rich set of models and their examples**

  various spin [models](https://jurajhasik.github.io/peps-torch/models.html) on square and Kagome lattices, 
  using both dense and abelian-symmetric tensors. Wide set of [examples](https://github.com/jurajHasik/peps-torch/tree/master/examples) 
  which show how to optimize and compute observables for these models.       

<br>

**For the full documentation, continue to** [jurajhasik.github.io/peps-torch](https://jurajhasik.github.io/peps-torch/)
* * *
<br>

#### Quickstart: Examples with J1-J2 model on square lattice
**Ex. 1) Optimize one-site (C4v) symmetric iPEPS** with bond dimension D=2
and environment dimension X=32 for J1-J2 model at J2/J1=0.3 run 

```
python examples/j1j2/optim_j1j2_c4v.py --bond_dim 2 --chi 32 --seed 123 --j2 0.3 --out_prefix ex-c4v
```
Using the resulting state `ex-c4v_state.json`, compute the observables such as spin-spin 
and dimer-dimer correlations for distance up to 20 sites

```
python examples/j1j2/ctmrg_j1j2_c4v.py --instate ex-c4v_state.json --chi 48 --j2 0.3 --corrf_r 20
```

\
**Ex. 2) Optimize iPEPS with 2x2 unit cell** containing four distinct on-site tensors by running

```
python examples/j1j2/optim_j1j2.py --tiling 4SITE --bond_dim 2 --chi 32 --seed 123 --j2 0.3 \
--CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-4site
```

The memory requirements of AD would increase sharply if all the intermediate variables are stored.
Instead, by passing `--CTMARGS_fwd_checkpoint_move` flag we opt for recomputing them on the fly 
while saving only the intermediate tensors of the CTM environment.

Compute observables and spin-spin correlation functions in horizontal and vertical direction
of the resulting state

```
python examples/j1j2/ctmrg_j1j2.py --tiling 4SITE --chi 48 --j2 0.3 --instate ex-4site_state.json --corrf_r 20
```

\
**Ex. 3) Optimize one-site iPEPS and impose both C4v symmetry and U(1) symmetry.** We choose a particular U(1) class,
defined in `u1sym/D4_U1_B.txt`, given by a set of 25 elementary tensors. The on-site tensor is then constructed
as their linear combination. This approach to symmetric iPEPS has been developed in Ref. [4]. Run 

```
python examples/j1j2/optim_j1j2_u1_c4v.py --bond_dim 4 --u1_class B --chi 32 --j2 0.2 \
--OPTARGS_line_search backtracking --OPTARGS_line_search_svd_method SYMARP --CTMARGS_fwd_checkpoint_move \
--out_prefix ex-u1b
```

The optimization is performed together with backtracking linesearch. Moreover, the CTM steps during linesearching are accelerated
by using partial eigenvalue decomposition (SCIPY's Arnoldi) instead of full-rank one.

Using the resulting state `ex-u1b_state.json`, compute the observables such as leading part of transfer matrix spectrum or spin-spin 
and dimer-dimer correlations for distance up to 20 sites

```
python examples/j1j2/ctmrg_j1j2_u1_c4v.py --instate ex-u1b_state.json --j2 0.2 --chi 32 --top_n 4 --corrf_r 20
```

\
**Ex. 4) Optimize complex-valued one-site (C4v) symmetric iPEPS**. Simply pass the flag `--GLOBALARGS_dtype complex128`. For instance,
the Ex. 1 becomes

```
python examples/j1j2/optim_j1j2_c4v.py --GLOBALARGS_dtype complex128 --bond_dim 3 --chi 32 \
--seed 123 --j2 0.5 --out_prefix ex-c4v
```
The computation of expectations values can be done right away, as the state stored in `ex-c4v_state.json` carries the 
datatype (dtype) of its tensors
```
python examples/j1j2/ctmrg_j1j2_c4v.py --GLOBALARGS_dtype complex128 --instate ex-c4v_state.json \
--chi 48 --j2 0.5 --corrf_r 20
```

\
**Ex. 5) Optimize U(1)-symmetric iPEPS with 2 sites in the unit cell**. The U(1)-symmetric tensors are implemented in block-sparse
fashion using YAST (see Dependencies). To specify the ansatz, one has to select charge sectors and their sizes for each
tensor making up the ansatz. Here, we adopt U(1)-structure determined
in Ref. [5]. which is provided in one of the test states, available in `test-input/abelian/c4v`. Run 


```
python examples/j1j2/abelian/optim_j1j2_u1.py --tiling BIPARTITE \
--instate test-input/abelian/c4v/BFGS100LS_U1B_D3-chi72-j20.0-run0-iRNDseed321_blocks_2site_state.json \
--chi 32 --j2 0.3 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-2site-u1
```
The computation of expectations values can be carried out by invoking
```
python examples/j1j2/abelian/ctmrg_j1j2_u1.py --tiling BIPARTITE --chi 48 --j2 0.3 --instate ex-2site-u1_state.json
```

* * *
<br>

#### Supports:
- spin systems
- arbitrary rectangular unit cells
- abelian-symmetric tensors
- both real- and complex-valued tensors

#### Dependencies
- PyTorch 1.11+ (see https://pytorch.org/)
- (optional) YAST (see https://gitlab.com/marekrams/yast)
- (optional) scipy 1.3.+

YAST is linked to **peps-torch** as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
To obtain it, you can use git:

`git submodule update --init --recursive`

#### Building documentation
- PyTorch 1.11+
- sphinx
- sphinx_rtd_theme

All the dependencies can be installed through ``conda`` (see https://docs.conda.io).

Afterwards, build documentation as follows:

`cd docs && make html`

The generated documentation is found at `docs/build/html/index.html`
* * *
\
Inspired by the pioneering work of Hai-Jun Liao, Jin-Guo Liu, Lei Wang, and Tao Xiang,
[Phys. Rev. X 9, 031041](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041) or arXiv version [arXiv:1903.09650](https://arxiv.org/abs/1903.09650)

References:

1.	*Corner Transfer Matrix Renormalization Group Method*, T. Nishino and K. Okunishi, 
	[Journal of the Physical Society of Japan 65, 891 (1996)](https://journals.jps.jp/doi/10.1143/JPSJ.65.891) 
	or arXiv version [arXiv:cond-mat/9507087 ](https://arxiv.org/abs/cond-mat/9507087)
2.	*Faster Methods for Contracting Infinite 2D Tensor Networks*,  
	M.T. Fishman, L. Vanderstraeten, V. Zauner-Stauber, J. Haegeman, and F. Verstraete,
	[Phys. Rev. B 98, 235148 (2018) ](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235148)
	or arXiv version [arXiv:1711.05881 ](https://arxiv.org/abs/1711.05881)
3.	*Competing States in the t-J Model: Uniform d-Wave State versus Stripe State (Supplemental Material)*, 
	P. Corboz, T. M. Rice, and M. Troyer, [Phys. Rev. Lett. 113, 046402 (2014) ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.046402>) or arXiv version 
	[arXiv:1402.2859](https://arxiv.org/abs/1402.2859)
4.	*Systematic construction of spin liquids on the square lattice from tensor networks with SU(2) symmetry*, 
	M. Mambrini, R. Orus, and D. Poilblanc, [Phys. Rev. B 94, 205124 (2016)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.205124) or arXiv version [arXiv:1608.06003](https://arxiv.org/abs/1608.06003)
5.	*Investigation of the Néel phase of the frustrated Heisenberg antiferromagnet by differentiable symmetric tensor networks*,
	J. Hasik, D. Poilblanc, and F. Becca, [SciPost Phys. 10, 012 (2021)](https://scipost.org/submissions/scipost_202011_00009v2/)