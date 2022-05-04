iPEPS
=====

The following iPEPS classes define various wavefunctions on a square lattice.

First category covers simple iPEPS, which are defined by a set of one or more rank-5 on-site tensors (the variational parameters of the iPEPS) and encode the tilling of the lattice by these tensors. The tiling is defined by ``vertexToSite`` function, which takes a tuple of integers ``(x,y)``, indicating a vertex of the square lattice, and returns an on-site tensor. 

.. toctree::
    :glob:

    ipeps/ipeps
    ipeps/ipeps_kagome


Second category is formed by specialized classes of iPEPS. For example, constrained 
by spatial symmetries or with on-site tensors possesing additional structure.

.. toctree::
    :glob:

    ipeps/ipeps_c4v
    ipeps/ipeps_lc_1site_pg
    ipeps/ipess_kagome
    ipeps/ipess_kagome_pg
    ipeps/ipess_kagome_pg_lc

