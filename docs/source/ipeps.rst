iPEPS
=====

The following iPEPS classes define various wavefunctions on a square lattice.

First category covers simple iPEPS, which are defined by a set of one or more rank-5 on-site tensors (the variational parameters of the iPEPS). These on-site tensors are arranged in a unit-cell,
specified as dictionary ``sites= {(x,y): tensor, ...}``, where tuple ``(x,y)`` denotes a site within a unit cell. The unit cell then tiles the entire square lattice. To encode the precise way in which the lattice is tiled by these tensors, one defines a ``vertexToSite`` function, which takes a tuple of integers ``(x,y)``, indicating a vertex of the square lattice, and returns an on-site tensor 
from the unit cell.

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

