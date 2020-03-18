iPEPS
=====

The following iPEPS classes define various wavefunctions on a square lattice.
In essence, they hold a set of one or more rank-5 on-site tensors (the variational 
parameters of the iPEPS) and encode the tilling of the lattice by these tensors. 
The tiling is defined by ``vertexToSite`` function, which takes a tuple of integers 
``(x,y)``, indicating a vertex of the square lattice, and returns an on-site tensor. 

.. toctree::
    :glob:

    ipeps/*
