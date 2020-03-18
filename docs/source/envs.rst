Environments
============

The evalution of observables of iPEPS is realized through reduced
density matrices. In order to construct them, one first computes
an effective `environment` consisting of a set of tensors that
approximate contraction of (infinite) parts of the tensor network.
Once done, these environment tensors can be combined with on-site
tensors to build various reduced density matrices.
Following modules encapsulate CTMRG algorithm which computes 
the environment tensors for various iPEPS. Once done, the convenience 
functions facilitate construction of reduced density matrices of commonly used 
regions and computation correlations functions. 

.. toctree::
    :glob:

    envs/*
