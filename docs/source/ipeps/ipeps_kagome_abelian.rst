Abelian-symmetric Kagome iPEPS
==============================

Specialized class for abelian-symmetric iPEPS on Kagome lattice, where sites `s0,s1,s2` on 
down-triangles are fused together::

           |/      |/
          s1      s1
         / |     / |                  |               |
        /  |    /  |                  |/(s0,s1,s2)    |/(s0,s1,s2)
        --s0--s2--s0--s2--   =>     --a---------------a--
           | /     | /                |               |
           |/      |/                 |               |
          s1      s1                  |               |
         / |     / |                  |               |
        /  |    /  |                  |/(s0,s1,s2)    |/(s0,s1,s2)
    --s2--s0--s2--s0--s2--          --a---------------a--
           | /     | /                |               |
           |/      |/
          s1      s1
          /|      /|

The resulting tensor network is defined on a square lattice in terms of rank-5 on-site tensors.
Physical index runs over the product of Hilbert spaces of `s0,s1,s2` in this order. 
These Hilbert spaces are assumed to be identical.

.. autoclass:: ipeps.ipeps_kagome_abelian.IPEPS_KAGOME_ABELIAN
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipeps_kagome_abelian
    :exclude-members: IPEPS_KAGOME_ABELIAN
    :members: