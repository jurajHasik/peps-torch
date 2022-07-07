Abelian-symmetric generic Kagome iPESS
======================================

Specialized sub-class of :class:`ipeps.ipeps_kagome_abelian.IPEPS_KAGOME_ABELIAN`. 
A single on-site tensor representing DoFs on down triangle is built from 
five different tensors: two rank-3 trivalent tensors with only auxiliary indices 
and three rank-3 bond tensors, each associated to one of the physical DoFs on the vertices of the down triangle.

.. autoclass:: ipeps.ipess_kagome_abelian.IPESS_KAGOME_GENERIC_ABELIAN
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipess_kagome_abelian
    :members: read_ipess_kagome_generic, write_ipess_kagome_generic