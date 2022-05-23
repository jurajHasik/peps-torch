Generic Kagome iPESS
====================

Specialized sub-class of :class:`ipeps.ipeps_kagome.IPEPS_KAGOME`. 
A single on-site tensor representing DoFs on down triangle is built from 
five different tensors: two rank-3 trivalent tensors with only auxiliary indices 
and three rank-3 bond tensors, each associated to one of the physical DoFs on the vertices
of the down triangle.

.. autoclass:: ipeps.ipess_kagome.IPESS_KAGOME_GENERIC
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipess_kagome
    :members: read_ipess_kagome_generic, write_ipess_kagome_generic