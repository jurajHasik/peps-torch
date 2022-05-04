Kagome iPESS with point groups
==============================

Specialized sub-class of :class:`ipeps.ipess_kagome.IPESS_KAGOME_GENERIC`.
This iPESS ansatz can be constrained in several ways:
    
    * make trivalent tensor of up and down triangles identical
    * make bond tensors identical
    * constrain individual iPESS tensors to some point group irrep 

Supported point groups are :math:`A` and :math:`B` for bond tensor and
:math:`A_1` and :math:`A_2` for trivalent tensors.

.. autoclass:: ipeps.ipess_kagome.IPESS_KAGOME_PG
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipess_kagome
    :noindex:
    :members: to_PG_symmetric, read_ipess_kagome_pg, write_ipess_kagome_pg