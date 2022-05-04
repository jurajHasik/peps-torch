Kagome iPESS from lin. combination
==================================

Specialized sub-class of :class:`ipeps.ipess_kagome.IPESS_KAGOME_PG`,
with iPESS tensors built as linear combination of a set of basis  
tensors `e` 

.. math:: 

    t = \sum_i \lambda_i e^i,

where :math:`\vec{\lambda}` is a vector of real coefficients, which are the variatonal parameters of this ansatz. Both 
trivalent tensors ``'T_u'``, ``'T_d'`` and bond tensors ``'B_a'``, ``'B_b'``, ``'B_c'`` are defined in this way.

.. autoclass:: ipeps.ipess_kagome.IPESS_KAGOME_PG_LC
    :show-inheritance:
    :members:
    :exclude-members: extend_bond_dim

.. automodule:: ipeps.ipess_kagome
    :noindex:
    :members: write_ipess_kagome_pg_lc, read_ipess_kagome_pg_lc