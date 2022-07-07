1-site iPEPS from lin. combination
==================================

Specialized class for iPEPS with single on-site tensor built
as a linear combination of a set of elementary tensors `e` 

.. math:: a_{suldr} = \sum_i \lambda_i e^i_{suldr},

where :math:`\vec{\lambda}` is a vector of real coefficients, which are the variatonal parameters of this ansatz.

.. autoclass:: ipeps.ipeps_lc.IPEPS_LC_1SITE_PG
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipeps_lc
    :exclude-members: IPEPS_LC, IPEPS_LC_1SITE_PG
    :members: