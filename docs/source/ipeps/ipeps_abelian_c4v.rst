Abelian-symmetric 1-site C4v iPEPS
==================================

Specialized class for iPEPS with single abelian-symmetric on-site tensor possesing
a (C4v) symmetry with respect to the permutation of its auxilliary 
indices

.. math:: a_{suldr} = e^{i\theta} a_{s\textrm{p}(uldr)},

where :math:`\textrm{p}` is a permutation associated to the symmetries of square lattice: rotations by :math:`\pi/2` and reflections. This property in turn 
implies (:math:`A_1`) symmetry for the double-layer tensor 
:math:`A_{(uu')(ll')(dd')(rr')}=a_{suldr}a_{su'l'd'r'}`.

.. autoclass:: ipeps.ipeps_abelian_c4v.IPEPS_ABELIAN_C4V
    :show-inheritance:
    :members:

.. automodule:: ipeps.ipeps_abelian_c4v
    :members: