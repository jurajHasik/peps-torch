SU(2) model on Kagome lattice
=============================

Dense
^^^^^

This implementation of SU(2)-symmetric model on Kagome lattice does
not assume any symmetry and hence works with dense tensors.

.. automodule:: models.spin_half_kagome
.. autoclass:: S_HALF_KAGOME
    :members:

With explict U(1) symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation of SU(2)-symmetric model on Kagome lattice
assumes explicit U(1) abelian symmetry (subgroup) of iPEPS tensors.

.. automodule:: models.abelian.kagome_u1
.. autoclass:: KAGOME_U1
    :members:
