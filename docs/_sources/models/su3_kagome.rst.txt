SU(3) model on Kagome lattice
=============================

Dense
^^^^^

This implementation of SU(3)-symmetric model on Kagome lattice does
not assume any symmetry and hence works with dense tensors.

.. automodule:: models.su3_kagome
.. autoclass:: KAGOME_SU3
    :members:

With explict U(1)xU(1) symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation of SU(3)-symmetric model on Kagome lattice
assumes explicit U(1)xU(1) abelian symmetry (subgroup) 
of iPEPS tensors.

.. automodule:: models.abelian.su3_kagome
.. autoclass:: KAGOME_SU3_U1xU1
    :members: