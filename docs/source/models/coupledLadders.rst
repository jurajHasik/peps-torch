Coupled Ladders
===============

Dense
^^^^^

This implementation of SU(2)-symmetric model of coupled spin-1/2 ladders 
on square lattice does not assume any symmetry and hence works with dense tensors.

.. automodule:: models.coupledLadders
.. autoclass:: COUPLEDLADDERS
    :members:


With explict U(1) symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation of SU(2)-symmetric model on Kagome lattice
assumes explicit U(1) abelian symmetry (subgroup) of iPEPS tensors.

.. automodule:: models.abelian.coupledLadders
.. autoclass:: COUPLEDLADDERS_U1
    :members:
