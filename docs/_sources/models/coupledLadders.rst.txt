Coupled Ladders
===============

Dense (PyTorch)
^^^^^^^^^^^^^^^ 

This implementation of SU(2)-symmetric model of coupled spin-1/2 ladders 
on square lattice does not assume any symmetry and works with dense PyTorch tensors.

.. automodule:: models.coupledLadders
.. autoclass:: COUPLEDLADDERS
    :members:


Dense (YAST)
^^^^^^^^^^^^

This implementation of SU(2)-symmetric model of coupled spin-1/2 ladders 
works with dense YAST tensors. In particular, the physical reduced density matrices, 
Hamiltonian terms, observables, etc. are dense YAST tensors. As such, their indices have signature.

.. note::
    The underlying iPEPS wavefunction can make use of explicit internal symmetry and thus 
    block-sparse tensors. After physical reduced density matrices are built from such
    iPEPS and its environment, they are converted to dense form (keeping signature information).


.. automodule:: models.abelian.coupledLadders
.. autoclass:: COUPLEDLADDERS_NOSYM
    :members:


With explict U(1) symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation of SU(2)-symmetric model of coupled spin-1/2 ladders 
works with explicit U(1) abelian symmetry (subgroup). The physical reduced density matrices, 
Hamiltonian terms, observables, etc. have U(1) block-sparse structure.

.. automodule:: models.abelian.coupledLadders
    :noindex:
.. autoclass:: COUPLEDLADDERS_U1
    :members:
