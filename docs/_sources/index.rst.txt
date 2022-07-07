.. tn-torch documentation master file, created by
   sphinx-quickstart on Mon Jul 22 18:30:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

peps-torch
====================================

The `peps-torch` performs gradient optimization of two-dimensional iPEPS
tensor networks. It is primarily composed of two parts:  
various iPEPS defined through one of the :doc:`ipeps` classes and 
their :doc:`environments <envs>` computed by the corner-transfer matrix 
algorithm. 

In order to optimize an iPEPS define a loss function which computes the 
variational energy with respect to target Hamiltonian.
Generally, to evaluate the energy, first :doc:`compute the environment <envs/generic/ctmrg>` 
of iPEPS and then the appropriate reduced density matrices. Afterwards,
the individual terms of Hamiltonian can be computed with these reduced
density matrices. There are several examples of energy computation 
for different :doc:`spin models <models>`, e.g. :doc:`spin=2 AKLT model<models/akltS2>`. Finally, invoke the :doc:`optimization <optim>`. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ipeps
   envs
   optim
   config
   models
   groups
   linalg

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
