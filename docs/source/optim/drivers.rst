Optimization drivers
--------------------

.. toctree::
    :glob:
    :hidden:

    drivers/*

These modules perform optimization of iPEPS by one of the supported
optimizers. In essence, they create appropriate closures from
supplied loss function for both regular and line search steps
and the invoke desired optimizer.

* :mod:`ad_optim <optim.ad_optim>` optimize with default PyTorch LBFGS optimizer
* :mod:`optim.ad_optim_lbfgs_mod` optimize with :doc:`extended LBFGS <optimizers/lbfgs_modified>` optimizer 
  supporting derivative-free backtracking linesearch
* :mod:`optim.ad_optim_sgd_mod` optimize with :doc:`extended SGD <optimizers/sgd_modified>` optimizer
  supporting derivative-free backtracking linesearch
