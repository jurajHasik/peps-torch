Custom optimizers
-----------------

Evaluation of energy of iPEPS involves the computation of environment (CTM)
which is typically the bottleneck of the optimization.

Currently, the reverse-mode differentiation of CTM relies on using full-rank decomposition
when constructing the projectors. Using line search algorithms which do not rely 
on derivatives allows for truncated decomposition in CTM and leads to considerable 
speed-up. The following optimizers (based on their PyTorch parents) implement 
backtracking line search and provide support for specialized cost functions
to be used within line search context.

.. toctree::
    :glob:

    optimizers/*
