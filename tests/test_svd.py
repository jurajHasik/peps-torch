import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds

rng = np.random.default_rng()
orthogonal = csc_matrix(ortho_group.rvs(10, random_state=rng))
s = [0.0001, 0.001, 3, 4, 5]  # singular values
u = orthogonal[:, :5]         # left singular vectors
vT = orthogonal[:, 5:].T      # right singular vectors
A = u @ diags(s) @ vT

u2, s2, vT2 = svds(A, k=3, solver='propack')

print(s2)

A2 = u2 @ np.diag(s2) @ vT2
np.allclose(A2, A.todense(), atol=1e-3)


A= np.random.rand(5,5) + np.random.rand(5,5)*1j
u2, s2, vT2 = svds(A, k=2, solver='propack')
print(s2)
