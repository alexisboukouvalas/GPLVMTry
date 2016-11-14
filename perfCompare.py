from __future__ import print_function
import GPflow
import numpy as np
import gplvm

# data
import GPy
N = 100
Q = 2  # latent dimensions
M = 5  # inducing points
output_dim = 10
# generate GPLVM-like data
X = np.random.rand(N, Q)
lengthscales = np.random.rand(Q)
k = GPy.kern.RBF(Q)
K = k.K(X)
Y = np.random.multivariate_normal(np.zeros(N), K, (output_dim,)).T

# Initial conditions and parameters
X_mean = GPflow.gplvm.PCA_reduce(Y, Q)

# GPy fit
my = GPy.models.BayesianGPLVM(Y, Q, X=X_mean, kernel=k, num_inducing=M)
llGpy = -my.objective_function()
my.optimize('scg', messages=True, max_iters=2)

# Old Psi stats model
ko = GPflow.kernels.RBF(Q)
mo = gplvm.BayesianGPLVM(X_mean=X_mean,
                         X_var=np.ones((N, Q)), Y=Y, kern=ko, M=M)
llke = mo.compute_log_likelihood()
mo.optimize(maxiter=2)

# GPflow model kernexp branch
k = GPflow.ekernels.RBF(Q)
m = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean,
                               X_var=np.ones((N, Q)), Y=Y, kern=k, M=M)
llo = m.compute_log_likelihood()
m.optimize(maxiter=2)

print('Likelihood at start GPy=%.2f, OldPsi=%.2f, kernexp=%.2f' % (llGpy, llke, llo))
