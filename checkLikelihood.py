from __future__ import print_function
import GPflow
import numpy as np
import gplvm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pods
import GPy
import time


def plot(X):
    if(fPlot):
        plt.figure(figsize=(5, 5))
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
        for i, c in zip(np.unique(labels), colors):
            plt.scatter(X[labels==i, 0], X[labels==i, 1], color=c, label=i)
        plt.legend(numpoints=1)

pods.datasets.overide_manual_authorize = True  # dont ask to authorize
plt.style.use('ggplot')

Qn = 2  # number of non-linear latent dimensions
Qlin = 0
fPlot = False  # do we do plots?
fOptimize = True
if(fPlot): 
    plt.ion()
# data
data = pods.datasets.oil_100()
Y = data['X']
labels = data['Y'].argmax(axis=1)

# Bayesian GPLVM parameters
X_mean = GPflow.gplvm.PCA_reduce(Y, Qn+Qlin)
X_var = np.random.uniform(0, .1, X_mean.shape)
M = 25
Z = np.random.permutation(X_mean.copy())[:M]
# PCA fit
XPCA = GPflow.gplvm.PCA_reduce(Y, 2)
plot(XPCA)


print('Old Psi stats model===============')
ko = GPflow.kernels.RBF(Qn, ARD=True)
t = time.time()
mo = gplvm.BayesianGPLVM(X_mean=X_mean.copy(),
                         X_var=X_var.copy(), Y=Y, kern=ko, M=M, Z=Z.copy())
lo = mo.compute_log_likelihood()
if(fOptimize):
    mo.optimize(display=True, max_iters=100)
timeOld = time.time()-t
plot(mo.X_mean.value)
print('GPflow model kernexp branch===============')
t = time.time()
ke = GPflow.ekernels.RBF(Qn, ARD=True)
me = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean.copy(),
                                X_var=X_var.copy(), Y=Y, kern=ke, M=M, Z=Z.copy())
le = me.compute_log_likelihood()
if(fOptimize):
    me.optimize(display=True, max_iters=100)
timeKe = time.time()-t
plot(me.X_mean.value)

print('Likelihood at start OldPsi=%.2f, kernexp=%.2f' % (lo, le))
print('Time OldPsi=%g, kernexp=%g seconds.' % ( timeOld, timeKe))

assert np.allclose(lo, le, atol=1e-6), ('initial likelihoos should be same', lo, le)
