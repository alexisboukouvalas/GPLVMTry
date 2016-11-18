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
fPlot = True  # do we do plots?
if(fPlot): plt.ion()
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

print('GPy fit===============')
t = time.time()
if(Qlin > 0):
    ky = GPy.kern.Add([
                GPy.kern.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn)),
                GPy.kern.Linear(Qlin, ARD=True, active_dims=np.arange(Qn, Qn+Qlin))
            ])
else:
    ky = GPy.kern.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn))

my = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
     Y, Qn + Qlin, X=X_mean, X_variance=X_var,
     kernel=ky, num_inducing=M)
my.likelihood[:] = my.Y.values.var()/10.
my.X.variance[:] = .1
my.kern['.*lengthscale'].fix()
if(Qlin > 0):
    my.kern['.*variances'].fix(my.Y.values.var()/1e5)
my.likelihood.fix()
my.update_model(True)
ly = -my.objective_function()
my.optimize(max_iters=500, messages=1)
my.likelihood.unfix()
my.optimize(max_iters=500, messages=1)
my.kern.unfix()
my.optimize(max_iters=1e5, messages=1)

my.data_labels = labels
timeGPy = time.time()-t
if fPlot:
    my.kern.plot_ARD()
    my.plot_latent(labels=labels, which_indices=[0, 1])
print(my)

print('Old Psi stats model===============')
if(Qlin > 0):
    ko = GPflow.kernels.Add([
                GPflow.kernels.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn)),
                GPflow.kernels.Linear(Qlin, ARD=True, active_dims=np.arange(Qn, Qn+Qlin))
            ])
else:
    ko = GPflow.kernels.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn))

t = time.time()
mo = gplvm.BayesianGPLVM(X_mean=X_mean,
                         X_var=X_var, Y=Y, kern=ko, M=M, Z=Z)
lo = mo.compute_log_likelihood()
mo.optimize(display=False)
timeOld = time.time()-t
# plot biggest ARD len scale
plot(mo.X_mean.value)
# assert np.allclose(lo, ly, atol=1e-6), (lo, ly)
#
print('GPflow model kernexp branch===============')
t = time.time()
if(Qlin > 0):
    ke = GPflow.kernels.Add([
                GPflow.ekernels.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn)),
                GPflow.ekernels.Linear(Qlin, ARD=True, active_dims=np.arange(Qn, Qn+Qlin))
            ])
else:
    ke = GPflow.ekernels.RBF(Qn, ARD=True, active_dims=np.arange(0, Qn))
me = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean,
                                X_var=X_var, Y=Y, kern=ke, M=M)
le = me.compute_log_likelihood()
me.optimize(display=False)
timeKe = time.time()-t
plot(me.X_mean.value)

print('Likelihood at start GPy=%.2f, OldPsi=%.2f, kernexp=%.2f' % (ly, lo, le))
print('Time GPy=%g , OldPsi=%g, kernexp=%g seconds.' % (timeGPy, timeOld, timeKe))

assert np.allclose(lo, le, atol=1e-6), ('initial likelihoos should be same', lo, le)
