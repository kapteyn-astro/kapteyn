#!/usr/bin/env python
# Demonstrate use of variance reduction to examine a model.
# Use data example from Bevington & Robinson with 1 outlier

from numpy.random import normal
import numpy
from kapteyn import kmpfit
from matplotlib.pyplot import figure, show, rc

def model_lin(p, x):
   a, b = p
   y = a + b*x
   return y

def model_par(p, x):
   a, b, c = p
   y = a + b*x + c*x*x
   return y

def residuals(p, data):
   x, y, err, mod = data
   if mod == 1:
      return (y-model_lin(p,x))/err
   else:
      return (y-model_par(p,x))/err

# Data
x = numpy.array([1,2,3,4,5,6,7,8,9, 10.0])
y = numpy.array([2.047, -0.966, -1.923, -1.064, 2.048, 6.573, 13.647, 24.679, 34.108, 44.969])
err = numpy.array([0.102, 0.048, 0.096, 0.053, 0.102, 0.329, 0.682, 1.234, 1.705, 2.248])
#err = numpy.ones(len(y))

# Do the fit and find the Variance Reduction
params0 = (1.6,0.5)
fitter = kmpfit.Fitter(residuals=residuals, data=(x,y,err,1))
fitter.fit(params0=params0)

print "======== Fit straight line =========="
print "Params:                      ", fitter.params
print "Uncertainties:               ", fitter.xerror
print "Errors assuming red.chi^2=1: ", fitter.stderr
print "Iterations:                  ", fitter.niter 
print "Function ev:                 ", fitter.nfev
print "dof:                         ", fitter.dof
print "chi^2, rchi2:                ", fitter.chi2_min, fitter.rchi2_min
print "Status:                      ", fitter.status

N = len(y)
varmod = (y-model_lin(fitter.params,x))**2.0
y_av = y.sum()/N
vardat = (y-y_av)**2.0
vr0 = 100.0*(1-(varmod.sum()/vardat.sum()))
print "Variance reduction (%):", vr0

params0 = (1,1, 0)
fitter2 = kmpfit.Fitter(residuals=residuals, data=(x,y,err,2))
fitter2.fit(params0=params0)

print "\n======== Fit results Parabola =========="
print "Params:                      ", fitter2.params
print "Uncertainties:               ", fitter2.xerror
print "Errors assuming red.chi^2=1: ", fitter2.stderr
print "Iterations:                  ", fitter2.niter 
print "Function ev:                 ", fitter2.nfev
print "dof:                         ", fitter2.dof
print "chi^2, rchi2:                ", fitter2.chi2_min, fitter2.rchi2_min
print "Status:                      ", fitter2.status

N = len(y)
varmod = (y-model_par(fitter2.params,x))**2.0
y_av = y.sum()/N
vardat = (y-y_av)**2.0
vr1 = 100.0*(1-(varmod.sum()/vardat.sum()))
print "variance reduction (%):", vr1

# Prepare plot
fig = figure()
rc('legend', fontsize=8)
frame = fig.add_subplot(1,1,1)
frame.set_xlabel("x")
frame.set_ylabel("y")
frame.set_title("Improve model using Variance Reduction")
frame.errorbar(x, y, yerr=err, fmt='bo')
delta = (x.max()-x.min())/10.0
X = numpy.linspace(x.min()-delta, x.max()+delta, 100)
label="Model: $a + bx$ VR=%.2f"%vr0
frame.plot(X, model_lin(fitter.params,X), 'g', label=label)
label="Model: $a + bx +cx^2$ VR=%.2f"%vr1
frame.plot(X, model_par(fitter2.params,X), 'm', label=label)
frame.set_xlim(x.min()-delta, x.max()+delta)
leg = frame.legend(loc=2)
show()