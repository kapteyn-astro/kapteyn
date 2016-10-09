#!/usr/bin/env python
# Demonstrate use of variance reduction to exclude poor data
# Use data example from Bevington & Robinson with 1 outlier

from numpy.random import normal
import numpy
from kapteyn import kmpfit
from matplotlib.pyplot import figure, show, rc

def model(p, x):
   a, b = p
   y = a+b*x
   return y

def residuals(p, data):
   a, b = p
   x, y, err = data
   return (y-model(p,x))/err

# Artificial data
# x = numpy.array([2, 4, 6, 8, 10, 12])
# y = numpy.array([3.5, 7.2, 9.5, 17.1, 20.0, 25.5])
# err = numpy.array([0.55, 0.65, 0.74, 0.5, 0.85, 0.6])

# With outlier
x = numpy.array([25, 2, 4, 6, 8, 10, 12])
y = numpy.array([15.0, 3.5, 7.2, 9.5, 17.1, 20.0, 25.5])
#err = numpy.array([0.6, 0.55, 0.65, 0.74, 0.5, 0.85, 0.6])
err = numpy.ones(len(y))

# Prepare plot
fig = figure()
rc('legend', fontsize=8)
frame = fig.add_subplot(1,1,1)
frame.plot(x, y, 'go', label="data")
frame.set_xlabel("x")
frame.set_ylabel("y")
frame.set_title("Exclude poor data with variance reduction")

# Do the fit and find the Variance Reduction
params0 = (1,1)
fitter = kmpfit.Fitter(residuals=residuals, data=(x,y,err))
fitter.fit(params0=params0)

print("======== Fit results all data included ==========")
print("Params:                      ", fitter.params)
print("Uncertainties:               ", fitter.xerror)
print("Errors assuming red.chi^2=1: ", fitter.stderr)
print("Iterations:                  ", fitter.niter) 
print("Function ev:                 ", fitter.nfev)
print("dof:                         ", fitter.dof)
print("chi^2, rchi2:                ", fitter.chi2_min, fitter.rchi2_min)
print("Status:                      ", fitter.status)
a, b = fitter.params

alpha = 0.01
from scipy.stats import chi2
rv = chi2(fitter.dof)
pval = 1-rv.cdf(fitter.chi2_min)
print("If H0 was correct, then") 
print("the probability to find a chi-squared higher than this:  ", pval)
if pval < alpha:
   print("pval=%g. If we set the threshold to alpha=%f, we REJECT H0."%(pval, alpha))
else:
   print("pval=%g. If we set the threshold to alpha=%f, we ACCEPT H0."%(pval, alpha))

N = len(y)
varmod = (y-model(fitter.params,x))**2.0
y_av = y.sum()/N
vardat = (y-y_av)**2.0
vr0 = 100.0*(1-(varmod.sum()/vardat.sum()))
print("Unfiltered sample: variance reduction(%):", vr0)

# Prepare loop where we exclude one point in each run
xf = numpy.zeros(N-1)
yf = numpy.zeros(N-1)
errf = numpy.zeros(N-1)
vr = []

fitter = kmpfit.Fitter(residuals=residuals, data=(xf,yf,errf))
N = len(yf)
for i in range(N):
   xf[:] = numpy.delete(x,i)      # Delete one point
   yf[:] = numpy.delete(y,i)
   errf[:] = numpy.delete(err,i)
   print("\nWe deleted from the sample: (%g,%g)"%(x[i],y[i]))
   fitter.fit(params0=params0)
   print("chi^2, rchi2: ", fitter.chi2_min, fitter.rchi2_min)
   varmod = (yf-model(fitter.params,xf))**2.0
   yf_av = yf.sum()/N
   vardat = (yf-yf_av)**2.0
   vr1 = 100.0*(1-(varmod.sum()/vardat.sum()))
   # A vr of 100% implies that the model is perfect
   # A bad model gives much lower values (sometimes negative)
   print("Variance reduction%:", vr1)
   print("Improvement: %g%%"%(vr1-vr0))
   vr.append([vr1,i])

vr.sort()
print(vr)
i = vr[-1][1]
xf[:] = numpy.delete(x,i)      # Delete one point
yf[:] = numpy.delete(y,i)
errf[:] = numpy.delete(err,i)
fitter.fit(params0=params0)
print("Filtered sample: chi^2, rchi2: ", fitter.chi2_min, fitter.rchi2_min)
pval = 1-rv.cdf(fitter.chi2_min)
print("If H0 was correct, then")
print("the probability to find a chi-squared higher than this:  ", pval)
if pval < alpha:
   print("pval=%g. If we set the threshold to alpha=%f, we REJECT H0."%(pval, alpha))
else:
   print("pval=%g. If we set the threshold to alpha=%f, we ACCEPT H0."%(pval, alpha))

frame.set_ylim(0, 1.1*y.max())
frame.set_xlim(0, 1.1*x.max())
frame.errorbar(x, y, yerr=err, fmt='bo')
frame.plot(x, a+b*x, 'g', label="Fit unfiltered data")
frame.plot(x, model(fitter.params,x), 'm', label="Fit filtered data")
frame.plot((x[i],), (y[i],), marker="o", ms=20, color="r", alpha=0.3)
leg = frame.legend(loc=2)
show()