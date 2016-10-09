#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate that the scaled covariance errors for
# unweighted fits are comparable to errors we find with 
# a bootstrap method.
# Vog, 13 Nov 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal, randint
from kapteyn import kmpfit

def residuals(p, data):
   x, y, err = data
   a, b = p
   model = a + b*x
   return (y-model)/err

N = 100
a0 = 0; b0 = 3
x = numpy.linspace(0.0, 2.0, N)
y = a0 + b0*x + normal(0.0, 1.2, N)  # Mean,sigma,N
err = numpy.ones(N)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])

if (fitobj.status <= 0):
   print('error message =', fitobj.errmsg)
   raise SystemExit

print("\n\n======== Results kmpfit with explicit partial derivatives =========")
print("Params:        ", fitobj.params)
print("Errors from covariance matrix         : ", fitobj.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr)
print("Chi^2 min:     ", fitobj.chi2_min)
print("Reduced Chi^2: ", fitobj.rchi2_min)
print("Iterations:    ", fitobj.niter)
print("Function ev:   ", fitobj.nfev)
print("Status:        ", fitobj.status)


A1, B1 = fitobj.params
xr = x.copy()
yr = y.copy()
ery = err.copy()
fitobj = kmpfit.Fitter(residuals=residuals, data=(xr, yr, ery))
slopes = []
offsets = []
trials = 10000
for i in range(trials):       # Start loop over pseudo sample
   indx = randint(0, N, N)    # Do the resampling using an RNG
   # indx is an array of random indices. Use this array to create a new one.
   xr[:] = x[indx]
   yr[:] = y[indx]
   ery[:] = err[indx]
         
   # Only do a regression if there are at least two different
   # data points in the pseudo sample
   ok = (xr != xr[0]).any()

   if (not ok):
      print("All elements are the same. Invalid sample.")
      print(xr, yr)
   else:
      fitobj.fit(params0=[1,1])
      offs, slope = fitobj.params
      slopes.append(slope)
      offsets.append(offs)

slopes = numpy.array(slopes) - B1
offsets = numpy.array(offsets) - A1
print("Bootstrap errors in A, B:", offsets.std(), slopes.std())


# Plot results
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.plot(x, y, 'bo', label='Observed data')
frame.plot(x, a0+b0*x, 'r', label='True model')
frame.plot(x, A1+B1*x, '--c', alpha=0.5, lw=4, label='kmpfit')
frame.set_xlabel("X"); frame.set_ylabel("Y")
frame.set_title("Least squares fit with errors as weights", fontsize=12)
frame.grid(True)
frame.legend(loc='upper left')
show()
