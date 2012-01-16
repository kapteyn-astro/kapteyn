#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate that the scaled covariance errors for
# weighted fits are comparable (or not) to errors we find with 
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

N = 200
a0 = 0; b0 = 3
x = numpy.linspace(-5, 5.0, N)
y = a0 + b0*x + normal(0.0, 0.8, N)  # Mean,sigma,N
err = normal(0.0, 0.8, N)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])

if (fitobj.status <= 0):
   print 'error message =', fitobj.errmsg
   raise SystemExit

print "\n\n======== Results kmpfit weighted fit ========="
print "Params:        ", fitobj.params
print "Errors from covariance matrix         : ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr
print "Chi^2 min:     ", fitobj.chi2_min
print "Reduced Chi^2: ", fitobj.rchi2_min
print "Iterations:    ", fitobj.niter
print "Function ev:   ", fitobj.nfev
print "Status:        ", fitobj.status
print "Covariance matrix: ", fitobj.covar

A1, B1 = fitobj.params

# Plot results
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, yerr=err, fmt='bo', label='Observed data')
frame.plot(x, a0+b0*x, 'r', label='True model')
frame.plot(x, A1+B1*x, '--c', alpha=0.5, lw=4, label='kmpfit')
frame.set_xlabel("X"); frame.set_ylabel("Y")
frame.set_title("Bootstrap with weighted and unweighted fits", fontsize=8)
frame.grid(True)
frame.legend(loc='upper left')


xr = x.copy()
yr = y.copy()
err *= numpy.sqrt(fitobj.rchi2_min)
scaled_err = err.copy() 


fitobj2 = kmpfit.Fitter(residuals=residuals, data=(xr, yr, scaled_err))
fitobj2.fit(params0=[1,1])

if (fitobj2.status <= 0):
   print 'error message =', fitobj2.errmsg
   raise SystemExit

print "\n\n======== Results kmpfit weighted fit with reduced chi^2 forced to 1.0 ========="
print "Params:        ", fitobj2.params
print "Errors from covariance matrix         : ", fitobj2.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj2.stderr
print "Chi^2 min:     ", fitobj2.chi2_min
print "Reduced Chi^2: ", fitobj2.rchi2_min
print "Iterations:    ", fitobj2.niter
print "Function ev:   ", fitobj2.nfev
print "Status:        ", fitobj2.status


col = ['y','g']
for k in [0,1]:
   if k == 1:
      fitobj2 = kmpfit.Fitter(residuals=residuals, data=(xr, yr, 1.0))
   slopes = []
   offsets = []
   trials = 2000
   for i in range(trials):       # Start loop over pseudo sample
      indx = randint(0, N, N)    # Do the resampling using an RNG
      # indx is an array of random indices. Use this array to create a new one.
      xr[:] = x[indx]
      yr[:] = y[indx]
      scaled_err[:] = err[indx]
            
      # Only do a regression if there are at least two different
      # data points in the pseudo sample
      ok = (xr != xr[0]).any()
   
      if (not ok):
         print "All elements are the same. Invalid sample."
         print xr, yr
      else:
         fitobj2.fit(params0=[1,1])
         offs, slope = fitobj2.params
         #print fitobj2.rchi2_min
         slopes.append(slope)
         offsets.append(offs)
         frame.plot(x, offs+slope*x, col[k], alpha=0.1)

   slopes = numpy.array(slopes) - B1
   offsets = numpy.array(offsets) - A1
   print "Bootstrap errors in A, B:", offsets.std(), slopes.std()

show()