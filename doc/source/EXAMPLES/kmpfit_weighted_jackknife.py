#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate that the scaled covariance errors for
# weighted fits are comparable (or not) to errors we find with 
# a Jackknife method.
# Vog, 09 Feb 2012
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
err = normal(0.0, 2.0, N)

fitobj_uw = kmpfit.Fitter(residuals=residuals, data=(x, y, 1.0))
fitobj_uw.fit(params0=[1,1])

if (fitobj_uw.status <= 0):
   print('error message =', fitobj_uw.errmsg)
   raise SystemExit

print("\n\n======== Results kmpfit UNweighted fit =========")
print("Params:        ", fitobj_uw.params)
print("Errors from covariance matrix         : ", fitobj_uw.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj_uw.stderr)
print("Chi^2 min:     ", fitobj_uw.chi2_min)
print("Reduced Chi^2: ", fitobj_uw.rchi2_min)
print("Iterations:    ", fitobj_uw.niter)
print("Function ev:   ", fitobj_uw.nfev)
print("Status:        ", fitobj_uw.status)


fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])

if (fitobj.status <= 0):
   print('error message =', fitobj.errmsg)
   raise SystemExit

print("\n\n======== Results kmpfit weighted fit =========")
print("Params:        ", fitobj.params)
print("Errors from covariance matrix         : ", fitobj.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr)
print("Chi^2 min:     ", fitobj.chi2_min)
print("Reduced Chi^2: ", fitobj.rchi2_min)
print("Iterations:    ", fitobj.niter)
print("Function ev:   ", fitobj.nfev)
print("Status:        ", fitobj.status)
print("Covariance matrix: ", fitobj.covar)

A1, B1 = fitobj.params
err *= numpy.sqrt(fitobj.rchi2_min)  # Perfect weights -> chi^2_red = 1

# Plot results
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, yerr=err, fmt='bo', label='Observed data')
frame.plot(x, a0+b0*x, 'r', label='True model')
frame.plot(x, A1+B1*x, '--c', alpha=0.5, lw=4, label='kmpfit weighted fit')
frame.set_xlabel("X"); frame.set_ylabel("Y")
frame.set_title("Jackknife with weighted and unweighted fits", fontsize=10)
frame.grid(True)

xr = x.copy()
yr = y.copy()
scaled_err = err.copy() 
#scaled_err *= numpy.sqrt(fitobj.rchi2_min)


fitobj2 = kmpfit.Fitter(residuals=residuals, data=(xr, yr, scaled_err))
fitobj2.fit(params0=[1,1])

if (fitobj2.status <= 0):
   print('error message =', fitobj2.errmsg)
   raise SystemExit

print("\n\n===== Results kmpfit weighted fit with reduced chi^2 forced to 1.0 =====")
print("Params:        ", fitobj2.params)
print("Errors from covariance matrix         : ", fitobj2.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj2.stderr)
print("Chi^2 min:     ", fitobj2.chi2_min)
print("Reduced Chi^2: ", fitobj2.rchi2_min)
print("Iterations:    ", fitobj2.niter)
print("Function ev:   ", fitobj2.nfev)
print("Status:        ", fitobj2.status)


xr2 = numpy.zeros(N-1)
yr2 = numpy.zeros(N-1)
scaled_err2 = numpy.zeros(N-1)
col = ['y','g']
for k in [0,1]:
   if k == 0:
      fitobj2 = kmpfit.Fitter(residuals=residuals, data=(xr2, yr2, scaled_err2))
   else:
      fitobj2 = kmpfit.Fitter(residuals=residuals, data=(xr2, yr2, 1.0))
   slopes = []
   offsets = []
   trials = N
   for i in range(trials):       # Start loop over pseudo sample
      xr2[:] = numpy.delete(x,i) # Delete one point
      yr2[:] = numpy.delete(y,i)
      scaled_err2[:] = numpy.delete(err,i)

      ok = True
      if (not ok):
         print("All elements are the same. Invalid sample.")
         print(xr, yr)
      else:
         fitobj2.fit(params0=[1,1])
         offs, slope = fitobj2.params
         # print offs, slope, fitobj2.rchi2_min
         slopes.append(slope)
         offsets.append(offs)
         if i == 0:
            if k == 0:
               frame.plot(x, offs+slope*x, col[k], alpha=0.5,
                          label='Jackknife with weighted fits')
            else:
               frame.plot(x, offs+slope*x, col[k], alpha=0.5,
                          label='Jackknife with unweighted fits')
         else:
            frame.plot(x, offs+slope*x, col[k], alpha=0.5)
   slopes = numpy.array(slopes)
   offsets = numpy.array(offsets)
   offdelta = numpy.sqrt(((offsets-offsets.mean())**2).sum() * (N-1)/N)
   slopedelta = numpy.sqrt(((slopes-slopes.mean())**2).sum() * (N-1)/N)
   if k == 0:
      print("Jackknife errors in A, B for procedure with weighted fits:", offdelta, slopedelta)
   else:
      print("Jackknife errors in A, B for procedure with unweighted fits:", offdelta, slopedelta)

frame.legend(loc='upper left')
show()