#!/usr/bin/env python
#------------------------------------------------------------
# Script compares efficiency of automatic derivatives vs
# analytical in mpfit.py
# Vog, 31 okt 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit

def my_model(p, x):
   #-----------------------------------------------------------------------
   # This describes the model and its parameters for which we want to find
   # the best fit. 'p' is a list with parameters.
   #-----------------------------------------------------------------------
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma)) + zerolev )


def my_residuals(p, data):
   #-----------------------------------------------------------------------
   # This function is the function called by the fit routine in kmpfit
   # It returns a weighted residual. De fit routine calculates the
   # square of these values.
   #-----------------------------------------------------------------------
   x, y, err = data
   return (y-my_model(p,x)) / err


def my_derivs(p, dflags, data):
   #-----------------------------------------------------------------------
   # This function is used by the fit routine to find the values for
   # the explicit partial derivatives. Argument 'dflags' is an array 
   # with booleans. If an element is True then an explicit partial
   # derivative is required.
   #-----------------------------------------------------------------------
   A, mu, sigma, zerolev = p
   x, y, err = data
   pderiv = numpy.zeros([len(p), len(x)])  # You need to create the required array
   sig2 = sigma*sigma
   sig3 = sig2 * sigma
   xmu  = x-mu
   xmu2 = xmu**2
   expo = numpy.exp(-xmu2/(2.0*sig2))
   fx = A * expo
   for i, flag in enumerate(dflags):
      if flag:
         if i == 0: 
            pderiv[0] = expo
         elif i == 1:
            pderiv[1] = fx * xmu/(sig2)
         elif i == 2:
            pderiv[2] = fx * xmu2/(sig3)
         elif i == 3:
            pderiv[3] = 1.0
   return numpy.divide(pderiv, -err)


# Artificial data
N = 100
x = numpy.linspace(-5, 10, N)
truepars = [10.0, 5.0, 1.0, 0.0]
p0 = [9, 4.5, 0.8, 0]
y = my_model(truepars, x) + 0.3*numpy.random.randn(len(x))
err = 0.3*numpy.random.randn(N)

# The fit
fa = (x, y, err)
fitobj = kmpfit.Fitter(my_residuals, data=fa, deriv=my_derivs)
fitobj.fit(params0=p0)

if (fitobj.status <= 0): 
   print 'error message =', fitobj.errmsg
   raise SystemExit 

print "\n\n======== Results kmpfit with explicit partial derivatives ========="
print "Params:        ", fitobj.params
print "Errors from covariance matrix         : ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr 
print "Chi^2 min:     ", fitobj.chi2_min
print "Reduced Chi^2: ", fitobj.rchi2_min
print "Iterations:    ", fitobj.niter
print "Function ev:   ", fitobj.nfev 
print "Status:        ", fitobj.status
print "Covariance:\n", fitobj.covar 

# Plot the result
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, yerr=err, fmt='go', alpha=0.7, label="Noisy data")
frame.plot(x, my_model(truepars,x), 'r', label="True data")
frame.plot(x, my_model(fitobj.params,x), 'b', lw=2, label="Fit with mpfit.py")
frame.set_xlabel("X")
frame.set_ylabel("Measurement data")
frame.set_title("Least-squares fit to noisy Gaussian data using KMPFIT")
leg = frame.legend(loc=2)
show()

