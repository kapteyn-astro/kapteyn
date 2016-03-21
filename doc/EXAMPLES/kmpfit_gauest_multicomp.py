#!/usr/bin/env python
#------------------------------------------------------------
# Script compares efficiency of automatic derivatives vs
# analytical in mpfit.py
# Vog, 31 okt 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit
from kapteyn.profiles import gauest


def my_model(p, x, ncomp):
   #-----------------------------------------------------------------------
   # This describes the model and its parameters for which we want to find
   # the best fit. 'p' is a sequence of parameters (array/list/tuple).
   #-----------------------------------------------------------------------
   y = 0.0
   zerolev = p[-1]   # Last element
   for i in range(ncomp):
      A, mu, sigma = p[i*3:(i+1)*3]
      y += A * numpy.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))
   return y + zerolev


def my_residuals(p, data):
   #-----------------------------------------------------------------------
   # This function is the function called by the fit routine in kmpfit
   # It returns a weighted residual. De fit routine calculates the
   # square of these values.
   #-----------------------------------------------------------------------
   x, y, err, ncomp = data
   return (y-my_model(p,x,ncomp)) / err


# Artificial data
N = 100
x = numpy.linspace(-5, 10, N)
truepars1 = [10.0, 5.0, 1.0, 3.0, -1.0, 1.5, 0.0]
#p0 = [9, 4.5, 0.8, 0]
y = my_model(truepars1, x, 2) + 0.3*numpy.random.randn(len(x))
err = 0.3*numpy.random.randn(N)

cutamp = 0.1*y.max()
cutsig = 5.0
rms = 0.3
# We use gauest to get the initial estimates
# Gauest returns a list with up to ncomp tuples of which each tuple contains the amplitude, 
# the centre and the dispersion of the gaussian, in that order.
ncomps = 0
Q = 1
while ncomps != 2 and Q < 8:
   comps = gauest(x, y, rms, cutamp, cutsig, q=Q, ncomp=2)
   ncomps = len(comps)
   Q += 1

if ncomps != 2:
   raise Exception, "Cannot estimate two components"

print "Gauest with cutamp, cutsig, rms", cutamp, cutsig, rms
print "Number of components found:", ncomps
print "Value of Q for which 2 comps. were found:", Q-1

p0 = []
for c in comps:
   p0 += c
p0.append(0.0)   # Zero level
print "Initial estimates p0=", p0

# The fit
fitobj = kmpfit.Fitter(residuals=my_residuals, data=(x, y, err, ncomps))
try:
   fitobj.fit(params0=p0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit with explicit partial derivatives ========="
print "Params:\n", fitobj.params
print "Errors from covariance matrix:\n ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1:\n", fitobj.stderr 
print "Chi^2 min:     ", fitobj.chi2_min
print "Reduced Chi^2: ", fitobj.rchi2_min
print "Iterations:    ", fitobj.niter
print "Function ev:   ", fitobj.nfev 
print "Status:        ", fitobj.status
print "Status Message:", fitobj.message

# Plot the result
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, yerr=err, fmt='go', alpha=0.7, label="Noisy data")
frame.plot(x, my_model(truepars1,x,2), 'r', label="True data")
frame.plot(x, my_model(fitobj.params,x,ncomps), 'b', lw=2, label="Fit with kmpfit")
frame.set_xlabel("X")
frame.set_ylabel("Measurement data")
frame.set_title("Least-squares fit to noisy multi-component Gaussian data",
                fontsize=10)
leg = frame.legend(loc=2)
show()