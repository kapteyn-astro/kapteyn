#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate, using kmpfit, that if you find best-fit
# parameters, the errors derived from the covariance matrix 
# correspond to an increase in chi^2 of 1.
# Vog, 23 Nov 2011
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


# Artificial data
#----------------
N = 100
a0 = 2; b0 = 3
x = numpy.linspace(0.0, 2.0, N)
y = a0 + b0*x + normal(0.0, 0.4, N)  # Mean,sigma,N
derr = normal(0.0, 0.5, N)
err = 0.9+derr

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])

if (fitobj.status <= 0):
   print('error message =', fitobj.errmsg)
   raise SystemExit

print("\n\n======== Results kmpfit for Y = A + B*X =========")
print("Params:        ", fitobj.params)
print("Errors from covariance matrix         : ", fitobj.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr)
print("Chi^2 min:     ", fitobj.chi2_min)

p1, p2 = fitobj.params
e1, e2 = fitobj.xerror
# Next we take one of the parameters to be fixed and change its value
# with the amount of one of the estimated errors (covariance, scaled or bootstrap)
# If we fit again, then, according to Bevington (Data Reduction and Error
# Analysis for the Physical Sciences  Section 11-5), one should expect the
# Chi square value to increase with 1.0

fitobj.parinfo[0] = dict(fixed=True)
fitobj.fit(params0=[p1+e1,1])
print("\nFix first parameter and set its value to fitted value+error")
print("Params:        ", fitobj.params)
print("Chi^2 min:     ", fitobj.chi2_min)
print("Errors from covariance matrix         : ", fitobj.xerror)

fitobj.parinfo = [{'fixed':False}, {'fixed':True}]
fitobj.fit(params0=[1, p2+e2])
print("\nFix second parameter and set its value to fitted value+error")
print("Params:        ", fitobj.params)
print("Chi^2 min:     ", fitobj.chi2_min)
print("Errors from covariance matrix         : ", fitobj.xerror)