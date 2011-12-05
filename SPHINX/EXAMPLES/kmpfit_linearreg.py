#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate that kmpfit gives the same solution
# as the analytical solution of the linear regression problem 
# (using weights).
# Vog, 02 Dec 2011
#------------------------------------------------------------
import numpy
from numpy.random import normal
from kapteyn import kmpfit

def lingres(xa, ya, err):
   w = numpy.where(err==0.0, 0.0, 1.0/(err*err))
   sum   =  w.sum()
   sumX  = (w*xa).sum()
   sumY  = (w*ya).sum()
   sumX2 = (w*xa*xa).sum()
   sumY2 = (w*ya*ya).sum()
   sumXY = (w*xa*ya).sum()
   delta = sum * sumX2 - sumX * sumX
   a = (sumX2*sumY - sumX*sumXY) / delta
   b = (sumXY*sum - sumX*sumY) / delta
   siga = numpy.sqrt(abs(sumX2/delta))
   sigb = numpy.sqrt(abs(sum/delta))
   return a, b, siga, sigb

def residuals(p, my_arrays):
   x, y, err = my_arrays
   a, b = p
   model = a + b*x
   return (y-model)/err

N = 100
a0 = 2; b0 = 0
x = numpy.linspace(0.0, 2.0, N)
y = a0 + b0*x + normal(0.0, 0.4, N)  # Mean,sigma,N
err = normal(0.0, 0.2, N) 

A0, B0, sigA0, sigB0  = lingres(x, y, err)
print "\n-- Results analytical solution:"
print "Params:                                 ", A0, B0
print "Analytical standard errors:             ", sigA0, sigB0

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])
print "\n-- Results kmpfit:"
print "Params:                                 ", fitobj.params
print "Errors from covariance matrix:          ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err*10))
fitobj.fit(params0=[1,1])
print "\n-- Results kmpfit with scaled individual errors (factor=10):"
print "Params:                                 ", fitobj.params
print "Errors from covariance matrix:          ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr

""" Example output:
-- Results analytical solution:
Params:                                  2.57437636426 -0.529649935093
Analytical standard errors:              0.0022155022284 0.00130816813236

-- Results kmpfit:
Params:                                  [2.574376363832668, -0.52964993481362832]
Errors from covariance matrix:           [ 0.0022155   0.00130817]
Uncertainties assuming reduced Chi^2=1:  [ 0.01794113  0.01059354]

-- Results kmpfit with scaled individual errors (factor=10):
Params:                                  [2.574376364584714, -0.52964993530453175]
Errors from covariance matrix:           [ 0.02215502  0.01308168]
Uncertainties assuming reduced Chi^2=1:  [ 0.01794113  0.01059354]
"""