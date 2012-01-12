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
   return a, b, siga, sigb, delta, sum, sumX2, sumX 

def model(p, x):
   a, b = p
   return a + b*x

def residuals(p, my_arrays):
   x, y, err = my_arrays
   a, b = p
   return (y-model(p,x))/err

N = 100
a0 = 0; b0 = 0
#x = numpy.linspace(0.0, 2.0, N)
#y = a0 + b0*x + normal(0.0, 0.4, N)  # Mean,sigma,N
#err = normal(0.0, 0.2, N) 
x = numpy.array([1,2,3,4,5,6,7])
y = numpy.array([6.9,11.95,16.8,22.5,26.2,33.5,41.0])
N = len(y)
err = numpy.ones(N)

A0, B0, sigA0, sigB0, delta, S, Sxx, Sx  = lingres(x, y, err)
chi2 = (((y-model((A0,B0),x))/err)**2).sum()
rchi2 = chi2/(N-2)
da = sigA0*numpy.sqrt(rchi2)
db = sigB0*numpy.sqrt(rchi2)

print "\n-- Results analytical solution:"
print "Best fit parameters:                        ", [A0, B0]
print "Parameter errors weighted fit:              ", [sigA0, sigB0]
print "Parameter errors un-/relative weighted fit: ", [da, db]
print "Minimum chi^2:                              ", chi2
print "Covariance matrix:"
print Sxx/delta, -Sx/delta
print -Sx/delta, S/delta
 

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])
print "\n-- Results kmpfit:"
print "Best-fit parameters:                        ", fitobj.params
print "Parameter errors weighted fit:              ", fitobj.xerror
print "Parameter errors un-/relative weighted fit: ", fitobj.stderr
print "Minimum chi^2:                              ", fitobj.chi2_min
print "Covariance matrix:"
print fitobj.covar

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err*10))
fitobj.fit(params0=[1,1])
print "\n-- Results kmpfit with scaled individual errors (factor=10):"
print "Best-fit parameters:                        ", fitobj.params
print "Parameter errors weighted fit:              ", fitobj.xerror
print "Parameter errors un-/relative weighted fit: ", fitobj.stderr
print "Minimum chi^2:                              ", fitobj.chi2_min
print "Covariance matrix:"
print fitobj.covar