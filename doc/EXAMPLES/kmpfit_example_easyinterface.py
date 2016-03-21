#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate simple use of fitter routine
# 
# Vog, 24 Nov 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show
from kapteyn import kmpfit

# The model:
def model(p, x):
   a, b = p
   y = a + b*x
   return y


# Artificial data
N = 50                             # Number of data points
mean = 0.0; sigma = 0.6            # Characteristics of the noise we add
x = numpy.linspace(2, 10, N)
paramsreal = [1.0, 1.0]
noise = numpy.random.normal(mean, sigma, N)
y = model(paramsreal, x) + noise
err = numpy.random.normal(mean, sigma, N)


# Simple interface
p0 = (0,0)
xl = range(10)
yl = [k*0.5 for k in xl]
fitobj = kmpfit.simplefit(model, p0, xl, yl)
print "Best fit parameters:", fitobj.params
print "Parameter errors:  :", fitobj.stderr

fitobj = kmpfit.simplefit(model, p0, x, y, err=err, xtol=1e-8)
print "Best fit parameters:", fitobj.params
print "Parameter errors:  :", fitobj.xerror

fitobj = kmpfit.simplefit(model, p0, x, y, maxiter=100)
print "Best fit parameters:", fitobj.params
print "Parameter errors:  :", fitobj.stderr
