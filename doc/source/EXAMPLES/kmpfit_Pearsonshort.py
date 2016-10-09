#!/usr/bin/env python

import numpy
from kapteyn import kmpfit

def model(p, x):
   # Model: y = a + numpy.tan(theta)*x
   a, theta = p
   return a + numpy.tan(theta)*x

def residuals(p, data):
   # Residuals function for data with errors in y only
   a, b = p
   x, y = data
   d = (y-model(p,x))
   return d

# Pearsons data
x = numpy.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
y = numpy.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
N = len(x)

fitobj2 = kmpfit.Fitter(residuals=residuals, data=(x, y))
fitobj2.fit(params0=[5,0])
a1, b1 = fitobj2.params[0], numpy.tan(fitobj2.params[1])
fitobj3 = kmpfit.Fitter(residuals=residuals, data=(y, x))
fitobj3.fit(params0=(0,5))
a2, b2  = fitobj3.params[0], numpy.tan(fitobj3.params[1])
tan2theta = 2*b1*b2/(b2-b1)
twotheta = numpy.arctan(tan2theta)
best_slope = numpy.tan(0.5*twotheta)
best_offs = y.mean() - best_slope*x.mean()
print("Best fit parameters: a=%.10f  b=%.10f"%(best_offs,best_slope))