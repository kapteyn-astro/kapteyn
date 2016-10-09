#!/usr/bin/env python
#------------------------------------------------------------
# Script demonstrates how to interpolate irregular data so that
# it can be used as input for gauest.py
# Vog, 7 feb 2012
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn.profiles import gauest
#from scipy import interpolate


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


# Artificial data
estimate_zerolev = 10.0
N = 100
x = numpy.random.random(N)*15 - 5  # Random numbers between -5 and +10
truepars1 = [10.0, 5.0, 1.0, 3.0, -1.0, 1.5, 0.0]
y = my_model(truepars1, x, 2) + 0.3*numpy.random.randn(len(x)) + estimate_zerolev

# Analysis of x. function gauest() requires data points yi sampled on 
# a regular grid (i.e. sorted and at equidistant x values)
# It sets a flag that indicates whether an array must be sorted and
# a flag is set when the sorted array x does not represent a regular
# grid. Then interpolation is required. 
# These flags needs to set each time a new array 'x' is detected.

sorted = numpy.all(numpy.diff(x) > 0)
print("Is x sorted?", sorted)
if sorted:
   xs = x
   ys = y
else:
   sortindx = numpy.argsort(x)
   xs = x[sortindx]
   ys = y[sortindx]

stap = xs[1] - xs[0]
equidis = numpy.all((numpy.diff(xs)-stap)< 0.01*stap)
print("Are x values (almost) on regular grid? ", equidis)

if not equidis:
  sizex = len(xs)
  delta = (xs[-1]-xs[0])/sizex
  xse = numpy.linspace(xs[0], xs[-1], sizex)
#  tck = interpolate.splrep(xs, ys, s=4)
#  yse = interpolate.splev(xse, tck, der=0)
  yse = numpy.interp(xse, xs, ys)
else:
  xse = xs
  yse = ys

# thresholds for filters
cutamp = 0.1*yse.max()
cutsig = 5.0
rms = 0.3

# Gauest returns a list with up to ncomp tuples 
# of which each tuple contains the amplitude, 
# the centre and the dispersion of the gaussian, in that order.

ncomps = 0
Q = 6
while ncomps != 2 and Q < 15:
   comps = gauest(yse, rms, cutamp, cutsig, q=Q, ncomp=2)
   ncomps = len(comps)
   Q += 1

if ncomps == 2:
   d = xse[1] - xse[0]
   p0 = []
   for comp in comps:          # Scale to real x range
      p0.append(comp[0])
      p0.append(xse[0] + comp[1]*d), 
      p0.append(comp[2]*d)

   print("Gauest with cutamp, cutsig, rms", cutamp, cutsig, rms)
   print("Number of components found:", ncomps)
   print("Value of Q for which 2 comps. were found:", Q-1)
   print("Found ampl, center, dispersion:", p0)
else:
   print("Could not find any components!")

# Add estimate for base level of profile
p0.append(estimate_zerolev)   # Zero level

fig = figure()
rc('legend', fontsize=8)
frame = fig.add_subplot(1,1,1)
frame.plot(x, y, 'or', alpha=0.5, label="Data")
frame.plot(xse, yse, 'g+', label="Interpolated data points")
frame.plot(xse, yse, 'g')
frame.plot(xse, my_model(p0,xse,ncomps), 'b', lw=2, label="Estimate with gauest()")
leg = frame.legend(loc=2)
show()

