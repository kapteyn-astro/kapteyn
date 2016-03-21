#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate that the scaled covariance errors for
# unweighted fits are comparable to errors we find with 
# a bootstrap method.
# Vog, 24 Nov 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal, randint
from kapteyn import kmpfit

# Residual and model in 1 function. Model is straight line
def residuals(p, data):
   x, y, err = data
   a, b = p
   model = a + b*x
   return (y-model)/err

# Artificial data
N = 100
a0 = 0; b0 = 1.2
x = numpy.linspace(0.0, 2.0, N)
y = a0 + b0*x + normal(0.0, 0.4, N)  # Mean,sigma,N
err = numpy.ones(N)                  # All weights equal to 1

# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
try:
   fitobj.fit(params0=[1,1])
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit unweighted fit ========="
print "Params:        ", fitobj.params
print "Errors from covariance matrix         : ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr
print "Chi^2 min:     ", fitobj.chi2_min
print "Reduced Chi^2: ", fitobj.rchi2_min
print "Iterations:    ", fitobj.niter
print "Function ev:   ", fitobj.nfev
print "Status:        ", fitobj.status
print "Status Message:", fitobj.message

# Bootstrap method to find uncertainties
A0, B0 = fitobj.params
xr = x.copy()
yr = y.copy()
ery = err.copy()
fitobj = kmpfit.Fitter(residuals=residuals, data=(xr, yr, ery))
slopes = []
offsets = []
trials = 10000                # Number of synthetic data sets
for i in range(trials):       # Start loop over pseudo sample
   indx = randint(0, N, N)    # Do the resampling using an RNG
   xr[:] = x[indx]
   yr[:] = y[indx]
   ery[:] = err[indx]
         
   # Only do a regression if there are at least two different
   # data points in the pseudo sample
   ok = (xr != xr[0]).any()

   if (not ok):
      print "All elements are the same. Invalid sample."
      print xr, yr
   else:
      fitobj.fit(params0=[1,1])
      offs, slope = fitobj.params
      slopes.append(slope)
      offsets.append(offs)

slopes = numpy.array(slopes) - B0
offsets = numpy.array(offsets) - A0
sigmaA0, sigmaB0 = offsets.std(), slopes.std()
print "Bootstrap errors in A, B:", sigmaA0, sigmaB0

# Plot results
rc('font', size=7)
rc('legend', fontsize=6)
fig = figure(figsize=(7,4))
fig.subplots_adjust(left=0.08, wspace=0.3, right=0.94)
frame = fig.add_subplot(1,3,1, aspect=1.0, adjustable='datalim')
frame.plot(x, y, 'bo', label='Observed data')
frame.plot(x, a0+b0*x, 'r', label='True: Y=%.1f+%.1fX'%(a0,b0))
frame.plot(x, A0+B0*x, '--c', alpha=0.5, lw=4, label='kmpfit')
frame.set_xlabel("X"); frame.set_ylabel("Y")
frame.set_title("Unweighted fit Y=A+B*X")
frame.grid(True)
frame.legend(loc='upper left')

ranges = [(offsets.min(), offsets.max()),(slopes.min(), slopes.max())]
nb = 40                                      # Number of bins in histogram
for i,sigma in enumerate([sigmaA0, sigmaB0]):
   framehist = fig.add_subplot(1, 3, 2+i)
   range = ranges[i]                         # (X) Range in histogram
   framehist.hist(slopes, bins=nb, range=range, fc='g')
   binwidth  = (range[1]-range[0])/nb        # Get width of one bin
   area = trials * binwidth                  # trials is total number of counts
   mu = 0.0
   amplitude = area / (numpy.sqrt(2.0*numpy.pi)*sigma)
   x = numpy.linspace(range[0], range[1], 100)
   y = amplitude * numpy.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))
   framehist.plot(x, y, 'r')
   if i == 0:
      lab = "$A_i-A_0$"
      title = "Distribution synthetic A"
   else:
      lab = "$B_i-B_0$"
      title = "Distribution synthetic B"
   framehist.set_xlabel(lab)
   framehist.set_ylabel("Counts")
   framehist.set_title(title)

show()