#!/usr/bin/env python
# Demonstrate regression through origin
# 02-03-2012

import numpy
from matplotlib.pyplot import figure, show
from numpy.random import normal
from scipy.special import erfc
from kapteyn import kmpfit


def residuals(p, data):       # Needed for kmpfit
   x, y, err = data           # arrays is a tuple given by programmer
   b = p                    
   return (y - b*x)/err

def chauvenet(x, y, mean=None):
   # Return filter of valid array elements
   if mean is None:
      mean = y.mean()           # Mean of incoming array y
   stdv = y.std()               # Its standard deviation
   N = len(y)                   # Lenght of incoming arrays
   criterion = 1.0/(2*N)        # Chauvenet's criterion
   d = abs(y-mean)/stdv         # Distance of a value to mean in stdv's
   d /= 2.0**0.5                # The left and right tail threshold values
   prob = erfc(d)               # Area normal dist.    
   filter = prob >= criterion   # The 'accept' filter array with booleans
   return filter                # Use boolean array outside this function

def h02age(h0):
   # Convert Hubble value to age
   eenMpc = 3.09E19
   agesec = eenMpc / h0
   ageyear = agesec / (3600*24*365)
   agebil = ageyear / 1e9
   return agebil

def lingres_origin(xa, ya, err):
   # Apply regression through origin
   N = len(xa)
   w = numpy.where(err==0.0, 0.0, 1.0/(err*err))
   sumX2 = (w*xa*xa).sum()
   sumXY = (w*xa*ya).sum()
   sum1divX = (1/(w*xa)).sum()   
   b = sumXY/sumX2
   sigma_b = 1.0/sumX2
   chi2 = (w*(ya-b*xa)**2).sum()
   red_chi2 = chi2 / (N-1)
   sigma_b_scaled = red_chi2 / sumX2
   return b, numpy.sqrt(sigma_b), numpy.sqrt(sigma_b_scaled)


do_chauvenet = True     # Filter outliers
# Data from student lab observations
d = numpy.array([42, 6.75, 25, 33.8, 9.36, 21.8, 5.58, 8.52, 15.1])
v = numpy.array([1294, 462, 2562, 2130, 750, 2228, 598, 224, 971])
N = len(d)

# Hubble space telescope 2009: H0 = 74.2 +- 3.6 (km/s)/Mpc.
H0_last = 74.2

# Filter possible outliers based on Chauvenet's criterion
# Use the current literature value of the Hubble constant
# to create a line which is used as a base to calculate
# deviations. These offsets are distributed normally and we can
# apply Chauvenet's criterion to filter outliers. 

filter = chauvenet(d, v, mean=H0_last*d)
print "\nExcluded data based on Chauvenet's criterion:", zip(d[~filter], v[~filter])
d = d[filter]; v = v[filter]
N = len(d)   # Length could have been changed

# Here one can experiment with errors on the measured values
err = numpy.zeros(N) + normal(200.0, 1.0, N)  # Mean 200, sigma 1

H0_fit, err_fit, err_fit_scaled = lingres_origin(d, v, err)
print "\nResults analytical method:"
print "============================"
print "Best fit H0:                           ", H0_fit
print "Asymptotic error:                      ", err_fit
print "Standard error assuming red.chi^2=1:   ", err_fit_scaled

print "\nModel parameters straight line analytical method: "
print "V = %f(+-%f)*D" % (H0_fit, err_fit_scaled)

x1 = h02age(H0_fit)
x2 = h02age(H0_fit+err_fit)
print "Age from fitted H0=%.1f (+- %.1f): %.1f (+- %.1f billion year)" %\
      (H0_fit, err_fit, x1, abs(x2-x1))
print "Age from literature H0=%.1f: %.1f (billion year)" %\
      (H0_last, h02age(H0_last))


paramsinitial = [70.0]
fitobj = kmpfit.Fitter(residuals=residuals, data=(d,v,err))
fitobj.fit(params0=paramsinitial)

print "\nFit status kmpfit:"
print "===================="
print "Best-fit parameters:        ", fitobj.params
print "Asymptotic error:           ", fitobj.xerror
print "Error assuming red.chi^2=1: ", fitobj.stderr
print "Chi^2 min:                  ", fitobj.chi2_min
print "Reduced Chi^2:              ", fitobj.rchi2_min
print "Iterations:                 ", fitobj.niter
print "Number of function calls:   ", fitobj.nfev
print "Number of free pars.:       ", fitobj.nfree
print "Degrees of freedom:         ", fitobj.dof
print "Covariance matrix:\n", fitobj.covar

# Plot results
fig = figure()
frame = fig.add_subplot(1,1,1)
frame.errorbar(d, v, yerr=err, fmt='bo', label="data")
frame.plot(d, d*H0_fit, '-r', label="Fit") 
frame.plot(d, d*H0_last, '-g', lw=2, label="Literature")
frame.set_xlabel("D (Mpc)")
frame.set_ylabel("V (Km/s)")
frame.set_title("Slope = H0 = %.1f (%.1f billion year)" % (H0_fit, h02age(H0_fit)))
frame.set_xlim(0,1.1*d.max())
frame.grid(True)
frame.legend(loc='upper left')
show()
