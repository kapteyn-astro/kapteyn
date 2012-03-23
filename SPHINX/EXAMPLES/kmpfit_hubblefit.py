#!/usr/bin/env python
# Demonstrate regression through origin
# 02-03-2012

import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal
from scipy.special import erfc
from kapteyn import kmpfit

def model(b, x):
   return b*x

def residuals(p, data):       # Needed for kmpfit
   x, y, err = data           # arrays is a tuple given by programmer
   b = p                    
   return (y - model(b,x))/err

def chauvenet(x, y, mean=None, stdv=None):
   #-----------------------------------------------------------
   # Input:  NumPy arrays x, y that represent measured data
   #         A single value of a mean can be entered or a 
   #         sequence of means with the same length as 
   #         the arrays x and y. In the latter case, the 
   #         mean could be a model with best-fit parameters.
   # Output: It returns a boolean array as filter.
   #         The False values correspond to the array elements
   #         that should be excluded
   # 
   # First standardize the distances to the mean value
   # d = abs(y-mean)/stdv so that this distance is in terms
   # of the standard deviation.  
   # Then the  CDF of the normal distr. is given by 
   # phi = 1/2+1/2*erf(d/sqrt(2))
   # Note that we want the CDF from -inf to -d and from d to +inf.
   # Note also erf(-d) = -erf(d).
   # Then the threshold probability = 1-erf(d/sqrt(2))
   # Note, the complementary error function erfc(d) = 1-erf(d)
   # So the threshold probability pt = erfc(d/sqrt(2))
   # If d becomes bigger, this probability becomes smaller.
   # If this probability (to obtain a deviation from the mean)
   # becomes smaller than 1/(2N) than we reject the data point
   # as valid. In this function we return an array with booleans 
   # to set the accepted values.
   # 
   # use of filter:
   # xf = x[filter]; yf = y[filter]
   # xr = x[~filter]; yr = y[~filter]
   # xf, yf are cleaned versions of x and y and with the valid entries
   # xr, yr are the rejected values from array x and y
   #-----------------------------------------------------------
   if mean is None:
      mean = y.mean()           # Mean of incoming array y
   if stdv is None:
      stdv = y.std()            # Its standard deviation
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
d0 = numpy.array([42, 6.75, 25, 33.8, 9.36, 21.8, 5.58, 8.52, 15.1])
v0 = numpy.array([1294, 462, 2562, 2130, 750, 2228, 598, 224, 971])
N0 = len(d0)
# Here one can experiment with errors on the measured values
err0 = numpy.zeros(N0) + normal(190.0, 1.0, N0)  # Mean 200, sigma 1

# Hubble space telescope 2009: H0 = 74.2 +- 3.6 (km/s)/Mpc.
H0_last = 74.2

# Filter possible outliers based on Chauvenet's criterion
# Use the current literature value of the Hubble constant
# to create a line which is used as a base to calculate
# deviations. These offsets are distributed normally and we can
# apply Chauvenet's criterion to filter outliers. 

print "\nPre filtering:"
print "================="
paramsinitial = [70.0]
fitobj = kmpfit.Fitter(residuals=residuals, data=(d0,v0,err0))
fitobj.fit(params0=paramsinitial)
H0_fit0 = fitobj.params[0]
H0_fit0_delta = fitobj.stderr
print "H0 with unfiltered data: ", H0_fit0

# If you want to know which data would have been excluded if
# you had a perfect fit with H0 = Ho_last, then use: mean = H0_last*d
mean = H0_fit0*d0
# Use expression below for unit weighting. Otherwise use
# the errors in the data as standard deviations.
# stdv = numpy.sqrt(((v0-mean)**2).sum()/(N0))
stdv = err0
filter = chauvenet(d0, v0, mean=mean, stdv=stdv)
print "Excluded data based on Chauvenet's criterion:", zip(d0[~filter], v0[~filter])
d = d0[filter]; v = v0[filter]; err = err0[filter]
N = len(d)   # Length could have been changed

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

print "\nFit results filtered data:"
print "============================"
print "Best-fit parameters:        ", fitobj.params
print "Asymptotic error:           ", fitobj.xerror
print "Error assuming red.chi^2=1: ", fitobj.stderr
print "Chi^2 min:                  ", fitobj.chi2_min
print "Reduced Chi^2:              ", fitobj.rchi2_min
print "Iterations:                 ", fitobj.niter
print "Number of function calls:   ", fitobj.nfev
print "Number of free pars.:       ", fitobj.nfree
print "Degrees of freedom:         ", fitobj.dof
print "Number of data points:      ", len(d)
print "Covariance matrix:\n", fitobj.covar


varmod = (v0-model(H0_fit0,d0))**2.0
v0_av = v0.sum()/N0
vardat = (v0-v0_av)**2.0
vr0 = 100.0*(1-(varmod.sum()/vardat.sum()))
print "\nVariance reduction unfiltered data: %.2f%%"%vr0

xf = numpy.zeros(N0-1)
yf = numpy.zeros(N0-1)
errf = numpy.zeros(N0-1) 
vrs = []
fitter = kmpfit.Fitter(residuals=residuals, data=(xf,yf,errf))
header = "%20s %10s %10s %10s"%('Excluded data', 'chi^2', 'red.chi^2', 'VR')
print "\n", header, "\n", "="*len(header)
for i in range(N0):
   xf[:] = numpy.delete(d0,i)      # Delete one point
   yf[:] = numpy.delete(v0,i)
   errf[:] = numpy.delete(err0,i)
   fitter.fit(params0=paramsinitial)
   varmod = (yf-model(fitter.params,xf))**2.0
   yf_av = yf.sum()/N0
   vardat = (yf-yf_av)**2.0
   vr1 = 100.0*(1-(varmod.sum()/vardat.sum()))   
   # A vr of 100% implies that the model is perfect
   # A bad model gives much lower values (sometimes negative)
   t = (d0[i], v0[i], fitter.chi2_min, fitter.rchi2_min, vr1)
   print "(%8.2f, %8.2f) %10.2f %10.2f %10.2f"%t
   vrs.append([vr1,i])

print "="*len(header)+"\n"
vrs.sort()
i = vrs[-1][1]
xf[:] = numpy.delete(d0,i)      # Delete one point
yf[:] = numpy.delete(v0,i)   
errf[:] = numpy.delete(err0,i)
fitter.fit(params0=paramsinitial)
H0_vr = fitter.params[0]
H0_vr_delta = fitter.stderr
print "H0 based on VR filter:", H0_vr

# Plot results
fig = figure()
rc('legend', fontsize=7)
frame = fig.add_subplot(1,1,1)
frame.errorbar(d, v, yerr=err, fmt='bo', label="Filtered data")
frame.plot(d0[~filter], v0[~filter], 'ro', label="Excluded data")
dd = numpy.linspace(0, 1.1*d0.max(), 20)  # Sample the fit lines
label = "Unfiltered: $H_0 = %.1f \pm %.1f (km/s)/Mpc$"%(H0_fit0, H0_fit0_delta)
frame.plot(dd, dd*H0_fit0, '-m', label=label) 
label = "Filtered with Chauv.: $H_0 = %.1f \pm %.1f (km/s)/Mpc$"%(H0_fit, err_fit_scaled)
frame.plot(dd, dd*H0_fit,  '-y', label=label) 
label = "Filter based on VR: $H_0 = %.1f \pm %.1f (km/s)/Mpc$"%(H0_vr, H0_vr_delta)
frame.plot(dd, dd*H0_vr,   '-c', label=label)
label="Literature: $H_0=74.2 \pm 3.6 (km/s)/Mpc$"
frame.plot(dd, dd*H0_last, '-g', lw=2, label=label)
frame.set_xlabel("D (Mpc)")
frame.set_ylabel("V (Km/s)")
frame.set_title("Hubble constant for filtered data")
frame.set_xlim(0, dd.max())
frame.grid(True)
frame.legend(loc='upper left')
show()
