#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program finds best-fit pararameters of a model
#          a*sin(bx+c) with data with errors in both variables
#          x and y. It uses the effective variance method for
#          kmpfit and the results are compared with SciPy's
#          ODR routine.
#          It can be used to demonstrate the sensitivity of 
#          the fit process to initial estimates by varying
#          values for beta0
# Vog, 09 Dec, 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal
from kapteyn import kmpfit

def model(p, x):
   # Model: Y = a*sin(b*x+c)
   a,b,c = p
   return a * numpy.sin(b*x+c)

def residuals(p, data):
   # Effective variance method
   a, b, c = p
   x, y, ex, ey = data
   e2 = ey*ey + (a*b*numpy.cos(b*x+c))**2*ex*ex
   w = numpy.sqrt(numpy.where(e2==0.0, 0.0, 1.0/(e2)))
   d = w*(y-model(p,x))
   return d

def residuals2(p, data):
   # Merit function for data with errors Y only
   a, b, c = p
   x, y, ey = data
   w = numpy.where(ey==0.0, 0.0, 1.0/(ey))
   d = w*(y-model(p,x))
   return d


# Generate noisy data points
N = 30
a0 = 2; b0 = 1; c0 = 1
x = numpy.linspace(-3, 7.0, N)
y = model((a0,b0,c0),x) + normal(0.0, 0.3, N)
errx = normal(0.1, 0.2, N) 
erry = normal(0.1, 0.3, N) 


# It is important to start with realistic initial estimates
beta0 = [1.8,0.9,0.9]
print("\nODR:")
print("==========")
from scipy.odr import Data, Model, ODR, RealData, odr_stop
linear = Model(model)
mydata = RealData(x, y, sx=errx, sy=erry)
myodr = ODR(mydata, linear, beta0=beta0, maxit=5000)
myoutput = myodr.run()
print("Fitted parameters:      ", myoutput.beta)
print("Covariance errors:      ", numpy.sqrt(myoutput.cov_beta.diagonal()))
print("Standard errors:        ", myoutput.sd_beta)
print("Minimum chi^2:          ", myoutput.sum_square)
print("Minimum (reduced)chi^2: ", myoutput.res_var)
beta = myoutput.beta


# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
fitobj.fit(params0=beta0)
print("\n\n======== Results kmpfit with effective variance =========")
print("Fitted parameters:      ", fitobj.params)
print("Covariance errors:      ", fitobj.xerror)
print("Standard errors:        ", fitobj.stderr)
print("Chi^2 min:              ", fitobj.chi2_min)
print("Reduced Chi^2:          ", fitobj.rchi2_min)
print("Status Message:", fitobj.message)


# Compare to a fit with weights for y only
fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, erry))
fitobj2.fit(params0=beta0)
print("\n\n======== Results kmpfit errors in Y only =========")
print("Fitted parameters:      ", fitobj2.params)
print("Covariance errors:      ", fitobj2.xerror)
print("Standard errors:        ", fitobj2.stderr)
print("Chi^2 min:              ", fitobj2.chi2_min)
print("Reduced Chi^2:          ", fitobj2.rchi2_min)
print("Status Message:", fitobj2.message)


# Some plotting
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
frame = fig.add_subplot(1,1,1, aspect=1, adjustable='datalim')
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='bo')
# Plot first fit
frame.plot(x, model(beta,x), '-y', lw=4, label="SciPy's ODR", alpha=0.6)
frame.plot(x, model(fitobj.params,x), 'c', ls='--', lw=2, label="kmpfit (errors in X & Y")
frame.plot(x, model(fitobj2.params,x), 'm', ls='--', lw=2, label="kmpfit (errors in Y only)")
frame.plot(x, model((a0,b0,c0),x), 'r', label="Model with true parameters")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("ODR and kmpfit with weighted fit. Model: $y=a\,\sin(bx+c)$")
frame.grid(True)
leg = frame.legend(loc=2)
show()