#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to straight line parameters
#          to data with errors in both coordinates. Compare 
#          the results with SciPy's ODR routine.
# Vog, 27 Nov, 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal
from kapteyn import kmpfit
from scipy.odr import Data, Model, ODR, RealData, odr_stop

def model(p, x):
   # Model: Y = a + b*x
   a, b = p
   return a + b*x

def residuals(p, data):
   # Merit function for data with errors in both coordinates
   a, b = p
   x, y, ex, ey = data
   w1 = ey*ey + b*b*ex*ex
   w = numpy.sqrt(numpy.where(w1==0.0, 0.0, 1.0/(w1)))
   d = w*(y-model(p,x))
   return d


# Create the data
N = 40
a0 = 2; b0 = 1
x = numpy.linspace(0.0, 7.0, N)
y = model((a0,b0),x) + normal(0.0, 1.0, N)  # Mean 0, sigma 1
errx = normal(0.0, 0.3, N) 
erry = normal(0.0, 0.4, N) 

beta0 = [0,0]
print "\n========== Results SciPy's ODR ============"
linear = Model(model)
mydata = RealData(x, y, sx=errx, sy=erry)
myodr = ODR(mydata, linear, beta0=beta0, maxit=5000)
myoutput = myodr.run()
print "Fitted parameters:      ", myoutput.beta
print "Covariance errors:      ", numpy.sqrt(myoutput.cov_beta.diagonal())
print "Standard errors:        ", myoutput.sd_beta
print "Minimum (reduced)chi^2: ", myoutput.res_var

beta = myoutput.beta

# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
try:
   fitobj.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit errors in both variables ========="
print "Params:                 ", fitobj.params
print "Covariance errors:      ", fitobj.xerror
print "Standard errors         ", fitobj.stderr
print "Chi^2 min:              ", fitobj.chi2_min
print "Reduced Chi^2:          ", fitobj.rchi2_min
print "Status Message:         ", fitobj.message

# Some plotting
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
frame = fig.add_subplot(1,1,1, aspect=1)
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='bo')
# Plot first fit
frame.plot(x, beta[1]*x+beta[0], '-y', lw=4, label="ODR", alpha=0.6)
frame.plot(x, fitobj.params[1]*x+fitobj.params[0], 'c', ls='--', lw=2, label="kmpfit")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("Weights in both coords ($\chi^2_{min}$ ODR and Kmpfit)")
leg = frame.legend(loc=2)
show()