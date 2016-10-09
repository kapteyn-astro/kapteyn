#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to straight line parameters
#          to data with errors in both coordinates
# Vog, 27 Nov, 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal,randint
from kapteyn import kmpfit
from matplotlib.patches import Polygon

def model(p, x):
   # Model: Y = a + b*x + c*x*x
   a,b,c = p
   return a + b*x + c*x*x

def residuals(p, data):
   # Effective variance method
   a, b, c = p
   x, y, ex, ey = data
   w = ey*ey + (b+2*c*x)**2*ex*ex
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d

def residuals2(p, data):
   # Errors in Y only
   a, b, c = p
   x, y, ey = data
   w = ey*ey
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d


# Generate noisy data points
N = 40
a0 = -6; b0 = 1; c0 = 0.5
x = numpy.linspace(-4, 3.0, N)
y = model((a0,b0,c0),x) + normal(0.0, 0.5, N)  # Mean 0, sigma 1
errx = normal(0.3, 0.3, N) 
erry = normal(0.2, 0.4, N) 
print("\nTrue model values [a0,b0,c0]:", [a0,b0,c0])

beta0 = [1,1,1]
#beta0 = [1.8,-0.5,0.1]
print("\nODR:")
print("==========")
from scipy.odr import Data, Model, ODR, RealData, odr_stop
linear = Model(model)
mydata = RealData(x, y, sx=errx, sy=erry)
myodr = ODR(mydata, linear, beta0=beta0, maxit=5000)
#myodr.set_job(2)
myoutput = myodr.run()
print("Fitted parameters:      ", myoutput.beta)
print("Covariance errors:      ", numpy.sqrt(myoutput.cov_beta.diagonal()))
print("Standard errors:        ", myoutput.sd_beta)
print("Minimum chi^2:          ", myoutput.sum_square)
print("Minimum (reduced)chi^2: ", myoutput.res_var)
beta = myoutput.beta


# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry), 
                       xtol=1e-12, gtol=1e-12)
fitobj.fit(params0=beta0)
print("\n\n======== Results kmpfit with effective variance =========")
print("Fitted parameters:      ", fitobj.params)
print("Covariance errors:      ", fitobj.xerror)
print("Standard errors:        ", fitobj.stderr)
print("Chi^2 min:              ", fitobj.chi2_min)
print("Reduced Chi^2:          ", fitobj.rchi2_min)
print("Status Message:", fitobj.message)


# Prepare fit routine
fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, erry))
fitobj2.fit(params0=beta0)
print("\n\n======== Results kmpfit errors in y only =========")
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
frame.plot(x, model(beta,x), '-y', lw=4, label="ODR", alpha=0.6)
frame.plot(x, model(fitobj.params,x), 'c', ls='--', lw=2, label="kmpfit effective variance")
frame.plot(x, model(fitobj2.params,x), 'b',label="kmpfit error in Y only")
frame.plot(x, model((a0,b0,c0),x), 'r', label="True parameters")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("ODR and kmpfit with weighted fit. Model: $y=a+bx+cx^2$")
frame.grid(True)
leg = frame.legend(loc=2)
show()