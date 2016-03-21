#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate simple use of fitter routine
# 
# Vog, 12 Nov 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit


# The model
#==========
def model(p, x):
   a,b = p
   y = a + b*x
   return y


# The residual function
#======================
def residuals(p, data):
   x, y = data                     # 'data' is a tuple given by programmer
   return y - model(p,x)


# Artificial data
#================
N = 50                             # Number of data points
mean = 0.0; sigma = 0.6            # Characteristics of the noise we add
xstart = 2.0; xend = 10.0
x = numpy.linspace(3.0, 10.0, N)
paramsreal = [1.0, 1.0]
noise = numpy.random.normal(mean, sigma, N)
y = model(paramsreal, x) + noise


# Prepare a 'Fitter' object'
#===========================
paramsinitial = (0.0, 0.0)
fitobj = kmpfit.Fitter(residuals=residuals, data=(x,y))

try:
   fitobj.fit(params0=paramsinitial)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "Fit status: ", fitobj.message
print "Best-fit parameters:      ", fitobj.params
print "Covariance errors:        ", fitobj.xerror
print "Standard errors           ", fitobj.stderr
print "Chi^2 min:                ", fitobj.chi2_min
print "Reduced Chi^2:            ", fitobj.rchi2_min
print "Iterations:               ", fitobj.niter
print "Number of function calls: ", fitobj.nfev
print "Number of free pars.:     ", fitobj.nfree
print "Degrees of freedom:       ", fitobj.dof
print "Number of pegged pars.:   ", fitobj.npegged
print "Covariance matrix:\n", fitobj.covar


# Plot the result
#================
rc('font', size=10)
rc('legend', fontsize=8)
fig = figure()
xp = numpy.linspace(xstart-1, xend+1, 200)
frame = fig.add_subplot(1,1,1, aspect=1.0)
frame.plot(x, y, 'ro', label="Data")
frame.plot(xp, model(fitobj.params,xp), 'm', lw=1, label="Fit with kmpfit")
frame.plot(xp, model(paramsreal,xp), 'g', label="The model")
frame.set_xlabel("X")
frame.set_ylabel("Response data")
frame.set_title("Least-squares fit to noisy data using KMPFIT", fontsize=10)
s = "Model: Y = a + b*X    real:(a,b)=(%.2g,%.2g), fit:(a,b)=(%.2g,%.2g)"%\
     (paramsreal[0],paramsreal[1], fitobj.params[0],fitobj.params[1])
frame.text(0.95, 0.02, s, color='k', fontsize=7,
           ha='right', transform=frame.transAxes)
frame.set_xlim(0,12)
frame.set_ylim(0,None)
frame.grid(True)
leg = frame.legend(loc=2)
show()