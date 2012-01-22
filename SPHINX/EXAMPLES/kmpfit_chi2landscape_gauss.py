#!/usr/bin/env python
#------------------------------------------------------------
# Script compares efficiency of automatic derivatives vs
# analytical in mpfit.py
# Vog, 31 okt 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from mpl_toolkits.mplot3d import axes3d
from kapteyn import kmpfit

def my_model(p, x):
   #-----------------------------------------------------------------------
   # This describes the model and its parameters for which we want to find
   # the best fit. 'p' is a sequence of parameters (array/list/tuple).
   #-----------------------------------------------------------------------
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma)) + zerolev )


def my_residuals(p, data):
   #-----------------------------------------------------------------------
   # This function is the function called by the fit routine in kmpfit
   # It returns a weighted residual. De fit routine calculates the
   # square of these values.
   #-----------------------------------------------------------------------
   x, y, err = data
   return (y-my_model(p,x)) / err


def my_derivs(p, data, dflags):
   #-----------------------------------------------------------------------
   # This function is used by the fit routine to find the values for
   # the explicit partial derivatives. Argument 'dflags' is a list
   # with booleans. If an element is True then an explicit partial
   # derivative is required.
   #-----------------------------------------------------------------------
   x, y, err = data
   A, mu, sigma, zerolev = p
   pderiv = numpy.zeros([len(p), len(x)])  # You need to create the required array
   sig2 = sigma*sigma
   sig3 = sig2 * sigma
   xmu  = x-mu
   xmu2 = xmu**2
   expo = numpy.exp(-xmu2/(2.0*sig2))
   fx = A * expo
   for i, flag in enumerate(dflags):
      if flag:
         if i == 0: 
            pderiv[0] = expo
         elif i == 1:
            pderiv[1] = fx * xmu/(sig2)
         elif i == 2:
            pderiv[2] = fx * xmu2/(sig3)
         elif i == 3:
            pderiv[3] = 1.0
   return pderiv/-err
   #return numpy.divide(pderiv, -err)


# Artificial data
N = 100
x = numpy.linspace(-5, 10, N)
truepars = [10.0, 5.0, 2.0, 0.0]
p0 = [9, 4.5, 0.8, 0]
y = my_model(truepars, x) + 1.2*numpy.random.randn(len(x))
err = 0.4*numpy.random.randn(N)

# The fit
fitobj = kmpfit.Fitter(residuals=my_residuals, deriv=my_derivs, data=(x, y, err))
try:
   fitobj.fit(params0=p0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit with explicit partial derivatives ========="
print "Params:        ", fitobj.params
print "Errors from covariance matrix         : ", fitobj.xerror
print "Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr 
print "Chi^2 min:     ", fitobj.chi2_min
print "Reduced Chi^2: ", fitobj.rchi2_min
print "Iterations:    ", fitobj.niter
print "Function ev:   ", fitobj.nfev 
print "Status:        ", fitobj.status
print "Status Message:", fitobj.message
print "Covariance:\n", fitobj.covar 


# We want to plot the chi2 landscape
# for a range of values of mu and sigma.

A = fitobj.params[1] # mu
B = fitobj.params[2] # sigma
nx = 200
ny = 200
Da1 = 15.0; Da2 = 20.0
Dy = 20.0
aa = numpy.linspace(A-Da1,A+Da2,nx)
bb = numpy.linspace(0.5,B+Dy,ny)
Z = numpy.zeros( (ny,nx) )


# Get the Chi^2 landscape.
pars = fitobj.params
i = -1
for a in aa:
   i += 1
   j = -1
   for b in bb:
      j += 1
      pars[1] = a
      pars[2] = b
      Z[j,i] = (my_residuals(pars, (x,y,err))**2).sum()

Z /= 100000.0
XY = numpy.meshgrid(aa, bb)


# Plot the result
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, yerr=err, fmt='go', alpha=0.7, label="Noisy data")
frame.plot(x, my_model(truepars,x), 'r', label="True data")
frame.plot(x, my_model(fitobj.params,x), 'b', lw=2, label="Fit with kmpfit")
frame.set_xlabel("X")
frame.set_ylabel("Measurement data")
frame.set_title("Best fit parameters for Gaussian model with noisy data",
                 fontsize=10)
leg = frame.legend(loc=2)

# Plot chi squared landscape
fig2 = figure(2)
frame = fig2.add_subplot(1,1,1, projection='3d', azim=-31, elev=31)
frame.plot((A,),(B,),(0,), 'or', alpha=0.8)
frame.plot_surface(XY[0], XY[1], Z, color='g', alpha=0.9)
frame.set_xlabel('$X=\\mu$')
frame.set_ylabel('$Y=\\sigma$')
frame.set_zlabel('$Z=\\chi^2_{\\nu}$')
frame.set_zlim3d(Z.min(), Z.max(), alpha=0.5)
frame.set_title("Chi-squared landscape $(\\mu,\\sigma)$ of Gaussian model",
                 fontsize=10)

contlevs = [1.0, 0.1, 0.5, 1.5, 2.0, 5, 10, 15, 20, 100, 200]
fig3 = figure(3)   
frame = fig3.add_subplot(1,1,1)
cs = frame.contour(XY[0], XY[1], Z, contlevs)
zc = cs.collections[0]
zc.set_color('red')
zc.set_linewidth(2)
frame.clabel(cs, contlevs, inline=False, fmt='%1.1f', fontsize=10, color='k')
frame.set_title("Chi-squared contours $(\\mu,\\sigma)$ of Gaussian model",
                 fontsize=10)

show()
