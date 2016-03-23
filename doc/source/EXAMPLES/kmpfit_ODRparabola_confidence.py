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

def confidence_band(x, dfdp, confprob, fitobj, f, abswei=False):
   #----------------------------------------------------------
   # Given a value for x, calculate the error df in y = model(p,x)
   # This function returns for each x in a NumPy array, the
   # upper and lower value of the confidence interval. 
   # The arrays with limits are returned and can be used to
   # plot confidence bands.  
   # 
   #
   # Input:
   #
   # x        NumPy array with values for which you want
   #          the confidence interval.
   #
   # dfdp     A list with derivatives. There are as many entries in
   #          this list as there are parameters in your model.
   #
   # confprob Confidence probability in percent (e.g. 90% or 95%).
   #          From this number we derive the confidence level 
   #          (e.g. 0.05). The Confidence Band
   #          is a 100*(1-alpha)% band. This implies
   #          that for a given value of x the probability that
   #          the 'true' value of f falls within these limits is
   #          100*(1-alpha)%.
   # 
   # fitobj   The Fitter object from a fit with kmpfit
   #
   # f        A function that returns a value y = f(p,x)
   #          p are the best-fit parameters and x is a NumPy array
   #          with values of x for which you want the confidence interval.
   #
   # abswei   Are the weights absolute? For absolute weights we take
   #          unscaled covariance matrix elements in our calculations.
   #          For unit weighting (i.e. unweighted) and relative 
   #          weighting, we scale the covariance matrix elements with 
   #          the value of the reduced chi squared.
   #
   # Returns:
   #
   # y          The model values at x: y = f(p,x)
   # upperband  The upper confidence limits
   # lowerband  The lower confidence limits   
   #
   # Note:
   #
   # If parameters were fixed in the fit, the corresponding 
   # error is 0 and there is no contribution to the condidence
   # interval.
   #----------------------------------------------------------   
   from scipy.stats import t
   # Given the confidence probability confprob = 100(1-alpha)
   # we derive for alpha: alpha = 1 - confprob/100 
   alpha = 1 - confprob/100.0
   prb = 1.0 - alpha/2
   tval = t.ppf(prb, fitobj.dof)
   
   C = fitobj.covar
   n = len(fitobj.params)              # Number of parameters from covariance matrix
   p = fitobj.params
   N = len(x)
   if abswei:
      covscale = 1.0
   else:
      covscale = fitobj.rchi2_min
   df2 = numpy.zeros(N)
   for j in range(n):
      for k in range(n):
         df2 += dfdp[j]*dfdp[k]*C[j,k]
   df = numpy.sqrt(fitobj.rchi2_min*df2)
   y = f(p, x)
   delta = tval * df   
   upperband = y + delta
   lowerband = y - delta 
   return y, upperband, lowerband


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
N = 20
a0 = 8/9.; b0 = -2/9.; c0 = 1/9.
x = numpy.linspace(-3, 8.0, N)
y = model((a0,b0,c0),x) + normal(0.0, 0.5, N)  # Mean 0, sigma 1
errx = normal(0.3, 0.2, N) 
erry = normal(0.3, 0.3, N) 
print("\nTrue model values [a0,b0,c0]:", [a0,b0,c0])

beta0 = [1,-0.2,0.1]
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



dfdp = [1, x, x**2]
confprob = 95.0
ydummy, upperband, lowerband = confidence_band(x, dfdp, confprob, fitobj, model)
verts = list(zip(x, lowerband)) + list(zip(x[::-1], upperband[::-1]))


# Some plotting
rc('font', size=9)
rc('legend', fontsize=7)
fig = figure(1)
frame = fig.add_subplot(1,1,1, aspect=1)
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='ko')
frame.plot(x, model(beta,x), 'y', ls='--', lw=2, label="ODR")
frame.plot(x, model(fitobj.params,x), 'c', ls='--', lw=2, label="kmpfit effective variance")
frame.plot(x, model(fitobj2.params,x), 'b',ls='--', lw=2, label="kmpfit error in Y only")
frame.plot(x, model((a0,b0,c0),x), 'r', lw=2, label="True parameters")
poly = Polygon(verts, closed=True, fc='c', ec='c', alpha=0.3, 
               label="CI (95%) relative weighting in X & Y")
frame.add_patch(poly)
ydummy, upperband, lowerband = confidence_band(x, dfdp, confprob, fitobj2, model)
verts = list(zip(x, lowerband)) + list(zip(x[::-1], upperband[::-1]))
poly = Polygon(verts, closed=True, fc='b', ec='b', alpha=0.3,
               label="CI (95%) relative weighting in Y")
frame.add_patch(poly)

frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("ODR and kmpfit with weighted fit. Model: $y=a+bx+cx^2$")
frame.grid(True)

from matplotlib.cm import copper
frame.imshow([[0, 0],[1,1]], interpolation='bicubic', cmap=copper,
             vmin=-0.5, vmax=0.5,
             extent=(frame.get_xlim()[0], frame.get_xlim()[1], 
                     frame.get_ylim()[0], frame.get_ylim()[1]), 
             alpha=1)

leg = frame.legend(loc=2)
show()