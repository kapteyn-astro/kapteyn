#!/usr/bin/env python
#------------------------------------------------------------
# Script compares efficiency of automatic derivatives vs
# analytical in mpfit.py
# Vog, 31 okt 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit
from matplotlib.patches import Polygon

def confpred_band(x, dfdp, prob, fitobj, f, prediction, abswei=False, err=None):
   #----------------------------------------------------------
   # Return values for a confidence or a prediction band.
   # See documentation for methods confidence_band and 
   # prediction_band
   #----------------------------------------------------------   
   from scipy.stats import t
   # Given the confidence or prediction probability prob = 1-alpha
   # we derive alpha = 1 - prob 
   alpha = 1 - prob
   prb = 1.0 - alpha/2
   tval = t.ppf(prb, fitobj.dof)
   
   C = fitobj.covar
   n = len(fitobj.params)              # Number of parameters from covariance matrix
   p = fitobj.params
   N = len(x)
   if abswei:
      covscale = 1.0  # Do not apply correction with red. chi^2
   else:
      covscale = fitobj.rchi2_min
   df2 = numpy.zeros(N)
   for j in range(n):
      for k in range(n):
         df2 += dfdp[j]*dfdp[k]*C[j,k]
   if prediction:
      df = numpy.sqrt(err*err+covscale*df2)
   else:
      df = numpy.sqrt(covscale*df2)
   y = f(p, x)
   delta = tval * df   
   upperband = y + delta
   lowerband = y - delta 
   return y, upperband, lowerband


def confidence_band(x, dfdp, confprob, fitobj, f, err=None, abswei=False):
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
   #          the 'true' value of f(p,x) falls within these limits is
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
   # error is 0 and there is no contribution to the confidence
   # interval.
   #----------------------------------------------------------   
   return confpred_band(x, dfdp, confprob, fitobj, f, prediction=False, err=err, abswei=abswei)



def prediction_band(x, dfdp, predprob, fitobj, f, err=None, abswei=False):
   #----------------------------------------------------------
   # Given a value for x, calculate the error df in y = model(p,x)
   # This function returns for each x in a NumPy array, the
   # upper and lower value of the prediction interval. 
   # The arrays with limits are returned and can be used to
   # plot confidence bands.  
   # 
   #
   # Input:
   #
   # x        NumPy array with values for which you want
   #          the prediction interval.
   #
   # dfdp     A list with derivatives. There are as many entries in
   #          this list as there are parameters in your model.
   #
   # predprob Prediction probability in percent (e.g. 0.9 or 0.95).
   #          From this number we derive the prediction level 
   #          (e.g. 0.05). The Prediction Band
   #          is a 100*(1-alpha)% band. This implies
   #          that values of one or more future observations from
   #          the same population from which a given data set was sampled,
   #          will fall in this band with a probability of 100*(1-alpha)%
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
   # upperband  The upper prediction limits
   # lowerband  The lower prediction limits   
   #
   # Note:
   #
   # If parameters were fixed in the fit, the corresponding 
   # error is 0 and there is no contribution to the prediction
   # interval.
   #----------------------------------------------------------   
   return confpred_band(x, dfdp, predprob, fitobj, f, 
                        prediction=True, err=err, abswei=abswei)


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
   x, y, err = data    # y is dummy here
   A, mu, sigma, zerolev = p
   pderiv = numpy.zeros([len(p), len(x)])  # You need to create the required array
   sig2 = sigma * sigma
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


# Artificial data
N = 50
x = numpy.linspace(-5, 10, N)
truepars = [10.0, 5.0, 1.0, 0.0]
p0 = [9, 4.5, 0.8, 0]
rms_data = 0.8
rms_err = 0.1
y = my_model(truepars, x) + numpy.random.normal(0.0, rms_data, N)
err = numpy.random.normal(0.6, rms_err, N)
#err = err*0 + 1


# The fit
fitobj = kmpfit.Fitter(residuals=my_residuals, deriv=my_derivs, data=(x, y, err))
try:
   fitobj.fit(params0=p0)
except Exception as mes:
   print("Something wrong with fit: ", mes)
   raise SystemExit

print("\n\n======== Results kmpfit with explicit partial derivatives =========")
print("Params:        ", fitobj.params)
print("Errors from covariance matrix         : ", fitobj.xerror)
print("Uncertainties assuming reduced Chi^2=1: ", fitobj.stderr) 
print("Chi^2 min:     ", fitobj.chi2_min)
print("Reduced Chi^2: ", fitobj.rchi2_min)
print("Iterations:    ", fitobj.niter)
print("Function ev:   ", fitobj.nfev) 
print("Status:        ", fitobj.status)
print("Status Message:", fitobj.message)
print("Covariance:\n", fitobj.covar) 

# Re-use my_derivs() but rescale derivatives back again with -err
dervs = my_derivs(fitobj.params, (x,y,err), (True,True,True,True))*-err

dfdp = [dervs[0], dervs[1], dervs[2], dervs[3]]
confprob = 0.95
ydummy, upperband, lowerband = confidence_band(x, dfdp, confprob, fitobj, my_model)
verts_conf = list(zip(x, lowerband)) + list(zip(x[::-1], upperband[::-1]))

predprob = 0.90
ydummy, upperband, lowerband = prediction_band(x, dfdp, predprob, fitobj, my_model, 
                               err=err, abswei=False)
verts_pred = list(zip(x, lowerband)) + list(zip(x[::-1], upperband[::-1]))


# Plot the result
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure()
frame = fig.add_subplot(1,1,1)
X = numpy.linspace(x.min(), x.max(), 100)
frame.errorbar(x, y, yerr=err, fmt='go', alpha=0.7, label="Noisy data")
frame.plot(X, my_model(truepars,X), 'r', label="True data")
frame.plot(X, my_model(fitobj.params,X), 'b', lw=2, label="Fit with kmpfit")
poly = Polygon(verts_conf, closed=True, fc='g', ec='g', alpha=0.3, 
               label="CI (%g)"%confprob)
frame.add_patch(poly)
poly = Polygon(verts_pred, closed=True, fc='r', ec='r', alpha=0.3, 
               label="PI (%g)"%predprob)
frame.add_patch(poly)
frame.set_xlabel("X")
frame.set_ylabel("Measurement data")
frame.set_title("Confidence- and prediction bands for Gaussian model",
                fontsize=10)
delta = (x.max()-x.min())/10.0
frame.set_xlim(x.min()-delta, x.max()+delta)
frame.grid(True)

# Check prediction intervals
"""
for i in range(500):
   y = my_model(truepars, x) + numpy.random.normal(0.0, rms_data, N)
   err = numpy.random.normal(0.0, rms_err, N)
   #frame.plot(x,y,'o')
   frame.errorbar(x, y, yerr=err, fmt='o')
"""
# A nice background for the entire plot
from matplotlib.cm import copper
frame.imshow([[0, 0],[1,1]], interpolation='bicubic', cmap=copper,
             vmin=-0.5, vmax=0.5,
             extent=(frame.get_xlim()[0], frame.get_xlim()[1], 
                     frame.get_ylim()[0], frame.get_ylim()[1]), 
             alpha=1)

leg = frame.legend(loc=2)
show()
