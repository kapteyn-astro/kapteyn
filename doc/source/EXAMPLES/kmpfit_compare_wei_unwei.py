#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate quality improvement weighted vs 
#          unweighted  fit for Wolberg data. Wolberg's
#          best fit parameters for a weighted fit is not
#          accurate (a,b) = (1.8926, 4.9982)
#          Improved values are derived from the analytical
#          solutions and kmpfit: (a,b) = (1.8705, 5.0290)
#
# Vog, 01 Jan 2012
#------------------------------------------------------------
import numpy
from numpy.random import normal
from kapteyn import kmpfit

def model(p, x):
   a, b = p
   return a + b*x

def residuals(p, my_arrays):
   x, y, err = my_arrays
   a, b = p
   return (y-model(p,x))/err


x = numpy.array([1.0, 2, 3, 4, 5, 6, 7])
y = numpy.array([6.9, 11.95, 16.8, 22.5, 26.2, 33.5, 41.0])
N = len(y)
err = numpy.ones(N)
errw = numpy.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.5, 4.0])
print("Data x:", x)
print("Data y:", y)
print("Errors:", errw)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, err))
fitobj.fit(params0=[1,1])
print("\n-- Results kmpfit unit weighting wi=1.0:")
print("Best-fit parameters:                              ", fitobj.params)
print("Parameter errors using measurement uncertainties: ", fitobj.xerror)
print("Parameter errors unit-/relative weighted fit:     ", fitobj.stderr)
print("Minimum chi^2:                                    ", fitobj.chi2_min)
print("Covariance matrix:")
print(fitobj.covar)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, 10*err))
fitobj.fit(params0=[1,1])
print("\n-- Results kmpfit with (scaled) equal weights wi=10*1.0:")
print("Best-fit parameters:                              ", fitobj.params)
print("Parameter errors using measurement uncertainties: ", fitobj.xerror)
print("Parameter errors unit-/relative weighted fit:     ", fitobj.stderr)
print("Minimum chi^2:                                    ", fitobj.chi2_min)
print("Covariance matrix:")
print(fitobj.covar)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errw))
fitobj.fit(params0=[1,1])
print("\n-- Results kmpfit with weights:")
print("Best-fit parameters:                              ", fitobj.params)
print("Parameter errors using measurement uncertainties: ", fitobj.xerror)
print("Parameter errors unit-/relative weighted fit:     ", fitobj.stderr)
print("Minimum chi^2:                                    ", fitobj.chi2_min)
print("Minimum reduced chi^2:                            ", fitobj.rchi2_min)
print("Covariance matrix:")
print(fitobj.covar)
rchi2 = fitobj.rchi2_min      # Store for future scaling purposes


errw10 = errw * 10.0
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errw10))
fitobj.fit(params0=[1,1])
print("\n-- Results kmpfit with scaled individual errors (factor=10):")
print("Best-fit parameters:                              ", fitobj.params)
print("Parameter errors using measurement uncertainties: ", fitobj.xerror)
print("Parameter errors unit-/relative weighted fit:     ", fitobj.stderr)
print("Minimum chi^2:                                    ", fitobj.chi2_min)
print("Minimum reduced chi^2:                            ", fitobj.rchi2_min)
print("Covariance matrix:")
print(fitobj.covar)

scaled_errw = errw * numpy.sqrt(rchi2)
print("""\n\nNew array with measurement errors, scaled with factor %g to give 
a reduced chi-squared of 1.0:"""%rchi2) 
print(scaled_errw)

fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y,  scaled_errw))
fitobj.fit(params0=[1,1])
print("\n-- Results kmpfit with scaled individual errors to force red_chi2=1:")
print("Best-fit parameters:                              ", fitobj.params)
print("Parameter errors using measurement uncertainties: ", fitobj.xerror)
print("Parameter errors unit-/relative weighted fit:     ", fitobj.stderr)
print("Minimum chi^2:                                    ", fitobj.chi2_min)
print("Minimum reduced chi^2:                            ", fitobj.rchi2_min)
print("Covariance matrix:")
print(fitobj.covar)