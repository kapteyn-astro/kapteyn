#!/usr/bin/env python
#------------------------------------------------------------
# Script which demonstrates how to use goodness of fit 
# values
# 
# Vog, 17 Feb 2012
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit
from scipy.special import gammainc, chdtrc
from scipy.optimize import fminbound

def func(p, x):
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) + zerolev )

def residuals(p, data):
   x, y, err = data
   return (y-func(p,x)) / err

N = 50
x = numpy.linspace(-5,10,N)
truepars = [10.0, 5.0, 1.0, 0.0]
p0 = [10, 4.5, 0.8, 0]
y = func(truepars, x) + numpy.random.normal(0, 0.3, N)
print "A max=", y.max()
N = len(x)
err = numpy.random.normal(0.0, 0.8, N)

fitter = kmpfit.Fitter(residuals=residuals, data=(x,y,err))
fitter.parinfo = [{}, {}, {}, {'fixed':True}]  # Take zero level fixed in fit
fitter.fit(params0=p0)

if (fitter.status <= 0): 
   print "Status:  ", fitter.status
   print 'error message = ', fitter.errmsg
   raise SystemExit 

# Rescale the errors to force a reasonable result:
err[:] *= numpy.sqrt(0.9123*fitter.rchi2_min)
fitter.fit()

print "======== Fit results =========="
print "Initial params:", fitter.params0
print "Params:        ", fitter.params
print "Iterations:    ", fitter.niter
print "Function ev:   ", fitter.nfev 
print "Uncertainties: ", fitter.xerror
print "dof:           ", fitter.dof
print "chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min
print "stderr:        ", fitter.stderr   
print "Status:        ", fitter.status

print "\n======== Statistics ========"

from scipy.stats import chi2
rv = chi2(fitter.dof)
print "Three methods to calculate the right tail cumulative probability:"
print "1. with gammainc(dof/2,chi2/2):  ", 1-gammainc(0.5*fitter.dof, 0.5*fitter.chi2_min)
print "2. with scipy's chdtrc(dof,chi2):", chdtrc(fitter.dof,fitter.chi2_min)
print "3. with scipy's chi2.cdf(chi2):  ", 1-rv.cdf(fitter.chi2_min)
print ""


xc = fitter.chi2_min
print "Threshold chi-squared at alpha=0.05: ", rv.ppf(1-0.05)
print "Threshold chi-squared at alpha=0.01: ", rv.ppf(1-0.01)

f = lambda x: -rv.pdf(x)
x_max = fminbound(f,1,200)
print """For %d degrees of freedom, the maximum probability in the distribution is
at chi-squared=%g """%(fitter.dof, x_max)

alpha = 0.05           # Select a p-value
chi2max = max(3*x_max, fitter.chi2_min)
chi2_threshold = rv.ppf(1-alpha)

print "For a p-value alpha=%g, we found a threshold chi-squared of %g"%(alpha, chi2_threshold)
print "The chi-squared of the fit was %g. Therefore: "%fitter.chi2_min 
if fitter.chi2_min <= chi2_threshold:
   print "we do NOT reject the hypothesis that the data is consistent with the model"
else:
   print "we REJECT the hypothesis that the data is consistent with the model"


# Plot the result
rc('legend', fontsize=8)
fig = figure(figsize=(7.2,9.5))
fig.subplots_adjust(hspace=0.5)
frame3 = fig.add_subplot(3,1,3)
xchi = numpy.linspace(0, chi2max, 100)
ychi = rv.pdf(xchi)
delta = (xchi.max()-xchi.min())/40.0
frame3.plot(xchi, ychi, label="Degrees of freedom = %d"%(fitter.dof))
frame3.set_xlabel("$\chi^2$")
frame3.set_ylabel("$\mathrm{Probability}$")
frame3.set_title("$\chi^2 \mathrm{Probability\ density\ function\ for\, } \\nu=%d$"%fitter.dof)
frame3.plot((xc,xc),(0,ychi.max()), 'g', label="chi square (fit) = %g"%fitter.chi2_min)
frame3.plot((chi2_threshold,chi2_threshold),(0,ychi.max()), 'r', label="chi square threshold = %g"%chi2_threshold)

bbox_props = dict(boxstyle="larrow,pad=0.2", fc="cyan", ec="b", lw=2)
t = frame3.text(chi2_threshold-delta, ychi.max()/2, "Accept", ha="right", va="center", size=12,
               bbox=bbox_props)
bbox_props = dict(boxstyle="rarrow,pad=0.2", fc="red", ec="b", lw=2)
t = frame3.text(chi2_threshold+delta, ychi.max()/2, "Reject", ha="left", va="center", size=12,
               bbox=bbox_props)
leg = frame3.legend(loc=1)

ychi = rv.cdf(xchi)
frame2 = fig.add_subplot(3,1,2)
frame2.plot(xchi, ychi, label="Degrees of freedom = %d"%(fitter.dof))
frame2.set_xlabel("$\chi^2$")
frame2.set_ylabel("$\mathrm{Cumulative\ probability}$")
frame2.plot((xc,xc),(0,ychi.max()), 'g', label="chi square (fit) = %g"%fitter.chi2_min)
frame2.plot((chi2_threshold,chi2_threshold),(0,ychi.max()), 'r', label="chi square threshold = %g"%chi2_threshold)
frame2.plot((0,chi2_threshold),(1-alpha,1-alpha), 'r--', label="threshold for alpha = %g (=1-%g)"%(alpha,1-alpha))
frame2.set_title("$\chi^2 \mathrm{Cumulative\ distribution\ function}$")
leg = frame2.legend(loc=4)

frame1 = fig.add_subplot(3,1,1)
xd = numpy.linspace(x.min(), x.max(), 200)
label = "fit model: $y = A\ \exp\\left( \\frac{-(x-\mu)^2}{\sigma^2}\\right) + 0$"
frame1.plot(xd, func(fitter.params,xd), 'g', label=label)
frame1.errorbar(x, y, yerr=err,  fmt='bo', label="data")
frame1.set_xlabel("$x$")
frame1.set_ylabel("$y$")
vals = (fitter.chi2_min, fitter.rchi2_min, fitter.dof)
title = r"$\mathrm{Fit\ with\ } \chi^2=%g \mathrm{\ and\ } \chi^2_{\nu}=%g \,(\nu=%d)$"%vals
frame1.set_title(title, y=1.05)
leg = frame1.legend(loc=2)

show()