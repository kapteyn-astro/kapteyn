#!/usr/bin/env python
#------------------------------------------------------------
# Script which demonstrates how to use the Kolmogorov-Smirnov
# goodness of fit test
# 
# Vog, 17 Feb 2012
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit

def func(p, x):
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) + zerolev )

def residuals(p, data):
   x, y, err = data
   return (y-func(p,x)) / err

N = 50
x = numpy.linspace(-5,10,N)
# Parameters: A, mu, sigma, zerolev
truepars = [10.0, 5.0, 1.0, 0.0]
p0 = [10, 4.5, 0.8, 0]
#y = func(truepars, x) - func([2,-2,1,0],x) + numpy.random.normal(0, 0.8, N)
y = func(truepars, x) + numpy.random.normal(0, 0.8, N)
N = len(x)
err = numpy.random.normal(0.0, 0.6, N)

fitter = kmpfit.Fitter(residuals=residuals, data=(x,y,err))
fitter.parinfo = [{}, {}, {}, {'fixed':True}]  # Take zero level fixed in fit
fitter.fit(params0=p0)

if (fitter.status <= 0): 
   print "Status:  ", fitter.status
   print 'error message = ', fitter.errmsg
   raise SystemExit 

print "\n========= Fit results =========="
print "Initial params:", fitter.params0
print "Params:        ", fitter.params
print "Iterations:    ", fitter.niter
print "Function ev:   ", fitter.nfev 
print "Uncertainties: ", fitter.xerror
print "dof:           ", fitter.dof
print "chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min
print "stderr:        ", fitter.stderr   
print "Status:        ", fitter.status

print "\n======== Kolmogorov-Smirnov statistics ========"
from scipy.stats import ksone
rv = ksone(N)

# Create the normalized cumulative distributions
yfit = func(fitter.params,x)
ydatares = y - yfit           # Residuals
yfitsum = yfit.sum()          # sum fit values
ydatasum = y.sum()            # sum data values
N = len(yfit)
yfitcum = numpy.zeros(N)
ydatacum = numpy.zeros(N)

D = numpy.zeros(N)            # create D statistic array for all data points
y_fit_ks=0                    # zero fit integral
y_data_ks=0                   # zero data integral
for i in range(N):
    y_fit_ks += yfit[i]       # fit integral up to point i
    y_data_ks += y[i]         # data integral up to point i
    yfitcum[i] = y_fit_ks/yfitsum     # Store cumulative values for plot
    ydatacum[i] = y_data_ks/ydatasum
    # normalized difference
    D[i] = abs(y_data_ks/ydatasum - y_fit_ks/yfitsum)

# Find the maximum difference between the two distributions
Di = D.argmax()
Dn = D[Di]

print "Integrals Data, Fit: ", ydatasum, yfitsum
print "D max              : ", Dn

# Select significance level for rejecting null hypothesis
alpha = 0.05
Dn_crit = rv.ppf(1-alpha)
print "Critical value of D: ", Dn_crit
if Dn > Dn_crit:
   print "We REJECT the hypothesis that the data is consistent with the model"
else:
   print "We ACCEPT the hypothesis that the data is consistent with the model"


# Plot the result
rc('legend', fontsize=8)
fig = figure(figsize=(7.2,9.5))
fig.subplots_adjust(hspace=0.5)

frame1 = fig.add_subplot(4,1,1)
xd = numpy.linspace(x.min(), x.max(), 200)
frame1.errorbar(x, y, yerr=err,  fmt='bo', label="data")
label = "fit model: $y = A\ \exp\\left( \\frac{-(x-\mu)^2}{\sigma^2}\\right) + 0$"
frame1.plot(xd, func(fitter.params,xd), 'g', label=label)
frame1.set_xlabel("$x$")
frame1.set_ylabel("$y$")
vals = (fitter.chi2_min, fitter.rchi2_min, fitter.dof)
title = r"$\mathrm{Fit\ with\ } \chi^2=%g \mathrm{\ and\ } \chi^2_{\nu}=%g \,(\nu=%d)$"%vals
frame1.set_title(title, y=1.05)
leg = frame1.legend(loc=2)

frame2 = fig.add_subplot(4,1,2)
frame2.plot(x, numpy.array(yfitcum), label="Cumulative plot of model")
frame2.plot(x, ydatacum, '+', label="Cumulative plot of data")
frame2.plot((x[Di],x[Di]), (ydatacum[Di],yfitcum[Di]), 'r', label="Dmax=%g"%Dn)
frame2.set_xlabel("$X$")
frame2.set_ylabel("$\mathrm{Cumulative\ sum}$")
frame2.set_title("$\mathrm{Kolmogorov-Smirnov\ D\ statistic}$")
#frame.set_ylim(-0.05, 1.05)
leg = frame2.legend(loc=2)

frame3 = fig.add_subplot(4,1,3)
xdist = numpy.linspace(0.1*Dn, max(Dn*2, Dn_crit+Dn/10), 100)
ydist = rv.pdf(xdist)
delta = (xdist.max()-xdist.min())/40.0
frame3.plot(xdist, ydist, label="Kolmogorov-Smirnov one-sided test for N=%d"%(N))
frame3.set_xlabel("$D$")
frame3.set_ylabel("$\mathrm{Probability\ density}$")
frame3.set_title("$\mathrm{Probability\ density\ function\ for\, } N=%d$"%N)
frame3.plot((Dn,Dn),(0,ydist.max()), 'g', label="D max (fit) = %g"%Dn)
frame3.plot((Dn_crit,Dn_crit),(0,ydist.max()), 'r', label="D max threshold = %g"%Dn_crit)

bbox_props = dict(boxstyle="larrow,pad=0.2", fc="cyan", ec="b", lw=2)
t = frame3.text(Dn_crit-delta, ydist.max()/2, "Accept", ha="right", va="center", size=12,
               bbox=bbox_props)
bbox_props = dict(boxstyle="rarrow,pad=0.2", fc="red", ec="b", lw=2)
t = frame3.text(Dn_crit+delta, ydist.max()/2, "Reject", ha="left", va="center", size=12,
               bbox=bbox_props)
leg = frame3.legend(loc=1)

ydist = rv.cdf(xdist)
frame4 = fig.add_subplot(4,1,4)
frame4.plot(xdist, ydist, label="N = %d"%(N))
frame4.set_xlabel("$D$")
frame4.set_ylabel("$\mathrm{Cumulative\ probability}$")
frame4.plot((Dn,Dn),(0,ydist.max()), 'g', label="D max (fit) = %g"%Dn)
frame4.plot((Dn_crit,Dn_crit),(0,ydist.max()), 'r', label="D max threshold = %g"%Dn_crit)
frame4.plot((0,Dn_crit),(1-alpha,1-alpha), 'r--', label="threshold for alpha = %g (=1-%g)"%(alpha,1-alpha))
frame4.set_title("$\mathrm{Cumulative\ distribution\ function\ for\, } N=%d$"%N)
leg = frame4.legend(loc=4)

show()