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
   # Model function is a gaussian
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) + zerolev )

def residuals(p, data):
   # Return weighted residuals
   x, y, err = data
   return (y-func(p,x)) / err

def cdf(dat1, dat2):
    cdfnew = []
    n = len(dat2)
    for yy in dat1:
       fr = len(dat2[dat2 <= yy])/float(n)
       cdfnew.append(fr)
    return numpy.asarray(cdfnew)

# Create data    
N = 30
x = numpy.linspace(-5,10,N)
# Parameters: A, mu, sigma, zerolev
truepars = [10.0, 4.0, 2.5, 0.0]
p0 = [10, 4.5, 0.8, 0]
y = func(truepars, x) + numpy.random.normal(0, 0.8, N)
err = numpy.random.normal(0.0, 0.6, N)

# Do the fit
fitter = kmpfit.Fitter(residuals=residuals, data=(x,y,err))
#fitter.parinfo = [{}, {}, {}, {'fixed':True}]  # Take zero level fixed in fit
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

# Create the ECDF and the model CDF
data1 = numpy.asarray(y)
data1 = numpy.sort(data1)
n1 = len(data1)
cdf_data_high = numpy.arange(1.0, N+1.0)/N
cdf_data_lo   = numpy.arange(0.0, 1.0*N)/N
X = numpy.linspace(min(x), max(x), 300)
data_hypo = func(fitter.params, X)
data_hypo.sort()
cdf_hypo = cdf(data1, data_hypo)

# Find biggest distance between cumulative functions
DD = cdf_data_high - cdf_hypo
Dip = DD.argmax()
Dplus = DD[Dip]
DD = cdf_hypo - cdf_data_lo
Dim = DD.argmax()
Dmin = DD[Dim]
if Dplus >= Dmin:
   Di = Dip
else:
   Di = Dim
Dmax = max(Dplus, Dmin)

header = "\n============= Kolmogorov-Smirnov statistics ============="
print header
from scipy.stats import kstwobign
# Routine based on NR. It's a good approximation for N>4
dist = kstwobign()
alphas = [0.2, 0.1, 0.05, 0.025, 0.01]
# Select significance level for rejecting null hypothesis
alpha = 0.05 # 10%
for a in alphas:
   Dn_crit = dist.ppf(1-a)/numpy.sqrt(N)
   print "Critical value of D at alpha=%.3f(two sided):  %g"%(a, Dn_crit)
print "Selected alpha:               :", alpha
print "This implies that in %d%% of the time we reject H0 while it is true."%(alpha*100)
Dn_crit = dist.ppf(1-alpha)/numpy.sqrt(N)
print "\nCritical value of D from kstwobign() at alpha=%g(two sided):  %g"%(alpha, Dn_crit)
print "Dplus, Dmin                   :", Dplus, Dmin
print "Dmax                          :", Dmax
print "Confidence level kstwobign()  :", dist.sf(Dmax*numpy.sqrt(N))
if Dmax > Dn_crit:
   print "We REJECT the hypothesis that the data is consistent with the model"
else:
   print "We ACCEPT the hypothesis that the data is consistent with the model"

# Compare to SciPy's kstest()
from scipy.stats import kstest
d, prob = kstest(data1, cdf, args=(data_hypo,))
print "\nCompare to SciPy's kstest():"
print "Dmax             :", d
print "Confidence level :", prob
print "This probability from SciPy's kstest() is equal"
print "or close to the value from kstwobign()"
print "="*len(header)+"\n\n"

# Plot the result
rc('legend', fontsize=8)
fig = figure(figsize=(7.2,10))
fig.subplots_adjust(hspace=0.5)

frame1 = fig.add_subplot(3,1,1)
xd = numpy.linspace(x.min(), x.max(), 200)
frame1.errorbar(x, y, yerr=err,  fmt='bo', label="data")
label = "Model"
frame1.plot(xd, func(fitter.params,xd), 'g', label=label)
frame1.set_xlabel("$x$")
frame1.set_ylabel("$y$")
vals = (fitter.chi2_min, fitter.rchi2_min, fitter.dof)
title = r"$\mathrm{Fit\ with\ } \chi^2=%g \mathrm{\ and\ } \chi^2_{\nu}=%g \,(\nu=%d)$"%vals
frame1.set_title(title, y=1.05)
frame1.grid(True)
leg = frame1.legend(loc=2)

frame2 = fig.add_subplot(3,1,2)
xdist = numpy.linspace(0.1*Dmax, max(Dmax*2, Dn_crit+Dmax/10.), 100)
ydist = dist.pdf(xdist*numpy.sqrt(N))
ydist /= max(ydist)
delta = (xdist.max()-xdist.min())/40.0
frame2.plot(xdist, ydist, 'm', label="Two sided KS-test (kstwobign), N=%d"%(N))
frame2.set_xlabel("$D$")
frame2.set_ylabel("$\mathrm{Probability\ density}$")
frame2.set_title("$\mathrm{Probability\ density\ function\ for\, } N=%d$"%N)
frame2.plot((Dmax,Dmax),(0,ydist.max()), 'g', label="D max (fit) = %g"%Dmax)
frame2.plot((Dn_crit,Dn_crit),(0,ydist.max()), 'r', label="D max threshold = %g"%Dn_crit)

bbox_props = dict(boxstyle="larrow,pad=0.2", fc="cyan", ec="b", lw=2)
t = frame2.text(Dn_crit-delta, ydist.max()/2.0, "Accept", ha="right", va="center", size=12,
               bbox=bbox_props)
bbox_props = dict(boxstyle="rarrow,pad=0.2", fc="red", ec="b", lw=2)
t = frame2.text(Dn_crit+delta, ydist.max()/2.0, "Reject", ha="left", va="center", size=12,
               bbox=bbox_props)
frame2.grid(True)
frame2.set_ylim(-0.01,1.01)
leg = frame2.legend(loc=1)

ydist = dist.cdf(xdist*numpy.sqrt(N))
frame3 = fig.add_subplot(3,1,3)
frame3.plot(xdist, ydist, 'm', label="Cdf of kstwobign(), N=%d"%(N))
frame3.set_xlabel("$D$")
frame3.set_ylabel("$\mathrm{Cumulative\ probability}$")
frame3.plot((Dmax,Dmax),(0,ydist.max()), 'g', label="D max (fit) = %g"%Dmax)
frame3.plot((Dn_crit,Dn_crit),(0,ydist.max()), 'r', label="D max threshold = %g"%Dn_crit)
frame3.plot((0,Dn_crit),(1-alpha,1-alpha), 'r--', label="Threshold for alpha = %g (=1-%g)"%(alpha,1-alpha))
frame3.set_title("$\mathrm{Cumulative\ distribution\ function\ for\, } N=%d$"%N)
frame3.grid(True)
frame3.set_ylim(-0.01,1.01)
leg = frame3.legend(loc=4)

# Make a plot of the ECDF and the CDF of the model
fig = figure(2)
frame = fig.add_subplot(1,1,1)
frame.plot(data1, cdf_data_high, 'mx', ms=6, label="Upper values step function")
frame.plot(data1, cdf_data_lo, 'm*', ms=6, label="Lower values step function")
frame.step(data1, cdf_data_high, where='pre', color='#1122ff', label='data lower step')
frame.step(data1, cdf_data_high, where='post',color='#aabb77',  ls='--', label='data upper step')
frame.step(data1, cdf_data_lo, color='#33bb7c', ls='--')
frame.set_title("Empirical and model CDF's")

cdf_hypo_all = cdf(data_hypo, data_hypo)
frame.plot(data_hypo, cdf_hypo_all, 'go', alpha=0.1, label="Hypothesized distribution $F_0(x)$")

cdf2 = cdf(data1, data_hypo)
frame.plot(data1, cdf2, 'mo', ms=2, label="$F_0(x)$ where $x$ are the Y values of sample")
frame.set_xlabel("Y values of sample")
frame.set_ylabel("Cumulative probability")

label="Dmax=%4f (Dcrit=%4f)"%(Dmax, Dn_crit)
if Dplus >= Dmin:
   cdfdat = cdf_data_high
else:
   cdfdat = cdf_data_lo
frame.plot((data1[Di],data1[Di]), (cdfdat[Di],cdf_hypo[Di]), 'r', label=label)
frame.set_ylim(-0.05,1.05)
frame.grid(True)
leg = frame.legend(loc=2)
show()