#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to best fit straight line parameters
#          to data given by Pearson, 1901
# Vog, 12 Dec, 2011
#
# The data for x and y are from Pearson
# Pearson, K. 1901. On lines and planes of closest fit to systems 
# of points in space. Philosophical Magazine 2:559-572
# Copy of this article can be found at:
# stat.smmu.edu.cn/history/pearson1901.pdf
#
# Pearson's best fit through (3.82,3.70) ->
# a=5.784  b=-0.54556
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit

def model(p, x):
   # Model: y = a + numpy.tan(theta)*x
   a, theta = p
   return a + numpy.tan(theta)*x

def residuals(p, data):
   # Residuals function for data with errors in both coordinates
   a, theta = p
   x, y = data
   B = numpy.tan(theta)
   wi = 1/numpy.sqrt(1.0 + B*B)
   d = wi*(y-model(p,x))
   return d

def residuals2(p, data):
   # Residuals function for data with errors in y only
   a, b = p
   x, y = data
   d = (y-model(p,x))
   return d


# Pearsons data
x = numpy.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
y = numpy.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
N = len(x)
beta0 = [5.0, 0.0]         # Initial estimates

# Analytical solutions following Pearson's formulas
print "\nAnalytical solution"
print "==================="
x_av = x.mean()
y_av = y.mean()
sx = (x-x_av)
sy = (y-y_av)
Sx = sx.sum()
Sy = sy.sum()
Sxx = (sx*sx).sum()
Syy = (sy*sy).sum()
Sxy = (sx*sy).sum()
tan2theta = 2*Sxy/(Sxx-Syy)
twotheta = numpy.arctan(tan2theta)
b_pearson = numpy.tan(twotheta/2)
a_pearson = y_av - b_pearson*x_av
print "Best fit parameters: a=%.10f  b=%.10f"%(a_pearson,b_pearson)
rxy = Sxy/numpy.sqrt(Sxx*Syy)
print "Pearson's Corr. coef: ", rxy
tan2theta = 2*rxy*numpy.sqrt(Sxx*Syy)/(Sxx-Syy)
twotheta = numpy.arctan(tan2theta)
print "Pearson's best tan2theta, theta, slope: ", \
      tan2theta, 0.5*twotheta, numpy.tan(0.5*twotheta)
b1 = rxy*numpy.sqrt(Syy)/numpy.sqrt(Sxx)
print "b1 (Y on X), slope: ", b1, b1
b2 = rxy*numpy.sqrt(Sxx)/numpy.sqrt(Syy)
print "b2 (X on Y), slope", b2, 1/b2


# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y))
fitobj.fit(params0=beta0)
print "\n======== Results kmpfit: effective variance ========="
print "Params:                 ", fitobj.params[0], numpy.tan(fitobj.params[1])
print "Covariance errors:      ", fitobj.xerror
print "Standard errors         ", fitobj.stderr
print "Chi^2 min:              ", fitobj.chi2_min
print "Reduced Chi^2:          ", fitobj.rchi2_min

# Prepare fit routine
fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y))
fitobj2.fit(params0=beta0)
print "\n======== Results kmpfit Y on X ========="
print "Params:                 ", fitobj2.params
print "Covariance errors:      ", fitobj2.xerror
print "Standard errors         ", fitobj2.stderr
print "Chi^2 min:              ", fitobj2.chi2_min
print "Reduced Chi^2:          ", fitobj2.rchi2_min
a1, b1 = fitobj2.params[0], numpy.tan(fitobj2.params[1])

fitobj3 = kmpfit.Fitter(residuals=residuals2, data=(y, x))
fitobj3.fit(params0=(0,5))
print "\n======== Results kmpfit X on Y ========="
print "Params:                 ", fitobj3.params
print "Covariance errors:      ", fitobj3.xerror
print "Standard errors         ", fitobj3.stderr
print "Chi^2 min:              ", fitobj3.chi2_min
print "Reduced Chi^2:          ", fitobj3.rchi2_min
a2, b2  = fitobj3.params[0], numpy.tan(fitobj3.params[1])
A2 = -a2/b2; B2 = 1/b2   # Get values for XY plane 

print "\nLeast squares solution"
print "======================"
print "a1, b1 (Y on X)", a1, b1
print "a2, b2 (X on Y)", A2, B2
tan2theta = 2*b1*b2/(b2-b1)
twotheta = numpy.arctan(tan2theta)
best_slope = numpy.tan(0.5*twotheta)
print "Best fit tan2theta, Theta, slope: ", tan2theta, \
       0.5*twotheta, best_slope
best_offs = y_av - best_slope*x_av
print "Best fit parameters: a=%.10f  b=%.10f"%(best_offs,best_slope)

bislope = (b1*B2-1+numpy.sqrt((1+b1*b1)*(1+B2*B2)))/(b1+B2)
abi = y_av - bislope*x_av
print "Bisector through centroid a, b: ",abi, bislope 
bbi = numpy.arctan(bislope)  # Back to angle again
B2_angle = numpy.arctan(B2)  # Back to angle again


# Some plotting
rc('font', size=9)
rc('legend', fontsize=7)
fig = figure(1)
d = (x.max() - x.min())/10
for i in [0,1]:
   if i == 0: 
      X = numpy.linspace(x.min()-d, x.max()+d, 50)
      frame = fig.add_subplot(2,1,i+1, aspect=1, adjustable='datalim')
   else:
      X = numpy.linspace(-0.9, -0.3, 50)
      frame = fig.add_subplot(2,1,i+1, aspect=1)
   frame.plot(x, y, 'oy')
   frame.plot(X, model(fitobj.params,X), 'c', ls='--', lw=4, label="kmpfit effective variance")
   frame.plot(X, model(fitobj2.params,X), 'g', label="kmpfit regression Y on X")
   frame.plot(X, model((A2,B2_angle),X), 'r', label="kmpfit regression X on Y")
   frame.plot(X, model((abi,bbi),X), 'y', label="Bisector")
   frame.plot(X, model((a_pearson,numpy.arctan(b_pearson)),X), 'm', lw=2, label="Pearson's values")
   frame.plot((x_av,),(y_av,), '+k', markersize=14)  # Mark the centroid
   frame.set_ylabel("Y")
   frame.grid(True)
   if i == 1:
      frame.set_xlabel("X")
      frame.set_xlim(-0.9,-0.3)
      frame.set_ylim(6,6.38)
   else:
      frame.set_title("$\mathrm{Pearson^'s\ data\ and\ model:\ } y = a+b*x$")
   leg = frame.legend(loc=1)
show()