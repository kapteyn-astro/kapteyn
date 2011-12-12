#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to straight line parameters
#          to data with errors in both coordinates. Compare 
#          the results with SciPy's ODR routine.
# Vog, 27 Nov, 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from numpy.random import normal
from kapteyn import kmpfit
from scipy.odr import Data, Model, ODR, RealData, odr_stop

def model(p, x):
   # Model: Y = a + b*x
   a, b = p
   return a + b*x

def residuals(p, data):
   # Residuals function for data with errors in both coordinates
   a, b = p
   x, y, ex, ey = data
   w = ey*ey + b*b*ex*ex
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d

def residuals3(p, data):
   # Merit function for data with errors in both coordinates
   a, b = p
   x, y, ey = data
   w = ey*ey
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d


def residuals2(p, data):
   a, b = p
   x, y, ex, ey = data
   xd = -(a*b-y*b-x)/(1+b*b)
   yd = a+b*xd
   wx = 1/(ex*ex)
   wy = 1/(ey*ey)
   wi = wx*wy/(b*b*wy+wx)
   D = numpy.sqrt( wi*((x-xd)**2+(y-yd)**2) )
   return D

def residuals4(p, data):
   a, b = p
   x, y, ex, ey = data
   xd = -(a*b-y*b-x)/(1+b*b)
   yd = a+b*xd
   wi = 1/(ey*ey+b*b*ex*ex)
   D = numpy.sqrt( wi*((x-xd)**2+(y-yd)**2) )
   return D


def residuals5(p, data):
   a, b = p
   x, y, ex, ey = data
   wx = 1/(ex*ex)
   wy = 1/(ey*ey)
   xd = (wx*x+wy*(y*b-a*b))/(wx+b*b*wy)
   yd = a+b*xd   
   D = numpy.sqrt( wx*(x-xd)**2+wy*(y-yd)**2 )
   return D



# Create the data
N = 20
a0 = 2; b0 = 1.6
x = numpy.linspace(0.0, 12.0, N)
y = model((a0,b0),x) + normal(0.0, 1.0, N)  # Mean 0, sigma 1
errx = normal(0.0, 0.3, N) 
erry = normal(0.0, 0.4, N) 
#errx = numpy.ones(N)
#erry = numpy.ones(N)

beta0 = [0,0]
print "\n========== Results SciPy's ODR ============"
linear = Model(model)
mydata = RealData(x, y, sx=errx, sy=erry)
myodr = ODR(mydata, linear, beta0=beta0, maxit=5000)
myoutput = myodr.run()
print "Fitted parameters:      ", myoutput.beta
print "Covariance errors:      ", numpy.sqrt(myoutput.cov_beta.diagonal())
print "Standard errors:        ", myoutput.sd_beta
print "Minimum (reduced)chi^2: ", myoutput.res_var

beta = myoutput.beta

# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
try:
   fitobj.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit: w1 = ey*ey + b*b*ex*ex ========="
print "Params:                 ", fitobj.params
print "Covariance errors:      ", fitobj.xerror
print "Standard errors         ", fitobj.stderr
print "Chi^2 min:              ", fitobj.chi2_min
print "Reduced Chi^2:          ", fitobj.rchi2_min


# Prepare fit routine
fitobj3 = kmpfit.Fitter(residuals=residuals3, data=(x, y, erry))
try:
   fitobj3.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit errors in Y only ========="
print "Params:                 ", fitobj3.params
print "Covariance errors:      ", fitobj3.xerror
print "Standard errors         ", fitobj3.stderr
print "Chi^2 min:              ", fitobj3.chi2_min
print "Reduced Chi^2:          ", fitobj3.rchi2_min



fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, errx, erry))
try:
   fitobj2.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit: wi = wx*wy/(b*b*wy+wx)  ========="
print "Params:                 ", fitobj2.params
print "Covariance errors:      ", fitobj2.xerror
print "Standard errors         ", fitobj2.stderr
print "Chi^2 min:              ", fitobj2.chi2_min
print "Reduced Chi^2:          ", fitobj2.rchi2_min


fitobj4 = kmpfit.Fitter(residuals=residuals4, data=(x, y, errx, erry))
try:
   fitobj4.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit: wi = 1/(ey*ey+b*b*ex*ex)========="
print "Params:                 ", fitobj4.params
print "Covariance errors:      ", fitobj4.xerror
print "Standard errors         ", fitobj4.stderr
print "Chi^2 min:              ", fitobj4.chi2_min
print "Reduced Chi^2:          ", fitobj4.rchi2_min

fitobj5 = kmpfit.Fitter(residuals=residuals5, data=(x, y, errx, erry))
try:
   fitobj5.fit(params0=beta0)
except Exception, mes:
   print "Something wrong with fit: ", mes
   raise SystemExit

print "\n\n======== Results kmpfit: r = ex*ex/(ey*ey), xd = (x-a*r+y*b*r)/(1+r) ========="
print "Params:                 ", fitobj5.params
print "Covariance errors:      ", fitobj5.xerror
print "Standard errors         ", fitobj5.stderr
print "Chi^2 min:              ", fitobj5.chi2_min
print "Reduced Chi^2:          ", fitobj5.rchi2_min



# Some plotting
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
frame = fig.add_subplot(1,1,1, aspect=1)
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='bo')
# Plot first fit
frame.plot(x, beta[1]*x+beta[0], '-y', lw=4, label="ODR", alpha=0.6)
frame.plot(x, fitobj.params[1]*x+fitobj.params[0], 'c', ls='--', lw=2, label="kmpfit")
frame.plot(x, fitobj2.params[1]*x+fitobj2.params[0], 'g', label="kmpfit min. distance")
frame.plot(x, fitobj3.params[1]*x+fitobj3.params[0], 'b', label="kmpfit errors in Y only")
frame.plot(x, fitobj5.params[1]*x+fitobj5.params[0], '#ffaa00', label="kmpfit correct")
frame.plot(x, b0*x+a0, '#ab12cc', label="True")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("Weights in both coords ($\chi^2_{min}$ ODR and Kmpfit)")


t = "\nTHE WILLAMSON APPROACH"
print t, "\n", "="*len(t)
# Step 1: Get a and b for a, b with standard weighted least squares calculation

def lingres(xa, ya, w):
   # Return a, b for the relation y = a + b*x
   # given data in xa, ya and weights in w
   sum   =  w.sum()
   sumX  = (w*xa).sum()
   sumY  = (w*ya).sum()
   sumX2 = (w*xa*xa).sum()
   sumY2 = (w*ya*ya).sum()
   sumXY = (w*xa*ya).sum()
   delta = sum * sumX2 - sumX * sumX 
   a = (sumX2*sumY - sumX*sumXY) / delta
   b = (sumXY*sum - sumX*sumY) / delta
   return a, b

w = numpy.where(erry==0.0, 0.0, 1.0/(erry*erry))
a,b = lingres(x, y, w)
a_y = a; b_y = b
print "Williamson initial Parameters: ", a,b

ui = errx**2
vi = erry**2

n = 0 
cont = True
while cont:
   # Step 2: Use this slope to find weighting for each point
   wi = (vi+b*b*ui)**-1

   # Step 3: Calcu;ate weighted avarages
   w_sum = wi.sum()
   x_av = (wi*x).sum() / w_sum
   x_diff = x - x_av
   y_av = (wi*y).sum() / w_sum
   y_diff = y - y_av

   # Step 4: Calculate the 'improvement' vector zi
   zi = wi*(vi*x_diff + b*ui*y_diff)
   b_new = (wi*zi*y_diff).sum()/ (wi*zi*x_diff).sum()
   cont = abs(b-b_new) > 1e-12 and n < 100
   n += 1
   b = b_new

# Step 5: Repeat steps 2-4 until convergence

# Step 6: Calculate 'a' using the averages of a and y
a_new = y_av - b_new*x_av 
print "Improved a: ", a_new

# Step 7: The variances

wi = (vi+b_new*b_new*ui)**-1
w_sum = wi.sum()

z_av = (wi*zi).sum() / w_sum
zi2 = zi - z_av
Q =1.0/(wi*(x_diff*y_diff/b_new + 4*zi2*(zi-x_diff))).sum()
sigb2 = Q*Q * (wi*wi*(x_diff**2*vi+y_diff**2*ui)).sum()
siga2 = 1.0/w_sum + 2*(x_av+2*z_av)*z_av*Q + (x_av+2*z_av)**2*sigb2
siga = numpy.sqrt(siga2)
sigb = numpy.sqrt(sigb2)

print "Williamson Fitted A, B: ", a_new, b_new
print "Errors: ", siga, sigb
print


a, b = myoutput.beta
D = residuals2((a,b),(x,y,errx,erry))
print "(Minimum at kmpfit ODR, b Sum van D:\n ", a, b, (D**2).sum()

a, b = fitobj.params
D = residuals2((a,b),(x,y,errx,erry))
print "(Minimum at kmpfit effective variance, b Sum van D:\n ", a, b, (D**2).sum()

a, b = fitobj2.params
D = residuals2((a,b),(x,y,errx,erry))
print "(Minimum at a, b Sum van D:\n ", a, b, (D**2).sum()
xd = -(a*b-y*b-x)/(1+b*b)
yd = a+b*xd
for X1,Y1,X,Y in zip(xd,yd,x,y):
   frame.plot((X,X1),(Y,Y1), 'r')

amin = a; bmin = b
Dmin = (D**2).sum()
for b in numpy.linspace(fitobj.params[1]-0.1, fitobj.params[1]+0.1, 20):
   for a in numpy.linspace(fitobj.params[0]-0.1, fitobj.params[0]+0.1, 20):
      D = residuals2((a,b),(x,y,errx,erry))
      Dsum = ((D)**2).sum()
      if Dsum < Dmin:
         Dmin = Dsum
         amin = a; bmin = b 
      #for X1,Y1,X,Y in zip(xd,yd,x,y):
      #   frame.plot((X,X1),(Y,Y1), 'r')

print "Minimum a, b, Sum van D: ", amin, bmin, Dmin
#frame.plot(x, bmin*x+amin, 'm', label="Locally optimized")
leg = frame.legend(loc=2)


show()