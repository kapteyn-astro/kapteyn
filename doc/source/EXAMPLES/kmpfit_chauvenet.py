#!/usr/bin/env python
# Demonstrate criterion of Chauvenet to exclude poor data

from numpy.random import normal
from scipy.special import erf, erfc
import numpy
from kapteyn import kmpfit
from matplotlib.pyplot import figure, show, rc

def chauvenet(x, y, mean=None, stdv=None):
   #-----------------------------------------------------------
   # Input:  NumPy arrays x, y that represent measured data
   #         A single value of a mean can be entered or a 
   #         sequence of means with the same length as 
   #         the arrays x and y. In the latter case, the 
   #         mean could be a model with best-fit parameters.
   # Output: It returns a boolean array as filter.
   #         The False values correspond to the array elements
   #         that should be excluded
   # 
   # First standardize the distances to the mean value
   # d = abs(y-mean)/stdv so that this distance is in terms
   # of the standard deviation.  
   # Then the  CDF of the normal distr. is given by 
   # phi = 1/2+1/2*erf(d/sqrt(2))
   # Note that we want the CDF from -inf to -d and from d to +inf.
   # Note also erf(-d) = -erf(d).
   # Then the threshold probability = 1-erf(d/sqrt(2))
   # Note, the complementary error function erfc(d) = 1-erf(d)
   # So the threshold probability pt = erfc(d/sqrt(2))
   # If d becomes bigger, this probability becomes smaller.
   # If this probability (to obtain a deviation from the mean)
   # becomes smaller than 1/(2N) than we reject the data point
   # as valid. In this function we return an array with booleans 
   # to set the accepted values.
   # 
   # use of filter:
   # xf = x[filter]; yf = y[filter]
   # xr = x[~filter]; yr = y[~filter]
   # xf, yf are cleaned versions of x and y and with the valid entries
   # xr, yr are the rejected values from array x and y
   #-----------------------------------------------------------
   if mean is None:
      mean = y.mean()           # Mean of incoming array y
   if stdv is None:
      stdv = y.std()            # Its standard deviation
   N = len(y)                   # Lenght of incoming arrays
   criterion = 1.0/(2*N)        # Chauvenet's criterion
   d = abs(y-mean)/stdv         # Distance of a value to mean in stdv's
   d /= 2.0**0.5                # The left and right tail threshold values
   prob = erfc(d)               # Area normal dist.    
   filter = prob >= criterion   # The 'accept' filter array with booleans
   return filter                # Use boolean array outside this function

def residuals(p, data):
   a, b = p
   x, y = data
   return (y-a-b*x)


def prob( Xlo, Xhi ):
   sq2 = numpy.sqrt(2.0)
   # Left tail and right tail area
   return 1.0 - 0.5 * (erf(Xhi/sq2) - erf(Xlo/sq2))

# Artificial data
N = 100
a0 = 2; b0 = 3
x = numpy.linspace(0.0, 2.0, N)
y = a0 + b0*x + normal(0.0, 1.0, N)  # Mean 0, sigma 1

# Add outlier(s)
y[40] = 14       
y[7] = 12
y[-5] = 0.2

# Prepare plot
fig = figure()
rc('legend', fontsize=8)
frame = fig.add_subplot(1,1,1)
frame.plot(x, y, 'go', label="data")
frame.set_xlabel("x")
frame.set_ylabel("y")
frame.set_title("Exclude poor data with criterion of Chauvenet")

fitter = kmpfit.Fitter(residuals=residuals, data=(x,y))
fitter.fit(params0=(a0,b0))

print("======== Fit results all data included ==========")
print("Params:                      ", fitter.params)
print("Uncertainties:               ", fitter.xerror)
print("Errors assuming red.chi^2=1: ", fitter.stderr)
print("Iterations:                  ", fitter.niter) 
print("Function ev:                 ", fitter.nfev)
print("dof:                         ", fitter.dof)
print("chi^2, rchi2:                ", fitter.chi2_min, fitter.rchi2_min)
print("Status:                      ", fitter.status)



#mean = y.mean() # Use this for data with slope 0
a, b = fitter.params
mean = a + b*x
stdv = (y-mean).std()
criterion = 1.0/(2*N)
for xf, yf, m in zip(x, y, mean):
   d = abs(yf-m)/stdv
   P = prob(-d, d)
   if P < criterion:
      print("Data y=%f has distance %fsigma to mean (%f), prob.=%f" %(yf,d,m,P))
      print("Based on Chauvenet's criterion (prob=%g), we reject data point y=%g"\
      %(criterion,yf))
      frame.plot((xf,), (yf,), 'rx', ms=20)

# The function chauvenet() does it the NumPy way
filter = chauvenet(x, y)
xf = x[filter]; yf = y[filter]
xe = x[~filter]; ye = y[~filter]
print("Excluded by function chauvenet() are:")
print(list(zip(xe, ye)))
fitter = kmpfit.Fitter(residuals=residuals, data=(xf,yf))
fitter.fit(params0=(a0,b0))

print("\n======== Fit results with filtered data ==========")
print("Params:                      ", fitter.params)


frame.set_ylim(0, 1.1*y.max())
frame.plot(x, a+b*x, 'g', label="Fit unfilterd data")
a, b = fitter.params
frame.plot(x, a+b*x, 'b', label="Fit filterd data")
leg = frame.legend(loc=2)
show()