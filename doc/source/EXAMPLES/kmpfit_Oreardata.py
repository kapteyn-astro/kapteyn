 #!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to straight line parameters
#          to data with errors in both coordinates. 
#          Use data from Orear's article
# Vog, 10 Dec, 2011
#
# The data is from a real physics experiment. Orear,
# Am. J.Phys., Vol.50, No. 10, October 1982, lists values
# for a and b that are not comparable to what we find with 
# kmpfit. In an erratum 
# Am. J.Phys., Vol.52, No. 3, March 1984, he published new
# values that are the same as we find with kmpfit. This
# is after an improvement of the minimalization of the
# objective function where the parameters and the weights 
# are iterated together rather than alternately. 
# The literature values are printed as output of this script.
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit


def model(p, x):
   a, b = p
   return a*x - b/x

def residuals(p, data):
   # Residuals function for data with errors in both coordinates
   a, b = p
   x, y, ex, ey = data
   w = ey*ey + ex*ex*(a+b/x**2)**2
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d

def residuals2(p, data):
   # Residuals function for data with errors in y only
   a, b = p
   x, y, ey = data
   wi = numpy.where(ey==0.0, 0.0, 1.0/ey)
   d = wi*(y-model(p,x))
   return d

def residuals3(p, data):
   # Minimum distance formula with expression for x_model
   a, b = p
   x, y, ex, ey = data
   wx = numpy.where(ex==0.0, 0.0, 1.0/(ex*ex))
   wy = numpy.where(ey==0.0, 0.0, 1.0/(ey*ey))
   df = a + b/(x*x)
   # Calculated the approximate values for the model
   x0 = x + (wy*(y-model(p,x))*df)/(wx+wy*df*df)
   y0 = model(p,x0)   
   D = numpy.sqrt( wx*(x-x0)**2+wy*(y-y0)**2 )
   return D


# Create the data
N = 20
beta0 = [0.1,650000]   # Initial estimates
y = numpy.array([-4.017, -2.742, -1.1478, 1.491, 6.873])
x = numpy.array([22000.0, 22930, 23880, 25130, 26390])
N = len(y)
errx = numpy.array([440.0, 470, 500, 530, 540])
erry = numpy.array([0.5, 0.25, 0.08, 0.09, 1.90])

print("\Literature values:")
print("===================") 
print("Orear's iteration method: a, b, min chi^2:", 1.0163e-3, 5.937e5, 2.187)
print("Orear's exact method:     a, b, min chi^2:", 1.0731e-3, 6.250e5, 2.134)

# Prepare fit routine
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
fitobj.fit(params0=beta0)
print("\n\n======== Results kmpfit: weights for both coordinates =========")
print("Params:                 ", fitobj.params)
print("Covariance errors:      ", fitobj.xerror)
print("Standard errors         ", fitobj.stderr)
print("Chi^2 min:              ", fitobj.chi2_min)
print("Reduced Chi^2:          ", fitobj.rchi2_min)
print("Iterations:             ", fitobj.niter)
print("Status:                 ", fitobj.message)

# Prepare fit routine
fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, erry))
fitobj2.fit(params0=beta0)
print("\n\n======== Results kmpfit errors in Y only =========")
print("Params:                 ", fitobj2.params)
print("Covariance errors:      ", fitobj2.xerror)
print("Standard errors         ", fitobj2.stderr)
print("Chi^2 min:              ", fitobj2.chi2_min)
print("Reduced Chi^2:          ", fitobj2.rchi2_min)

# Prepare fit routine
fitobj3 = kmpfit.Fitter(residuals=residuals3, data=(x, y, errx, erry))
fitobj3.fit(params0=beta0)
print("\n\n======== Results kmpfit with distance formula =========")
print("Params:                 ", fitobj3.params)
print("Covariance errors:      ", fitobj3.xerror)
print("Standard errors         ", fitobj3.stderr)
print("Chi^2 min:              ", fitobj3.chi2_min)
print("Reduced Chi^2:          ", fitobj3.rchi2_min)
print("Iterations:             ", fitobj3.niter)
print("Status:                 ", fitobj3.message)

# Some plotting
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
frame = fig.add_subplot(1,1,1)
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='bo')
frame.plot(x, model(fitobj.params,x), 'c', ls='--', lw=2, label="kmpfit errors in x and y")
frame.plot(x, model(fitobj2.params,x), 'g', label="kmpfit errors in y only")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("$\mathrm{Orear's\  data\  and\  model:\ } y=a*x - b/x$")
leg = frame.legend(loc=2)
show()