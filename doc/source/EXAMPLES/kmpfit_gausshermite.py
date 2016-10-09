#!/usr/bin/env python
#------------------------------------------------------------
# Script which demonstrates how to find the best-fit
# parameters of a Gauss-Hermite line shape model
# 
# Vog, 26 Mar 2012
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from scipy.special import wofz
from scipy.optimize import fsolve
from kapteyn import kmpfit
ln2 = numpy.log(2)
PI = numpy.pi
from math import sqrt

def gausshermiteh3h4(x, A, x0, s, h3, h4):
   #------------------------------------------------------------
   # The Gauss-Hermite function is a superposition of functions of the form
   # F = (x-xc)/s                                            
   # E =  A.Exp[-1/2.F^2] * {1 + h3[c1.F+c3.F^3] + h4[c5+c2.F^2+c4.F^4]} 
   #------------------------------------------------------------
   c0 =     sqrt(6.0)/4.0
   c1 =    -sqrt(3.0)
   c2 =    -sqrt(6.0)
   c3 = 2.0*sqrt(3.0)/3.0
   c4 =     sqrt(6.0)/3.0
   
   F = (x-x0)/s
   E = A*numpy.exp(-0.5*F*F)*( 1.0 + h3*F*(c3*F*F+c1) + h4*(c0+F*F*(c2+c4*F*F)) )
   return E


def hermite2gauss(par, dpar):
   #------------------------------------------------------------
   # Convert Gauss-Hermite parameters to Gauss(like)parameters.                                        
   #                                                             
   # We use the first derivative of the Gauss-Hermite function   
   # to find the maximum, usually around 'x0' which is the center
   # of the (pure) Gaussian part of the function.                          
   # If F = (x-x0)/s then the function for which we want the     
   # the zero's is A0+A1*F+A2*F^2+A3*F^3+A4*F^4+A5*F^5 = 0       
   # c0 = 1/4sqrt(6) c1 = -sqrt(3) c2 = -sqrt(6)                 
   # c3 = 2/3sqrt(3) c4 = 1/3sqrt(6)                             
   #------------------------------------------------------------ 
   sqrt2pi = sqrt(2.0*PI)
   amp, x0, s, h3, h4 = par
   damp, dx0, ds, dh3, dh4 = dpar   # The errors in those parameters
   c0 = sqrt(6.0)/4.0
   c1 = -sqrt(3.0)
   c2 = -sqrt(6.0)
   c3 = 2.0*sqrt(3.0)/3.0
   c4 = sqrt(6.0)/3.0

   A = numpy.zeros(6)
   A[0] = -c1*h3
   A[1] = h4*(c0-2.0*c2) + 1.0
   A[2] = h3*(c1-3.0*c3)
   A[3] = h4*(c2 - 4.0*c4)
   A[4] = c3*h3
   A[5] = c4*h4
   
   # Define the function that represents the derivative of
   # the GH function. You need it to find the position of the maximum.
   fx = lambda x: A[0] + x*(A[1]+x*(A[2]+x*(A[3]+x*(A[4]+x*A[5]))))
   xr = fsolve(fx, 0, full_output=True)
   xm = s*xr[0] + x0
   ampmax = gausshermiteh3h4(xm, amp, x0, s, h3, h4)

   # Get line strength
   f = 1.0 + h4 * sqrt(6.0) / 4.0
   area  = amp * s * f * sqrt2pi
   d_area = sqrt2pi * sqrt(s*s*f*f*damp*damp +\
                           amp*amp*f*f*ds*ds +\
                           3.0*amp*amp*s*s*dh4*dh4/8.0)

   # Get mean
   mean  = x0 + sqrt(3.0)*h3*s
   d_mean = sqrt(dx0*dx0 + 3.0*h3*h3*ds*ds + 3.0*s*s*dh3*dh3)

   # Get dispersion
   f = 1.0 + h4*sqrt(6.0)
   dispersion = abs(s * f)
   d_dispersion = sqrt(f*f*ds*ds + 6.0*s*s*dh4*dh4)

   # Skewness
   f = 4.0 * sqrt(3.0)
   skewness = f * h3
   d_skewness = f * dh3

   # Kurtosis
   f = 8.0 * sqrt(6.0)
   kurtosis = f * h4
   d_kurtosis = f * dh4

   res = dict(xmax=xm, amplitude=ampmax, area=area, mean=mean, dispersion=dispersion,\
              skewness=skewness, kurtosis=kurtosis, d_area=d_area, d_mean=d_mean,\
              d_dispersion=d_dispersion, d_skewness=d_skewness, d_kurtosis=d_kurtosis)
   return res
 

def voigt(x, y):
   # The Voigt function is also the real part of 
   # w(z) = exp(-z^2) erfc(iz), the complex probability function,
   # which is also known as the Faddeeva function. Scipy has 
   # implemented this function under the name wofz()
   z = x + 1j*y
   I = wofz(z).real
   return I

def Voigt(nu, alphaD, alphaL, nu_0, A):
   # The Voigt line shape in terms of its physical parameters
   f = numpy.sqrt(ln2)
   x = (nu-nu_0)/alphaD * f
   y = alphaL/alphaD * f
   V = A*f/(alphaD*numpy.sqrt(numpy.pi)) * voigt(x, y)
   return V

def funcV(p, x):
    # Compose the Voigt line-shape
    alphaD, alphaL, nu_0, I, z0 = p
    return Voigt(x, alphaD, alphaL, nu_0, I) + z0

def funcG(p, x):
   # Model function is a gaussian
   A, mu, sigma, zerolev = p
   return( A * numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) + zerolev )

def funcGH(p, x):
   # Model is a Gauss-Hermite function
   A, xo, s, h3, h4, zerolev= p
   return gausshermiteh3h4(x, A, xo, s, h3, h4) + zerolev

def residualsV(p, data):
   # Return weighted residuals of Voigt
   x, y, err = data
   return (y-funcV(p,x)) / err

def residualsG(p, data):
   # Return weighted residuals of Gauss
   x, y, err = data
   return (y-funcG(p,x)) / err

def residualsGH(p, data):
   # Return weighted residuals of Gauss-Hermite
   x, y, err = data
   return (y-funcGH(p,x)) / err




# Artificial data derive from GH-series
x = numpy.linspace(853, 859, 30)
A1 = -2.18
X1 = 855.54
S1 = 0.55
h31 = 0.17
h41 = 0.0 
z01 = 6.95
y = gausshermiteh3h4(x, A1, X1, S1, h31, h41) + z01
N = len(y)
y += numpy.random.normal(0.0, 0.05, N)   # Add somne noise

err = numpy.ones(N)
A = -2
alphaD = 0.5
alphaL = 0.5
z0 = 6
nu_0 = 855
p0 = [alphaD, alphaL, nu_0, A, z0]

# Do the fit
fitter = kmpfit.Fitter(residuals=residualsV, data=(x,y,err))
fitter.parinfo = [{'limits':(0,None)}, {'limits':(0,None)}, {}, {}, {}]
fitter.fit(params0=p0)

print("\n========= Fit results Voigt profile ==========")
print("Initial params:", fitter.params0)
print("Params:        ", fitter.params)
print("Iterations:    ", fitter.niter)
print("Function ev:   ", fitter.nfev) 
print("Uncertainties: ", fitter.xerror)
print("dof:           ", fitter.dof)
print("chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min)
print("stderr:        ", fitter.stderr)   
print("Status:        ", fitter.status)

alphaD, alphaL, nu_0, I, z0V = fitter.params
c1 = 1.0692
c2 = 0.86639
hwhm = 0.5*(c1*alphaL+numpy.sqrt(c2*alphaL**2+4*alphaD**2))
print("\nFWHM Voigt profile:     ", 2*hwhm)
f = numpy.sqrt(ln2)
Y = alphaL/alphaD * f
amp = I/alphaD*numpy.sqrt(ln2/numpy.pi)*voigt(0,Y)
print("Amplitude Voigt profile:", amp)
print("Area under profile:     ", I)

# Fit the Gaussian model
p0 = [-3, 855, 0.5, 6.3]
fitterG = kmpfit.Fitter(residuals=residualsG, data=(x,y,err))
#fitterG.parinfo = [{}, {}, {}, {}, {}]  # Take zero level fixed in fit
fitterG.fit(params0=p0)
print("\n========= Fit results Gaussian profile ==========")
print("Initial params:", fitterG.params0)
print("Params:        ", fitterG.params)
print("Iterations:    ", fitterG.niter)
print("Function ev:   ", fitterG.nfev) 
print("Uncertainties: ", fitterG.xerror)
print("dof:           ", fitterG.dof)
print("chi^2, rchi2:  ", fitterG.chi2_min, fitterG.rchi2_min)
print("stderr:        ", fitterG.stderr)   
print("Status:        ", fitterG.status)

fwhmG = 2*numpy.sqrt(2*numpy.log(2))*fitterG.params[2]
print("FWHM Gaussian: ", fwhmG)
z0G = fitterG.params0[-1]          # Store background

# Fit the Gauss-Hermite model
# Initial estimates for A, xo, s, h3, h4, z0
p0 = [-3, 855, 0.5, 0, 0, 6.3]
fitterGH = kmpfit.Fitter(residuals=residualsGH, data=(x,y,err))
#fitterGH.parinfo = [{}, {}, {}, {}, {}]  # Take zero level fixed in fit
fitterGH.fit(params0=p0)
print("\n========= Fit results Gaussian profile ==========")
print("Initial params:", fitterGH.params0)
print("Params:        ", fitterGH.params)
print("Iterations:    ", fitterGH.niter)
print("Function ev:   ", fitterGH.nfev) 
print("Uncertainties: ", fitterGH.xerror)
print("dof:           ", fitterGH.dof)
print("chi^2, rchi2:  ", fitterGH.chi2_min, fitterGH.rchi2_min)
print("stderr:        ", fitterGH.stderr)   
print("Status:        ", fitterGH.status)

A, x0, s, h3, h4, z0GH = fitterGH.params
#xm, ampmax, area, mean, dispersion, skewness, kurtosis 
res = hermite2gauss(fitterGH.params[:-1], fitterGH.stderr[:-1])
print("Gauss-Hermite max=%g at x=%g"%(res['amplitude'], res['xmax']))
print("Area      :", res['area'], '+-', res['d_area'])
print("Mean (X0) :", res['mean'], '+-', res['d_mean'])
print("Dispersion:", res['dispersion'], '+-', res['d_dispersion'])
print("Skewness  :", res['skewness'], '+-', res['d_skewness'])
print("Kurtosis  :", res['kurtosis'], '+-', res['d_kurtosis'])

# Plot the result
rc('legend', fontsize=6)
fig = figure()
frame1 = fig.add_subplot(1,1,1)
xd = numpy.linspace(x.min(), x.max(), 500)
frame1.plot(x, y, 'bo', label="data")
label = "Model with Voigt function"
frame1.plot(xd, funcV(fitter.params,xd), 'g', label=label)
label = "Model with Gaussian function"
frame1.plot(xd, funcG(fitterG.params,xd), 'm', ls='--', label=label)
label = "Model with Gauss-Hermite function"
frame1.plot(xd, funcGH(fitterGH.params,xd), 'c', ls='--', label=label)
frame1.plot((nu_0-hwhm,nu_0+hwhm), (z0V+amp/2,z0V+amp/2), 'r', label='fwhm')
frame1.plot(xd, [z0V]*len(xd), "y", label='Background Voigt')
frame1.plot(xd, [z0G]*len(xd), "y", ls="-.", label='Background G')
frame1.plot(xd, [z0GH]*len(xd), "y", ls="--", label='Background G-H')
frame1.set_xlabel("$\\nu$")
frame1.set_ylabel("$\\phi(\\nu)$")
vals = (fitter.chi2_min, fitter.rchi2_min, fitter.dof)
title = "Profile data with Voigt- vs. Gaussian model\n"
t = (res['area'], res['mean'], res['dispersion'], res['skewness'], res['kurtosis'])
title += "GH: $\gamma_{gh}$=%.1f $x_{0_{gh}}$=%.1f $\sigma_{gh}$ = %.2f $\\xi_1$=%.2f  $\\xi_f$=%.2f"%t
frame1.set_title(title, fontsize=9)
frame1.grid(True)
leg = frame1.legend(loc=4)
show()