#!/usr/bin/env python
#------------------------------------------------------------
# Script compares efficiency of automatic derivatives vs
# analytical in mpfit.py
# Vog, 31 okt 2011
#------------------------------------------------------------

import numpy
from matplotlib.pyplot import figure, show, rc
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import axes3d
from kapteyn import kmpfit
from kapteyn.tabarray import tabarray
from math import sin, cos, radians, sqrt, atan

def getestimates( x, y ):
   """
   Method described in http://en.wikipedia.org/wiki/Image_moments
   in section 'Raw moments' and 'central moments'.
   Note that we work with scalars and not with arrays. Therefore
   we use some functions from the math module because the are 
   faster for scalars
   """
   m00 = len(x)
   m10 = numpy.add.reduce(x)
   m01 = numpy.add.reduce(y) 
   m20 = numpy.add.reduce(x*x) 
   m02 = numpy.add.reduce(y*y) 
   m11 = numpy.add.reduce(x*y) 

   Xav = m10/m00
   Yav = m01/m00

   mu20 = m20/m00 - Xav*Xav
   mu02 = m02/m00 - Yav*Yav
   mu11 = m11/m00 - Xav*Yav

   theta = (180.0/numpy.pi) * (0.5 * atan(-2.0*mu11/(mu02-mu20)))
   if (mu20 < mu02):                   # mu20 must be maximum
      (mu20,mu02) = (mu02,mu20)        # Swap these values
      theta += 90.0
 
   d1 = 0.5 * (mu20+mu02)
   d2 = 0.5 * sqrt( 4.0*mu11*mu11 + (mu20-mu02)**2.0 )
   maj = sqrt(d1+d2)
   min = sqrt(d1-d2)
   return (Xav, Yav, maj, min, theta)


def func(p, data):
   """
   Calculate z = (x/maj)**2 + (y/min)**2
   Note that z = 1 is an ellipse
   """
   x0, y0, major, minor, pa = p
   x, y = data
   pa   = radians(pa)   
   sinP = sin(-pa)
   cosP = cos(-pa)    
   xt = x - x0
   yt = y - y0    
   xr = xt * cosP - yt * sinP
   yr = xt * sinP + yt * cosP
   return (xr/major)**2 + (yr/minor)**2

def residuals(p, data):
   """
   Note that the function calculates the height z of the 3d landscape
   of the function z = (x/maj)**2 + (y/min)**2.
   An ellipse is defined where z=1. The residuals we use
   for the least squares fit to minimize, is then 1-z
   """
   x, y = data
   return 1.0 - func(p,data)

x,y = tabarray('ellipse.dat').columns()  # Read from file
x += numpy.random.normal(0.0, 0.05, len(x))
y += numpy.random.normal(0.0, 0.05, len(y))
# This ellipse data was made using a function. The parameters 
# that were used are:
# xc = 5.0; yc = 4.0; major = 10.0; minor = 3.0; ang = 60.0
# Compare these values with the result of the esimate function.
p0 = getestimates(x, y)

fitter = kmpfit.Fitter(residuals=residuals, data=(x,y))
fitter.fit(params0=p0)
print "\n========= Fit results ellipse model =========="
print "Initial params:", fitter.params0
print "Params:        ", fitter.params
print "Iterations:    ", fitter.niter
print "Function ev:   ", fitter.nfev
print "Uncertainties: ", fitter.xerror
print "dof:           ", fitter.dof
print "chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min
print "stderr:        ", fitter.stderr
print "Status:        ", fitter.status
xcf, ycf, majorf, minorf, angf = fitter.params 

# Data for the 3D plot, which shows a projection of the ellipse
major = 5; minor = 2
X = numpy.arange(-6, 6, 0.25)
Y = numpy.arange(-6, 6, 0.25)
X, Y = numpy.meshgrid(X, Y)
Z = numpy.sqrt((X/major)**2 + (Y/minor)**2)

# Plot the projected ellipse from the 3D surface
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1, figsize=(5,10))
frame = fig.add_subplot(2,1,1, projection='3d', azim=20, elev=40)
frame.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                   linewidth=0, antialiased=False, alpha=0.6)
frame.set_xlabel('$X$')
frame.set_ylabel('$Y$')
frame.set_zlabel('$Z$')
frame.set_xlim3d(X.min(), X.max())
frame.set_ylim3d(Y.min(), Y.max())
frame.set_zlim3d(Z.min(), Z.max())
title = "Ellipse with major, minor axes = (%g,%g)"%(major, minor)
frame.set_title(title, fontsize=10)

contlevs = [1]  #[0.5,1,1.5,2,2.5]
cs = frame.contour(X, Y, Z, contlevs, )
zc = cs.collections[0]
zc.set_color('red')
zc.set_linewidth(4)
cset = frame.contour(X, Y, Z, contlevs, zdir='z', offset=0)

# Show the data from file (ellipse.dat)
frame = fig.add_subplot(2,1,2)
frame.plot(x, y, 'ro', label="data from file")
frame.set_xlabel('$X$')
frame.set_ylabel('$Y$')
frame.set_title("Ellipse data and fit")
ellipse = Ellipse(xy=(xcf,ycf), width=majorf*2, height=minorf*2, angle=angf,
                  alpha=0.5, color='y', label="Fit")
frame.add_artist(ellipse)
frame.grid(True)
leg = frame.legend(loc=2)

show()
