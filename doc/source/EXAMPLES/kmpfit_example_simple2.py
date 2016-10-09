#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Demonstrate simple use of fitter routine
# 
# Vog, 12 Nov 2011
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit

# The model:
def model(p, x):
   a,b = p
   y = a + b*x
   return y

# Define the residual function
#==============================
def residuals(p, d):
   x, y = d
   return y - model(p,x)


# Artificial data
#================
N = 50                             # Number of data points
mean = 0.0; sigma = 0.6            # Characteristics of the noise we add
xstart = 2.0; xend = 10.0
x = numpy.linspace(3.0, 10.0, N)
paramsreal = [1.0, 1.0]
noise = numpy.random.normal(mean, sigma, N)
y = model(paramsreal, x) + noise


# Prepare a 'Fitter' object'
#===========================
arrays = (x, y)
fitobj = kmpfit.Fitter(residuals, data=arrays)

paramsinitial = (0.0, 0.0)
fitobj.fit(params0=paramsinitial)
if (fitobj.status <= 0):
   print('Error message = ', fitobj.errmsg)
else:
   print("Optimal parameters: ", fitobj.params)


# Plot the result
#================
rc('legend', fontsize=8)
fig = figure()
xp = numpy.linspace(xstart-1, xend+1, 200)
frame = fig.add_subplot(1,1,1, aspect=1.0)
frame.plot(x, y, 'ro', label="Data")
frame.plot(xp, model(fitobj.params,xp), 'm', lw=1, label="Fit with kmpfit")
frame.plot(xp, model(paramsreal,xp), 'g', label="The model")
frame.set_xlabel("X")
frame.set_ylabel("Measurement data")
frame.set_title("Least-squares fit to noisy data using KMPFIT")
s = "Model: Y = a + b*X    real (a,b)=(%.2g,%.2g), fit (a,b)=(%.2g,%.2g)"%\
     (paramsreal[0],paramsreal[1], fitobj.params[0],fitobj.params[1])
frame.text(0.95, 0.02, s, color='k', fontsize=8,
           ha='right', transform=frame.transAxes)
frame.set_xlim(0,12)
frame.set_ylim(0,None)
leg = frame.legend(loc=2)
show()
