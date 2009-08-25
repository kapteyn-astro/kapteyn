#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')

#ch = [10,15,20,25,30,35]
ch = range(5,85,5)
numrows = 4
numcols = 4
count = 1
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
   fitsobject.set_limits()
   # Create an image to be used in Matplotlib
   image = fitsobject.createMPLimage(fig)
   image.set_aspectratio()
   image.add_subplot(numcols, numrows, count); count += 1
   image.imshow()
   image.motion_events()
   image.key_events()
   image.click_events()   

# Draw the graticule lines and plot WCS labels
#grat = wcsgrat.Graticule(fitsimage=fitsobject)
#gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)
#gratplot.add(grat)
#gratplot.plot()

plt.show()
