#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage(promptfie=maputils.getfitsfile)
fitsobject.set_imageaxes(promptfie=maputils.getimageaxes)
fitsobject.set_limits(promptfie=maputils.getbox)
   
# Get connected to Matplotlib
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()
image.motion_events()
image.key_events()
image.click_events()   

# Draw the graticule lines and plot WCS labels
grat = wcsgrat.Graticule(fitsimage=fitsobject)
gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)
gratplot.add(grat)
gratplot.plot()

plt.show()
