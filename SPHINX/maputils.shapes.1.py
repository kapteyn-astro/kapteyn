#!/usr/bin/env python
from kapteyn import wcsgrat, maputils, ellinteract
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()

movieimages = maputils.ImageContainer()


# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')

#ch = [10,15,20,25,30,35]
#ch = range(15,85)
ch = [20, 30]
count = 0
vmin, vmax = fitsobject.globalminmax()
print "Vmin, Vmax:", vmin, vmax
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
   fitsobject.set_limits()
   # Create an image to be used in Matplotlib
   image = fitsobject.createMPLimage(fig)
   image.set_aspectratio()
   image.add_subplot(2,1,count+1)
   image.imshow(vmin=vmin, vmax=vmax)
   image.motion_events()
   image.key_events()
   movieimages.append(image)
   count += 1
   # Draw the graticule lines and plot WCS labels
   grat = wcsgrat.Graticule(fitsimage=fitsobject)
   gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)
   gratplot.add(grat)
   gratplot.plot()


# movieimages.movie_events()


shapes = ellinteract.Shapecollection(movieimages.mplim, fig, wcs=True)

plt.show()

