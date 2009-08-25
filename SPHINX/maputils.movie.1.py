#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()

movieimages = maputils.ImageContainer()

# Create a maputils FITS object from a FITS file on disk
#fitsobject = maputils.FITSimage('rense.fits')
fitsobject = maputils.FITSimage('mclean.fits')

#ch = [10,15,20,25,30,35]
ch = range(1,60)
count = 0
#vmin, vmax = fitsobject.globalminmax()
vmin=-0.35; vmax=0.494
print "Vmin, Vmax:", vmin, vmax
for i in ch:
#   fitsobject.set_imageaxes(1,3, slicepos=i)
   fitsobject.set_imageaxes(2,3, slicepos=i)
   fitsobject.set_limits()
   # Create an image to be used in Matplotlib
   image = fitsobject.createMPLimage(fig)
   image.set_aspectratio()
   image.add_subplot(1,1,1)
   image.imshow(vmin=vmin, vmax=vmax)
   movieimages.append(image, visible=(count==0), schedule=(count==0))
   count += 1

movieimages.movie_events()

# Draw the graticule lines and plot WCS labels
grat = wcsgrat.Graticule(fitsimage=fitsobject)
gratplot = wcsgrat.Plotversion('matplotlib', fig, movieimages.mplim[0].frame)
gratplot.add(grat)
gratplot.plot()

plt.show()

