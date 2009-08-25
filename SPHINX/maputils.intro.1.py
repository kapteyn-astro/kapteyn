#!/usr/bin/env python
from kapteyn import maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,3, slicepos=51) # Get Position-Velocity image
   
# Get connected to Matplotlib
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.add_subplot(1,1,1)
image.imshow()
   
plt.show()
