#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=30) # Define image in cube
fitsobject.set_limits(promptfie=maputils.getbox)

# Get connection to Matplotlib
#fig = plt.figure(figsize=(20/2.54,18/2.54))  # in cm
fig = plt.figure()

# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()
plt.show()
