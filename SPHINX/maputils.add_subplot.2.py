#!/usr/bin/env python
from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=30)

# Get connected to Matplotlib
fig = plt.figure(figsize=(7,5))

# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.set_aspectratio()               # Always before creating a subplot
image.add_subplot(2,1,1, axisbg='y')

plt.show()

