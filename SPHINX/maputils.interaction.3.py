#!/usr/bin/env python
# Demonstrate mouse interaction in a panel with subplots.
# Press the right mouse button and move the mouse to 
# change the color limits interactively.
# Press keys *pagedown* or *pageup* to change the
# current color map or 'r' to reset.

from kapteyn import maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=20)

# Get connected to Matplotlib
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image1 = fitsobject.createMPLimage(fig)
image1.add_subplot(2,1,1)
image1.imshow()
image1.colorbar()

fitsobject.set_imageaxes(1,2, slicepos=60) 
image2 = fitsobject.createMPLimage(fig)
image2.add_subplot(2,1,2)
image2.imshow()
image2.colorbar()

# Here the order of connecting is important.
# The motion callbacks must be connected before
# the press callbacks.

# Allow interaction for images, i.e. 
# -position message in toolbar for mouse movements
# -change color limits with mouse movement + right mouse button
# -change color map with keys pageup and pagedown
# -reset colors with key r
# -write position information with left mouse button click

image1.motion_events()
image2.motion_events()
image1.click_events()
image2.click_events()
image1.key_events()
image2.key_events()

plt.show()
