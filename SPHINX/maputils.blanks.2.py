#!/usr/bin/env python
# Demonstrate mouse interaction in a panel with subplots.
# Press the right mouse button and move the mouse to 
# change the color limits interactively.
# Press keys *pagedown* or *pageup* to change the
# current color map.

from kapteyn import maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=20)

# Get connected to Matplotlib
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.add_subplot(2,1,1)
image.imshow()
image.colorbar()

fitsobject.set_imageaxes(1,2, slicepos=60) 
image2 = fitsobject.createMPLimage(fig)
image2.add_subplot(2,1,2)
image2.imshow()
image2.colorbar()

# Here the order of connecting is important.
# The motion callbacks must be connected before
# the press callbacks.
plt.connect('motion_notify_event', image.on_move)   
plt.connect('motion_notify_event', image2.on_move)   
plt.connect('key_press_event', image.key_pressed)
plt.connect('key_press_event', image2.key_pressed)

plt.show()
