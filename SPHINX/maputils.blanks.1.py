#!/usr/bin/env python
from kapteyn import maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('renseblank.fits')
fitsobject.set_imageaxes(1,2)   

# Get connected to Matplotlib
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
print "MINMAX", image.datmin, image.datmax
image.add_subplot(1,1,1)
image.imshow()
image.colorbar()
image.motion_events()
image.key_events()

plt.show()
