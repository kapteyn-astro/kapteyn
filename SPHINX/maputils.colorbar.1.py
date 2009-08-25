#!/usr/bin/env python

from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=30)

fig = plt.figure(figsize=(7,5))

image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow(interpolation='gaussian')
image.colorbar(fontsize=8, orientation='horizontal')

plt.show()

