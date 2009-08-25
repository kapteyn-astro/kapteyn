#!/usr/bin/env python
# Interaction demo with XV map

from kapteyn import maputils, wcsgrat
from matplotlib import pyplot as plt

fitsobject = maputils.FITSimage('rense.fits')
#fitsobject.set_imageaxes(1,3, slicepos=50, spectrans='VOPT-V2W')
fitsobject.set_imageaxes(1,3, slicepos=50)

fig = plt.figure(figsize=(14,10))

image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()

# Draw the graticule lines and plot WCS labels
#grat = wcsgrat.Graticule(fitsimage=fitsobject)
#gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)
#gratplot.add(grat)
#gratplot.plot()

# Position information in figure toolbar
image.motion_events()
image.key_events()

plt.show()

