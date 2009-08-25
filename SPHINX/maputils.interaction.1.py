#!/usr/bin/env python
# Mouse and key interaction demo with minimal imports

from kapteyn import maputils, wcsgrat
from matplotlib.pyplot import figure as mpl_figure, show as mpl_show

fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=30)

fig = mpl_figure(figsize=(14,10))

image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()

# Draw the graticule lines and plot WCS labels
grat = wcsgrat.Graticule(fitsimage=fitsobject)
gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)
gratplot.add(grat)
gratplot.plot()

# Position information in figure toolbar
image.motion_events()
image.key_events()

mpl_show()
