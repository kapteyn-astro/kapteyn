#!/usr/bin/env python

from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('rense.fits')
fitsobject.set_imageaxes(1,2, slicepos=30)
fitsobject.set_limits((20,60), (40,80))

fig = plt.figure(figsize=(7,5))

image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()
image.contour(levels=(0.02,0.025,0.06))
image.clabel(0.02, inline=1, colors='r', fontsize=14, fmt="%.3f")
image.set_contattr(0.02, linewidth=4, color='r', linestyle='dashed')
image.set_labelattr(0.025, backgroundcolor="yellow",
            text="Contour 0.025", rotation=0, color="magenta", clip_on=True)

plt.show()

