from kapteyn import wcsgrat
from matplotlib import pylab as plt
import pyfits

hdulist = pyfits.open('mclean.fits')
header = hdulist[0].header

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

xlim = (35,45)
grat = wcsgrat.Graticule(header, axnum=(3,2), spectrans='WAVE-???', pxlim=xlim)
grat.setp_tick(plotaxis=1, fun=lambda x: x*100, fmt="%.3f")
grat.setp_plotaxis(1, label="Wavelength (cm)")
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)

plt.show()
