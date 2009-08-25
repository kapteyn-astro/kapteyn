from kapteyn import wcsgrat
from matplotlib import pyplot as plt
import pyfits

# FITS part
hdulist = pyfits.open('example1test.fits')
header = hdulist[0].header

# Module wcsgrat part
grat = wcsgrat.Graticule(header)

# Matplotlib part
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.plot()

fig.savefig('fig1.axnumdemo.png')
plt.show()
