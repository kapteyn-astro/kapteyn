from kapteyn import wcsgrat
from matplotlib import pyplot as plt
import pyfits

hdulist = pyfits.open('manyaxes.fits')
header = hdulist[0].header
grat = wcsgrat.Graticule(header, axnum=(1,4))

fig = plt.figure(figsize=(9,9))
frame = fig.add_subplot(1,1,1)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.plot()
plt.show()
