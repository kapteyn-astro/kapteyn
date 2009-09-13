from kapteyn import wcsgrat
import pyfits
from matplotlib import pyplot as plt

hdulist = pyfits.open('example1test.fits')
header = hdulist[0].header
grat = wcsgrat.Graticule(header)

# Use the figure size as suggested by the Graticule object
fig = plt.figure(figsize=(7,7))   # or use attribute grat.figsize
# Create a frame that keeps the aspect ratio also after resizing the window
frame = fig.add_axes(grat.axesrect, aspect=grat.aspectratio, adjustable='box')

gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)

plt.show()
