from kapteyn import wcsgrat
from matplotlib import pylab as plt
import pyfits


hdulist = pyfits.open('ngc6946.fits')
header = hdulist[0].header

# Velocity - Dec
grat = wcsgrat.Graticule(header, axnum=(3,2))
xmax = grat.pxlim[1]+0.5; ymax = grat.pylim[1]+0.5
ruler = grat.ruler(xmax,0.5, xmax, ymax, lambda0 = 0.5, step=5.0/60.0, 
                   fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
                   fliplabelside=True)
ruler.setp_line(lw='2', color='r')
ruler.setp_labels(clip_on=True, color='r')
ruler2 = grat.ruler(0.5,0.5, xmax, ymax, lambda0 = 0.5, step=5.0/60.0, 
                    fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
                    fliplabelside=True)
ruler2.setp_line(lw='2', color='b')
ruler2.setp_labels(clip_on=True, color='b')

grat.setp_plotaxis(wcsgrat.right, label="Offset (Arcsec)", visible=True)

fig = plt.figure(figsize=(7,7))
frame = fig.add_subplot(1,1,1)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.add(ruler)
gratplot.add(ruler2)

plt.show()

