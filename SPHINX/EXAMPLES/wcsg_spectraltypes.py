from kapteyn import wcsgrat
from kapteyn import wcs
from matplotlib import pylab as plt
import pyfits

# Read header of FITS file
hdulist = pyfits.open('mclean.fits')
header = hdulist[0].header

# Create a projection object to get allowed spectral translations
proj = wcs.Projection(header)
altspec = proj.altspec
k = len(altspec) + 1

# Matplotlib 
fig = plt.figure(figsize=(7,10))
fig.subplots_adjust(left=0.12, bottom=0.05, right=0.97, 
                    top=0.97, wspace=0.20, hspace=0.90)
frame = fig.add_subplot(k,1,1)

# Limit range in x to neighbourhood of CRPIX
crpix = proj.crpix[proj.specaxnum-1]
xlim = (crpix-5, crpix+5)

print "Native system", proj.ctype[proj.specaxnum-1]
grat = wcsgrat.Graticule(header, axnum=(3,2), pxlim=xlim)
grat.setp_tick(plotaxis=wcsgrat.bottom, fmt="%.5g")
grat.setp_plotaxis((wcsgrat.bottom, wcsgrat.left), fontsize=9)
grat.setp_plotaxis(wcsgrat.bottom, color='r')
grat.setp_tick(wcsaxis=(0,1), fontsize='8')
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)

print "Spectral translations"
for i, as in enumerate(altspec):
   print i, as
   frame = fig.add_subplot(k,1,i+2)
   grat = wcsgrat.Graticule(header, axnum=(3,2), spectrans=as[0], pxlim=xlim)
   grat.setp_tick(plotaxis=wcsgrat.bottom, fmt="%g")
   grat.setp_plotaxis(wcsgrat.bottom, label=as[0]+' '+as[1], color='b', fontsize=9)
   grat.setp_plotaxis(wcsgrat.left, fontsize=9)
   grat.setp_tick(wcsaxis=(0,1), fontsize='8')
   gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
   gratplot.add(grat)

plt.show()
