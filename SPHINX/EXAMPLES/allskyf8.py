from kapteyn import maputils
import numpy
from service import *

fignum = 8
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Zenithal equidistant projection (ARC). (Cal. fig.11)"
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---ARC',
          'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--ARC',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0
         }
X = numpy.arange(0,360.0,30.0)
Y = numpy.arange(-90,90,30.0)
Y[0]= -89.999999   # Graticule for -90 exactly is not plotted
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                        startx=X, starty=Y)

doplot(frame, fignum, annim, grat, title,  
       markerpos=markerpos)