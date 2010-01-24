from kapteyn import maputils
import numpy
from service import *

fignum = 11
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Airy projection (AIR) with $\theta_b = 45^\circ$. (Cal. fig.14)"
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---AIR',
          'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--AIR',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0
         }
X = numpy.arange(0,360.0,30.0)
Y = numpy.arange(-30,90,10.0)
# Diverges at dec = -90, start at dec = -30
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                        startx=X, starty=Y)
lat_world = [-30, -20, -10, 10, 40, 70]
doplot(frame, fignum, annim, grat, title,
       lat_world=lat_world,
       markerpos=markerpos)