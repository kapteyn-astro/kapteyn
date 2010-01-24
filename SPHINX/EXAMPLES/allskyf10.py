from kapteyn import maputils
import numpy
from service import *

fignum = 10
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Zenith equal area projection (ZEA). (Cal. fig.13)"
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---ZEA',
          'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.0,
          'CTYPE2' : 'DEC--ZEA',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.0
         }
X = numpy.arange(0,360.0,30.0)
Y = numpy.arange(-90,90,30.0)
Y[0]= -dec0+0.00000001
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
# Set attributes for graticule line at lat = 0
grat.setp_lineswcs1(position=0, color='g', lw=2)   
lat_world = [-dec0, -30, 30, 60]
doplot(frame, fignum, annim, grat, title,
       lat_world=lat_world,
       markerpos=markerpos)