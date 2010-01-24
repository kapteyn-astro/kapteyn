from kapteyn import maputils
import numpy
from service import *

fignum = 15
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title =  "Mercator's projection (MER). (Cal. fig.19)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---MER',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--MER',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
         }
X = cylrange()
Y = numpy.arange(-80,90,10.0)  # Diverges at +-90 so exclude these values
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(header, axnum= (1,2), wylim=(-80,80.0), wxlim=(0,360),
                       startx=X, starty=Y)
lat_world = [-90, -60,-30, 30, 60, dec0]
lon_world = range(0,360,30)
lon_world.append(180+epsilon)
grat.setp_lineswcs1((-80,80), linestyle='--', color='g')
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       markerpos=markerpos)