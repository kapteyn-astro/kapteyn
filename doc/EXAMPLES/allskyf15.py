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
grat.setp_lineswcs1((-80,80), linestyle='--', color='g')
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
lat_world = [-60,-30, 30, 60]
# Remove the left 180 deg and print the right 180 deg instead
w1 = numpy.arange(0,179,30.0)
w2 = numpy.arange(180,360,30.0)
w2[0] = 180 + epsilon
lon_world = numpy.concatenate((w1, w2))
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'center', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       deltapy1=0.5,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
