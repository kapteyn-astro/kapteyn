from kapteyn import maputils
import numpy
from service import *

fignum = 25
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Polyconic projection (PCO). (Cal. fig.29)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---PCO',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--PCO',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0
         }
X = polrange()
Y = numpy.arange(-75,90,15.0)
# !!!!!! Let the world coordinates for constant latitude run from 180,180
# instead of 0,360. Then one prevents the connection between the two points
# 179.9999 and 180.0001 which is a jump, but smaller than the definition of
# a rejected jump in the wcsgrat module.
# Also we need to increase the value of 'gridsamples' to
# increase the relative size of a jump.
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2),
                       wylim=(-90,90.0), wxlim=(-180,180),
                       startx=X, starty=Y, gridsamples=2000)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Remove the left 180 deg and print the right 180 deg instead
w1 = numpy.arange(0,151,30.0)
w2 = numpy.arange(180,360,30.0)
w2[0] = 180 + epsilon
lon_world = numpy.concatenate((w1, w2))
lat_world = [-60, -30, 30, 60]
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
