from kapteyn import maputils
import numpy
from service import *

fignum = 27
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"COBE quadrilateralized spherical cube projection (CSC). (Cal. fig.31)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CSC',
          'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--CSC',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0
         }
X = numpy.arange(0,370.0,15.0)
Y = numpy.arange(-75,90,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                       startx=X, starty=Y)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Make a polygon for the border
perimeter = getperimeter(grat)
lon_world = list(range(0,360,30))
lat_world = [-dec0, -60, -30, 0, 30, 60, dec0]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'center', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       perimeter=perimeter, markerpos=markerpos)
