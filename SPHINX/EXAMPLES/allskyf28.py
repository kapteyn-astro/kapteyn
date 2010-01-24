from kapteyn import maputils
import numpy
from service import *

fignum = 28
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Quadrilateralized spherical cube projection (QSC). (Cal. fig.32)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---QSC',
          'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--QSC',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
         }
X = numpy.arange(-180,180,15)
Y = numpy.arange(-90,100,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lon_world = range(0,360,30)
lat_world = [-dec0, -60, -30, 30, 60, dec0]
perimeter = getperimeter(grat)
deltapx = 1
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       perimeter=perimeter, 
       deltapx=deltapx, plotdata=True)