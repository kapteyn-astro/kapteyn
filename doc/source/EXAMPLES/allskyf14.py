from kapteyn import maputils
import numpy
from service import *

fignum = 14
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = "Plate Carree projection (CAR). (Cal. fig.18)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CAR',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--CAR',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
         }
X = cylrange()
Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                        startx=X, starty=Y)
lat_world = [-90, -60,-30, 0, 30, 60, dec0]
# Remove the left 180 deg and print the right 180 deg instead
w1 = numpy.arange(0,179,30.0)
w2 = numpy.arange(180,360,30.0)
w2[0] = 180 + epsilon
lon_world = numpy.concatenate((w1, w2))
labkwargs0 = {'color':'r', 'va':'top', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)