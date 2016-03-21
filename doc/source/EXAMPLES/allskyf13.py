from kapteyn import maputils
import numpy
from service import *

fignum = 13
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Lambert's equal area projection (CEA) with $\lambda = 1$. (Cal. fig.17)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CEA',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--CEA',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
          'PV2_1'  : 1
         }
X = cylrange()
Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lat_world = [-60,-30, 30, 60]
#lon_world = range(0,360,30)
#lon_world.append(180+epsilon)
w1 = numpy.arange(0,179,30.0)
w2 = numpy.arange(180,360,30.0)
w2[0] = 180 + epsilon
lon_world = numpy.concatenate((w1, w2))
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'center', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       deltapy1=0.5,
       markerpos=markerpos)
