from kapteyn import maputils
import numpy
from service import *

fignum = 24
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta1 = 45
title = r"Bonne's equal area projection (BON) with $\theta_1=45$. (Cal. fig.28)"
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---BON',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--BON',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
          'PV2_1'  : theta1
         }
X = polrange()
Y = numpy.arange(-90,100,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lon_world = range(0,360,30)
lon_world.append(180.0+epsilon)
lat_world = [-dec0, -60, -30, 30, 60, dec0]
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       markerpos=markerpos)