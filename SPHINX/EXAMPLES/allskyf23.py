from kapteyn import maputils
import numpy
from service import *

fignum = 23
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta_a = 45
t1 = 20.0; t2 = 70.0
eta = abs(t1-t2)/2.0
title = r"""Conic orthomorfic projection (COO) with:
$\theta_a=45$, $\theta_1=20$ and $\theta2=70$. (Cal. fig.27)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---COO',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--COO',
          'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
          'PV2_1'  : theta_a, 'PV2_2' : eta
         }
X = cylrange()
Y = numpy.arange(-30,90,30.0)  # Diverges at theta_a= -90.0
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs1(-30, linestyle='--', color='g')
lon_world = range(0,360,30)
lon_world.append(180.0+epsilon)
lat_world = [-dec0, -60, -30, 30, 60, dec0]
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       markerpos=markerpos)