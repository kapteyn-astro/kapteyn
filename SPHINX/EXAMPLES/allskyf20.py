from kapteyn import maputils
import numpy
from service import *

fignum = 20
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta_a = 45
t1 = 20.0; t2 = 70.0
eta = abs(t1-t2)/2.0
title = r"""Conic perspective projection (COP) with:
$\theta_a=45^\circ$, $\theta_1=20^\circ$ and $\theta_2=70^\circ$. (Cal. fig.24)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---COP',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.5,
          'CTYPE2' : 'DEC--COP',
          'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.5,
          'PV2_1'  : theta_a, 'PV2_2' : eta
         }
X = numpy.arange(0,370.0,30.0);  X[-1] = 180+epsilon
Y = numpy.arange(-30,90,15.0)  # Diverges at theta_a +- 90.0
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs1(-30, linestyle='--', color='g')
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
lon_world = range(0,360,30)
lon_world.append(180+epsilon)
lat_world = [-30, 0, 30, 60]
addangle0 = -90
lat_constval = -31
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'left'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       lat_constval=lat_constval,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       addangle0=addangle0, markerpos=markerpos)