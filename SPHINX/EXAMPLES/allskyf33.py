from kapteyn import maputils
import numpy
from service import *

fignum = 33
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta_a = 45.0
t1 = 20.0; t2 = 70.0
eta = abs(t1-t2)/2.0
title = r"""Conic equidistant projection (COD) oblique with $\theta_a=45^\circ$, $\theta_1=20^\circ$
and $\theta_2=70^\circ$, $\alpha_p = 0^\circ$, $\delta_p = 30^\circ$, $\phi_p = 75^\circ$ also:
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$. (Cal. fig.33d)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---COD',
          'CRVAL1' : 0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--COD',
          'CRVAL2' : 30, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
          'PV2_1'  : theta_a, 
          'PV2_2' :  eta,
          'PV1_1'  : 0.0, 'PV1_2' : 90.0,  # IMPORTANT. This is a setting from section 7.1, p 1103
          'LONPOLE' :75.0
         }
X = numpy.arange(0,370.0,15.0);  X[-1] = 180.000001
Y = numpy.arange(-90,100,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Draw border with standard graticule
header['CRVAL1'] = 0.0
header['CRVAL2'] = theta_a
header['LONPOLE'] = 0.0
del header['PV1_1']
del header['PV1_2']
# Non oblique version as border
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon, -180+epsilon), starty=(-90,90))
border.setp_lineswcs0(color='g')   # Show borders in different color
border.setp_lineswcs1(color='g')
lon_world = range(0,360,30)
lat_world = [-60, -30, 30, 60]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'center', 'ha':'center'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
