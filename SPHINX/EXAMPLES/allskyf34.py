from kapteyn import maputils
import numpy
from service import *

fignum = 34
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"""Hammer Aitoff projection (AIT) oblique with:
$(\alpha_p,\delta_p) = (0^\circ,30^\circ)$, $\phi_p = 75^\circ$ also: 
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$. (Cal. fig.34d)"""
# Header works only with a patched wcslib 4.3
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---AIT',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--AIT',
          'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0, 
          'LONPOLE' :75.0,
          'PV1_1'  : 0.0, 'PV1_2' : 90.0,  # IMPORTANT. This is a setting from Cal.section 7.1, p 1103
         }
X = numpy.arange(0,390.0,15.0); 
Y = numpy.arange(-75,90,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Draw border with standard graticule
header['CRVAL1'] = 0.0
header['CRVAL2'] = 0.0
del header['PV1_1']
del header['PV1_2']
header['LONPOLE'] = 0.0
header['LATPOLE'] = 0.0
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon, -180+epsilon), skipy=True)
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
