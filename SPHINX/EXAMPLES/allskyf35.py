from kapteyn import maputils
import numpy
from service import *

fignum = 35
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"""COBE quadrilateralized spherical cube projection (CSC) oblique with:
$(\alpha_p,\delta_p) = (0^\circ,30^\circ)$, $\phi_p = 75^\circ$ also: 
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$. (Cal. fig.34d)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CSC',
          'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--CSC',
          'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
          'LONPOLE': 75.0,
          'PV1_1'  : 0.0, 'PV1_2' : 90.0,
         }
X = numpy.arange(0,370.0,30.0)
Y = numpy.arange(-60,90,30.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                       startx=X, starty=Y)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Take border from non-oblique version
header['CRVAL2'] = 0.0
del header['PV1_1']
del header['PV1_2']
del header['LONPOLE']
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         skipx=True, skipy=True)
perimeter = getperimeter(border)
lon_world = range(0,360,30)
lat_world = [-60, -30, 30, 60]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'left'}
labkwargs1 = {'color':'b', 'va':'top', 'ha':'center'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       perimeter=perimeter, markerpos=markerpos)
