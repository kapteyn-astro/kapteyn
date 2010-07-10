from kapteyn import maputils
import numpy
from service import *

# Fig 2 in celestial article (Calabretta et al) shows a  positive cdelt1
fignum = 1
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"""Plate Carree projection (CAR), oblique with:
$(\alpha_0,\delta_0,\phi_p) = (120^\circ,0^\circ,0^\circ)$
and obviously cdelt1 $>$ 0. (Cal. fig. 2)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CAR',
          'CRVAL1' : 120.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : 5.0,
          'CTYPE2' : 'DEC--CAR',
          'CRVAL2' : 60.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
          'LONPOLE' : 0.0,
         }
X = numpy.arange(0,360.0,30.0);
Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(header, axnum= (1,2),
                       wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
# Get the non-oblique version for the border
header['CRVAL1'] = 0.0
header['CRVAL2'] = 0.0
border = annim.Graticule(header, axnum= (1,2), boxsamples=10000,
                         wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon,-180+epsilon), starty=(-90,90))
lat_world = [-60, 0, 30, 60]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'center'}
doplot(frame, fignum, annim, grat, title,
       lat_world=lat_world, deltapx1=1, deltapy1=1,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
