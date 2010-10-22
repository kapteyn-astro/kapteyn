from kapteyn import maputils
import numpy
from service import *

fignum = 12
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title =  r"""Gall's stereographic projection (CYP) with 
$\mu = 1$ and $\theta_x = 45^\circ$. (Cal. fig.16)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---CYP',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
          'CTYPE2' : 'DEC--CYP',
          'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5,
          'PV2_1'  : 1, 'PV2_2' : numpy.sqrt(2.0)/2.0
         }
X = cylrange()
Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lat_world = [-90, -60,-30, 30, 60, dec0]
# Trick to get the right longs.
w1 = numpy.arange(0,179,30.0)
w2 = numpy.arange(210,360,30.0)
lon_world = numpy.concatenate((w1, w2))
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'center', 'ha':'center'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       lon_fmt='Hms', markerpos=markerpos)
