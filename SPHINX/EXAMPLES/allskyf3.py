from kapteyn import maputils
import numpy
from service import *

fignum = 3
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
mu = 2.0; gamma = 30.0
title = r"""Slant zenithal (azimuthal) perspective projection (AZP) with:
$\gamma=30$ and $\mu=2$ (Cal. fig.6)"""
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---AZP',
          'CRVAL1' :0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--AZP',
          'CRVAL2' : dec0, 'CRPIX2' : 30, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
          'PV2_1'  : mu, 'PV2_2'  : gamma,
         }
lowval = (180.0/numpy.pi)*numpy.arcsin(-1.0/mu) + 0.00001  # Calabretta eq.32
X = numpy.arange(0,360,15.0)
Y = numpy.arange(-30,90,15.0); 
Y[0] = lowval                    # Add lowest possible Y to array
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum=(1,2), 
                       wylim=(lowval,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0((0,90,180,270), lw=2)
grat.setp_lineswcs1(0, lw=2)
grat.setp_lineswcs1(lowval, lw=2, color='g')
lat_world = [0, 30, 60, 90]
lon_world = range(0,360,30)
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
addangle0 = -90
lat_constval= -5
doplot(frame, fignum, annim, grat, title, 
       lon_world=lon_world, lat_world=lat_world,
       lat_constval=lat_constval,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       addangle0=addangle0, markerpos=markerpos)
