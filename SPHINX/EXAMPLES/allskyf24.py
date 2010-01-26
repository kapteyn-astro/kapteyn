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
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
w1 = numpy.arange(0,151,30.0)
w2 = numpy.arange(210,360,30.0)
lon_world = numpy.concatenate((w1, w2))
lat_world = [-60, -30, 30, 60]
lat_constval = 10
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       lat_constval=lat_constval, markerpos=markerpos)