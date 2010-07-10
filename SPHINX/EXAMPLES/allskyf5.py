from kapteyn import maputils
import numpy
from service import *

fignum = 5
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Gnomonic projection (TAN) diverges at $\theta=0^\circ$. (Cal. fig.8)"
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
         }
X = numpy.arange(0,360.0,15.0)
Y = [20, 30,45, 60, 75, 90]
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), 
                       wylim=(20.0,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lat_constval = 18
lon_world = range(0,360,30)
lat_world = [20, 30, 60, dec0]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
grat.setp_lineswcs1(20, color='g', linestyle='--')
addangle0 = -90
doplot(frame, fignum, annim, grat, title, 
       lon_world=lon_world, lat_world=lat_world, lat_constval=lat_constval,
       addangle0=addangle0, labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
