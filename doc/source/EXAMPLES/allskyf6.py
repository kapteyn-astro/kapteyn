from kapteyn import maputils
import numpy
from service import *

fignum = 6
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Stereographic projection (STG) diverges at $\theta=-90^\circ$. (Cal. fig.9)"
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---STG',
          'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -12.0,
          'CTYPE2' : 'DEC--STG',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 12.0,
         }
X = numpy.arange(0,360.0,30.0)
Y = numpy.arange(-60,90,10.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), 
                       wylim=(-60,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lat_constval = -62
lon_world = list(range(0,360,30))
lat_world = list(range(-50, 10, 10))
addangle0 = -90
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title, 
       lon_world=lon_world, lat_world=lat_world, lat_constval=lat_constval,
       addangle0=addangle0, labkwargs1=labkwargs1, markerpos=markerpos)
