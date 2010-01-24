from kapteyn import maputils
import numpy
from service import *

fignum = 7
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"Slant orthograpic projection (SIN) with: $\xi=\frac{-1}{\sqrt{6}}$ and $\eta=\frac{1}{\sqrt{6}}$ (Cal. fig.10b)"
xi =  -1/numpy.sqrt(6); eta = 1/numpy.sqrt(6)
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---SIN',
          'CRVAL1' :0.0, 'CRPIX1' : 40, 'CUNIT1' : 'deg', 'CDELT1' : -2,
          'CTYPE2' : 'DEC--SIN',
          'CRVAL2' : dec0, 'CRPIX2' : 30, 'CUNIT2' : 'deg', 'CDELT2' : 2,
          'PV2_1'  : xi, 'PV2_2'  : eta
         }
X = numpy.arange(0,360.0,30.0)
Y = numpy.arange(-90,90,10.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), 
                       wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
# Special care for the boundary (algorithm from Calabretta et al)
a = numpy.linspace(0,360,500)
arad = a*numpy.pi/180.0
thetaxrad = -numpy.arctan(xi*numpy.sin(arad)-eta*numpy.cos(arad))
thetax = thetaxrad * 180.0/numpy.pi + 0.000001  # Little shift to avoid NaN's at border
g = grat.addgratline(a, thetax, pixels=False)
grat.setp_linespecial(g, color='g', lw=1)
lat_constval = 50
lon_constval = 180
lat_world = [0,30,60,dec0]
lon_world = range(0,360,30)
doplot(frame, fignum, annim, grat, title, 
       lon_world=lon_world, lat_world=lat_world, 
       lon_constval=lon_constval, lat_constval=lat_constval, 
       markerpos=markerpos)