from kapteyn import maputils
import numpy
from service import *

# Fig 2 in celestial article (Calabretta et al) shows a  positive cdelt1
fignum = 0                   # id of script and plot
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = r"""Linear equatorial coordinate system with:
$(\alpha_0,\delta_0) = (120^\circ,60^\circ)$ (Cal. fig.2-upper)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA',
          'CRVAL1' : 120.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : 5.0,
          'CTYPE2' : 'DEC',
          'CRVAL2' : 60.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
         }
X = numpy.arange(-60,301.0,30.0);
Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(header, axnum=(1,2), 
                       wylim=(-90,90.0), wxlim=(-60,300),
                       startx=X, starty=Y)
#print "Lonpole, latpole values: ", \
#      annim.projection.lonpole, annim.projection.latpole, 
lat_world = [-60, -30, 30, 60]
lon_world = list(range(-30,301,30))
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title, 
       lat_world=lat_world, lon_world=lon_world,
       lon_fmt='Hms',
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
