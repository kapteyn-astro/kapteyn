from kapteyn import maputils
import numpy
from service import *

fignum = 9
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)

title = r"""Zenithal polynomial projection (ZPN) with PV2_n parameters 0 to 7. 
(Cal. fig.12)"""
header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
         'CTYPE1' : 'RA---ZPN',
         'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
         'CTYPE2' : 'DEC--ZPN',
         'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0, 
         'PV2_0' : 0.05, 'PV2_1' : 0.975, 'PV2_2' : -0.807, 'PV2_3' : 0.337, 'PV2_4' : -0.065,
         'PV2_5' : 0.01, 'PV2_6' : 0.003,' PV2_7' : -0.001 
         }
X = numpy.arange(0,360.0,30.0)
# Y diverges (this depends on selected parameters). Take save range.
Y = [-70, -60, -45, -30, 0, 15, 30, 45, 60, 90]
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-70,90.0), wxlim=(0,360),
                     startx=X, starty=Y)
# Configure annotations
lat_constval = -72
lat_world = [-60, -30, 0, 60, dec0]
addangle0 = 90.0
annotatekwargs1.update({'ha':'left'})
# No marker position because this can not be evaluated for this projection
# (which has no inverse),
doplot(frame, fignum, annim, grat, title,
       lat_world=lat_world,
       lat_constval=lat_constval)