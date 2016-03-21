from kapteyn import maputils
import numpy
from service import *

fignum = 39
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta1 = 35
title = r"""Bonne's equal area projection (BON) with conformal latitude 
$\theta_1=35^\circ$ and $\alpha_p=0^\circ$, $\theta_p=+45^\circ$ and N.C.P. at $(45^\circ,0^\circ)$. 
(Cal. PGSBOX example)"""
header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---BON',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
          'CTYPE2' : 'DEC--BON',
          'CRVAL2' : 0.0, 'CRPIX2' : 35, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
          'PV2_1'  : theta1
         }
X = polrange()
Y = numpy.arange(-75.0,90,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0(color='#339333')  # Dark green
grat.setp_lineswcs1(color='#339333')
header['LONPOLE'] = 45.0  # Or PV1_3
header['CRVAL1'] = 0.0
header['CRVAL2'] = 45.0
oblique = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                          startx=X, starty=Y)
oblique.setp_lineswcs0(0.0, color='y')
oblique.setp_lineswcs1(0.0, color='y')
oblique.setp_lineswcs0(range(15,360,45), color='b')
oblique.setp_lineswcs1([15,-15,60, -60], color='b')
oblique.setp_lineswcs0(range(30,360,45), color='r')
oblique.setp_lineswcs1([30,-30,75, -75], color='r')
oblique.setp_lineswcs0(range(45,360,45), color='w')
oblique.setp_lineswcs1((-45,45), color='w')
framebgcolor = 'k'
if not smallversion:
   txt ="""Green:  Native, non-oblique graticule.  Yellow: Equator and prime meridian
Others: Colour coded oblique graticule"""
plt.figtext(0.1, 0.008, txt, fontsize=6)
labkwargs0 = {'visible':False}  # No labels at all!
labkwargs1 = {'visible':False}
doplot(frame, fignum, annim, grat, title, 
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       framebgcolor=framebgcolor)
