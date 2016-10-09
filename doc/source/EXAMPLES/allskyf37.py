from kapteyn import maputils
import numpy
from service import *

fignum = 37
fig = plt.figure(figsize=figsize)
frame = fig.add_axes((0.1,0.15,0.8,0.75))
title = 'WCS polyconic (PGSBOX fig.1)'
rot = 30.0 *numpy.pi/180.0
header = {'NAXIS'  : 2, 'NAXIS1': 512, 'NAXIS2': 512,
          'CTYPE1' : 'RA---PCO',
          'PC1_1' : numpy.cos(rot), 'PC1_2' : numpy.sin(rot),
          'PC2_1' : -numpy.sin(rot), 'PC2_2' : numpy.cos(rot),
          'CRVAL1' : 332.0, 'CRPIX1' : 192, 'CUNIT1' : 'deg', 'CDELT1' : -1.0/5.0,
          'CTYPE2' : 'DEC--PCO',
          'CRVAL2' : 40.0, 'CRPIX2' : 640, 'CUNIT2' : 'deg', 'CDELT2' : 1.0/5.0,
          'LONPOLE' : -30.0
         }
X = numpy.arange(-180,180.0,15.0);
Y = numpy.arange(-75,90,15.0) 
# Here we demonstrate how to avoid a jump at the right corner boundary 
# of the plot by increasing the value of 'gridsamples'.
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                       startx=X, starty=Y, gridsamples=4000)
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
grat.setp_tick(wcsaxis=0, position=15*numpy.array((18,20,22,23)), visible=False)
grat.setp_tick(wcsaxis=0, fmt="Hms")
grat.setp_tick(wcsaxis=1, fmt="Dms")
header['CRVAL1'] = 0.0
header['CRVAL2'] = 0.0
header['LONPOLE'] = 999
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon, -180+epsilon), starty=(-89.5,))
border.setp_gratline((0,1), color='g', lw=2)
border.setp_plotaxis((0,1,2,3), mode='no_ticks', visible=False)
lon_world = list(range(0,360,30))
lat_world = [-dec0, -60, -30, 30, 60, dec0]
labkwargs0 = {'color':'r', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world,
       labkwargs0=labkwargs0, labkwargs1=labkwargs1)
