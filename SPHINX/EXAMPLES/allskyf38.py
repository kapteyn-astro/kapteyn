from kapteyn import maputils
import numpy
from service import *

fignum = 38
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
theta_a = 60.0; eta = 15.0 
title = r"""WCS conic equal area projection with 
$\theta_a=60$ and $\eta=15$ (Cal. PGSBOX fig.2)"""
header = {'NAXIS'  : 2, 'NAXIS1': 512, 'NAXIS2': 512,
          'CTYPE1' : 'RA---COE',
          'CRVAL1' : 90.0, 'CRPIX1' : 256, 'CUNIT1' : 'deg', 'CDELT1' : -1.0/3.0,
          'CTYPE2' : 'DEC--COE',
          'CRVAL2' : 30.0, 'CRPIX2' : 256, 'CUNIT2' : 'deg', 'CDELT2' : 1.0/3.0,
          'LONPOLE' : 150.0,
          'PV2_1'  : theta_a, 'PV2_2' : eta
         }
X = numpy.arange(0,390.0,30.0);
Y = numpy.arange(-90,120,30.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0(color='r')
grat.setp_lineswcs1(color='b')
grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
grat.setp_tick(plotaxis=1, position=(150.0,210.0), visible=False)
deltapx = 10
# Draw border with standard graticule
header['CRVAL1'] = 0.0;
header['CRVAL2'] = 60.0
header['LONPOLE'] = 999
header['LATPOLE'] = 999
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon, -180+epsilon), starty=(-90,90))
border.setp_gratline((0,1), color='g', lw=2)
border.setp_plotaxis((0,1,2,3), mode='no_ticks', visible=False)
framebgcolor = 'k'  # i.e. black
lon_world = range(0,360,30)
lat_world = [-dec0, -60, -30, 30, 60, dec0]
labkwargs0 = {'color':'w', 'va':'bottom', 'ha':'right'}
labkwargs1 = {'color':'w', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title,
       lon_world=lon_world, lat_world=lat_world, 
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       framebgcolor=framebgcolor)