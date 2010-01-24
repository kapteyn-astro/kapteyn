from kapteyn import maputils
import numpy
from service import *

fignum = 36
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
title = 'Earth in zenithal perspective (AZP). (Cal. fig.36)'
# The ctype's are TLON, TLAT. These are recognized by WCSlib as longitude and latitude.
# Any other prefix is also valid.
header = {'NAXIS'  : 2, 'NAXIS1': 2048, 'NAXIS2': 2048,
          'PC1_1' : 0.9422, 'PC1_2' : -0.3350,
          'PC2_1' : 0.3350, 'PC2_2' : 0.9422,
          'CTYPE1' : 'TLON-AZP',
          'CRVAL1' : 31.15, 'CRPIX1' : 681.67, 'CUNIT1' : 'deg', 'CDELT1' : 0.008542,
          'CTYPE2' : 'TLAT-AZP',
          'CRVAL2' : 30.03, 'CRPIX2' : 60.12, 'CUNIT2' : 'deg', 'CDELT2' : 0.008542,
          'PV2_1' : -1.350, 'PV2_2' : 25.8458,
          'LONPOLE' : 143.3748,
         }
X = numpy.arange(-30,60.0,10.0)
Y = numpy.arange(-40,65,10.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(-20,60),
                       startx=X, starty=Y, gridsamples=4000)
grat.setp_lineswcs1(color='#B30000')
grat.setp_lineswcs0(color='#B30000')
grat.setp_lineswcs0(0, color='r', lw=2)
grat.setp_plotaxis('bottom', mode='all_ticks', label='Latitude / Longitude')
grat.setp_plotaxis('left', mode='switched_ticks', label='Latitude')
grat.setp_plotaxis('right', mode='native_ticks')
grat.setp_tick(wcsaxis=0, color='g')
grat.setp_tick(wcsaxis=1, color='m')
grat.setp_tick(wcsaxis=1, plotaxis=('bottom','right'), color='m', rotation=-30, ha='left')
grat.setp_tick(plotaxis='left', position=-10, visible=False)
g = grat.scanborder(560, 1962, 2)
grat.setp_linespecial(g, color='b', lw=2)
lat_world = lon_world = []
drawgrid = True
markerpos = None
# Proof that WCSlib thinks TLON, TLAT are valid longitudes & latitudes
print "TLON and TLAT are recognized as:", grat.gmap.types
annotatekwargs1.update({'color':'g','va':'top', 'ha':'left'})
doplot(frame, fignum, annim, grat, title,
       plotdata=True, markerpos=markerpos)