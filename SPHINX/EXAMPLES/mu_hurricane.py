from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

def plotcoast(fn, pxlim, pylim, col='k'):
   
   coasts = tabarray.tabarray(fn, comchar='s')  # Read two columns from file
   for segment in coasts.segments:
      coastseg = coasts[segment].T
      xw = coastseg[1]; yw = coastseg[0]        # First one appears to be Latitude
      xs = xw; ys = yw                          # Reset lists which store valid pos.
      if 1:  
        # Mask arrays if outside plot box
        xp, yp = annim.projection.topixel((numpy.array(xs),numpy.array(ys)))
        xp = numpy.ma.masked_where(numpy.isnan(xp) | 
                           (xp > pxlim[1]) | (xp < pxlim[0]), xp)
        yp = numpy.ma.masked_where(numpy.isnan(yp) | 
                           (yp > pylim[1]) | (yp < pylim[0]), yp)
        # Mask array could be of type numpy.bool_ instead of numpy.ndarray
        if numpy.isscalar(xp.mask):
           xp.mask = numpy.array(xp.mask, 'bool')
        if numpy.isscalar(yp.mask):
           yp.mask = numpy.array(yp.mask, 'bool')
        # Count the number of positions in these list that are inside the box
        j = 0
        for i in range(len(xp)):
           if not xp.mask[i] and not yp.mask[i]:
              j += 1
        if j > 200:   # Threshold to prevent too much detail and big pdf's
           frame.plot(xp.data, yp.data, color=col)     


# Get a header and change some values
f = maputils.FITSimage("m101.fits")
header = f.hdr
header['CDELT1'] = 0.1
header['CDELT2'] = 0.1
header['CRVAL1'] = 285
header['CRVAL2'] = 20

# Use the changed header as external source for new object
f = maputils.FITSimage(externalheader=header, externaldata=f.dat)
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame, cmap="YlGn")
annim.Image()
grat = annim.Graticule()
grat.setp_ticklabel(wcsaxis=0, fmt="%g^{\circ}")
grat.setp_ticklabel(wcsaxis=1, fmt='Dms')
grat.setp_axislabel(plotaxis='bottom', label='West - East')
grat.setp_axislabel(plotaxis='left', label='South - North')
annim.plot()
annim.projection.allow_invalid = True

# Plot coastlines in black, borders in red
plotcoast('WDB/namer-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('WDB/namer-bdy.txt', annim.pxlim, annim.pylim, col='r')
plotcoast('WDB/samer-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('WDB/samer-bdy.txt', annim.pxlim, annim.pylim, col='r')

annim.interact_imagecolors()
plt.show()
