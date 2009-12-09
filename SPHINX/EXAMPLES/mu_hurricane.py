from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

def plotcoast(fn, pxlim, pylim, col='k'):
   t = tabarray.tabarray(fn).T  # Read two columns from file
   xw = t[1]; yw = t[0]         # First one appears to be Latitude
   xs = []; ys = []             # Reset lists which store valid pos.
  
   # Process segments. A segment starts with nan
   for x,y in zip(xw,yw):
     if numpy.isnan(x) and len(xs):  # Useless if empty
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
        xs = []; ys = []
     else:
        # Store world coordinate that is member of current segment
        xs.append(x)
        ys.append(y)


f = maputils.FITSimage("m101cdelt.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame, cmap="YlGn")
annim.Image()
grat = annim.Graticule()
grat.setp_tick(wcsaxis=0, fmt="$%g^{\circ}$")
grat.setp_plotaxis(plotaxis='bottom', label='West - East')
grat.setp_plotaxis(plotaxis='left', label='South - North')
annim.plot()
annim.projection.allow_invalid = True

# Plot coastlines in black, borders in red
plotcoast('namerica-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('namerica-bdy.txt', annim.pxlim, annim.pylim, col='r')
plotcoast('samerica-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('samerica-bdy.txt', annim.pxlim, annim.pylim, col='r')

annim.interact_imagecolors()
plt.show()
