from kapteyn import maputils, tabarray
import numpy
import sys
from matplotlib import pyplot as plt


__version__ = '1.91'

epsilon = 0.0000000001

def radians(a):
   return a*numpy.pi/180.0

def degrees(a):
   return a*180.0/numpy.pi

def cylrange():
   X = numpy.arange(0,400.0,30.0); 
   # Replace last two (dummy) values by two values around 180 degrees   
   X[-1] = 180.0 - epsilon
   X[-2] = 180.0 + epsilon
   return X

def polrange(): 
   X = numpy.arange(0,380.0,15); 
   # Replace last two (dummy) values by two values around 180 degrees   
   X[-1] = 180.0 - epsilon
   X[-2] = 180.0 + epsilon
   return X


def getperimeter(grat):
   # Calculate perimeter of QUAD projection 
   xlo, y = grat.gmap.topixel((-45.0-epsilon, 0.0))
   xhi, y = grat.gmap.topixel((315+epsilon, 0.0))
   x, ylo = grat.gmap.topixel((180, -45))
   x, yhi = grat.gmap.topixel((180, 45))
   x1, y = grat.gmap.topixel((45-epsilon, 0.0))
   x, ylolo = grat.gmap.topixel((0, -135+epsilon))
   x, yhihi = grat.gmap.topixel((0, 135-epsilon))
   perimeter = [(xlo,ylo), (x1,ylo), (x1,ylolo), (xhi,ylolo), (xhi,yhihi),
                (x1,yhihi), (x1,yhi), (xlo,yhi), (xlo,ylo)]
   return perimeter


def plotcoast(fn, frame, grat, col='k', lim=100, decim=5, plotsym=None, sign=1.0):
   coasts = tabarray.tabarray(fn, comchar='s')  # Read two columns from file
   for segment in coasts.segments:
      coastseg = coasts[segment].T
      xw = sign * coastseg[1]; yw = coastseg[0] # First one appears to be Latitude
      xs = xw; ys = yw                          # Reset lists which store valid pos.
      if 1:
         # Mask arrays if outside plot box
         xp, yp = grat.gmap.topixel((numpy.array(xs),numpy.array(ys)))
         # Be sure you understand 
         # the operator precedence: (a > 2) | (a < 5) is the proper syntax 
         # because a > 2 | a < 5 will result in an error due to the fact 
         # that 2 | a is evaluated first.
         xp = numpy.ma.masked_where(numpy.isnan(xp) |
                              (xp > grat.pxlim[1]) | (xp < grat.pxlim[0]), xp)
         yp = numpy.ma.masked_where(numpy.isnan(yp) |
                              (yp > grat.pylim[1]) | (yp < grat.pylim[0]), yp)
         # Mask array could be of type numpy.bool_ instead of numpy.ndarray
         if numpy.isscalar(xp.mask):
            xp.mask = numpy.array(xp.mask, 'bool')
         if numpy.isscalar(yp.mask):
            yp.mask = numpy.array(yp.mask, 'bool')
         # Count the number of positions in this list that are inside the box
         xdc = []; ydc = []
         for i in range(len(xp)):
            if not xp.mask[i] and not yp.mask[i]:
               if not i%decim:
                  xdc.append(xp.data[i])
                  ydc.append(yp.data[i])
         if len(xdc) >= lim:
            if plotsym == None:
               frame.plot(xdc, ydc, color=col)
            else:
               frame.plot(xdc, ydc, '.', markersize=1, color=col)



# Set defaults which can be overwritten by the allskyfxx.py scripts
title = ''
titlepos = 1.02
dec0 = 89.9999999999
fsize = 10
figsize = (7,6)
drawgrid = False
grat = None
smallversion = False
plotbox = (0.1,0.05,0.8,0.8)
markerpos = "120 deg 60 deg"


def doplot(frame, fignum, annim, grat, title, 
           lon_world=None, lat_world=None, 
           lon_constval=None, lat_constval=None,
           lon_fmt=None, lat_fmt=None,
           markerpos=None, 
           plotdata=False, perimeter=None, drawgrid=None, 
           smallversion=False, addangle0=0.0, addangle1=0.0, 
           framebgcolor=None, deltapx0=0.0, deltapy0=0.0,
           deltapx1=0.0, deltapy1=0.0,
           labkwargs0={'color':'r'}, labkwargs1={'color':'b'}):
# Apply some extra settings
   
   if framebgcolor != None:
      frame.set_axis_bgcolor(framebgcolor)

   # Plot coastlines if required
   if plotdata:
      if fignum == 36:
         plotcoast('WDB/world.txt', frame, grat, col='k', lim=100)
      else:
         plotcoast('WDB/world.txt', frame, grat, col='r', lim=50, 
                    decim=20, plotsym=',', sign=-1)

   # Plot alternative borders
   if perimeter != None:
      p = plt.Polygon(perimeter, facecolor='#d6eaef', lw=2)
      frame.add_patch(p)
      Xp, Yp = zip(*perimeter)
      frame.plot(Xp, Yp, color='r')

   if lon_constval == None:
      lon_constval = 0.0    # Reasonable for all sky plots
   if lat_constval == None:
      lat_constval = 0.0    # Reasonable for all sky plots
   if lon_fmt == None:
      lon_fmt = 'Dms'
   if lat_fmt == None:
      lat_fmt = 'Dms'
   # Plot labels inside graticule if required
   labkwargs0.update({'fontsize':fsize})
   labkwargs1.update({'fontsize':fsize})
   ilabs1 = grat.Insidelabels(wcsaxis=0, 
                        world=lon_world, constval=lat_constval, 
                        deltapx=deltapx0, deltapy=deltapy0, 
                        addangle=addangle0, fmt=lon_fmt, **labkwargs0)
   ilabs2 = grat.Insidelabels(wcsaxis=1, 
                        world=lat_world, constval=lon_constval, 
                        deltapx=deltapx1, deltapy=deltapy1, 
                        addangle=addangle1, fmt=lat_fmt, **labkwargs1)

   # Plot just 1 pixel c.q. marker
   if markerpos != None:
      annim.Marker(pos=markerpos, marker='o', color='red')

   if drawgrid:
      pixellabels = annim.Pixellabels(plotaxis=(2,3))

   # Plot the title
   title = "Fig. %d: "%fignum + title
   if smallversion:
      t = frame.set_title(title, color='g', fontsize=10)
   else:
      t = frame.set_title(title, color='g', fontsize=13, linespacing=1.5)
   t.set_y(titlepos)
   annim.plot()
   annim.interact_toolbarinfo()

   plt.show()
