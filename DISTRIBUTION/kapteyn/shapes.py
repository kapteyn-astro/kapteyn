#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    shapes.py
# PURPOSE: Provide methods to plot polygons, ellipses, circles
#          rectangles and splines. For the area enclused by these shapes
#          the flux can be calculated, plotted and/or saved to disk.
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    April 25, 2010
# UPDATE:  April 25, 2010
#
# VERSION: 1.0
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#
#----------------------------------------------------------------------
"""
.. highlight:: python
   :linenothreshold: 10

Module shapes
===============
This module defines a class for drawing shapes that define an area in your
image. The drawing is interactive using mouse- and keyboard buttons.
For each defined area the module :mod:`maputils` calculates the sum of the intensities,
the area and some other properties of the data. The shapes are one of
polygon, ellipse, circle, rectangle or spline.

The strength of this module is that it duplicates a shape in different
images using transformations to world coordinates. This enables one to compare
flux in two images with different WCS systems.
 
.. autoclass:: Shapecollection

"""

#!/usr/bin/env python
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.pyplot import show, figure, get_current_fig_manager
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
from matplotlib.widgets import Button, RadioButtons
from datetime import datetime
from kapteyn import tabarray
import matplotlib.nxutils as nxutils
import numpy
from kapteyn.maputils import AxesCallback
from sys import stdout, exit
from kapteyn.mplutil import KeyPressFilter
try:
   from gipsy import finis, anyout
   gipsymod = True
except:
   gipsymod = False


KeyPressFilter.allowed = ['f', 'g']

def button_setcolor(btnobj, c):
   """
   -----------------------------------------------------------
   Purpose:    Change color of button
   Parameters:
    btnobj     The button
    c          A color
   -----------------------------------------------------------
   """
   btnobj.color = c
   btnobj._lastcolor = c
   btnobj.ax.set_axis_bgcolor(c)
   if btnobj.drawon:
      btnobj.ax.figure.canvas.draw()
Button.setcolor = button_setcolor


def cubicspline(xyu, nsamples):
   """
   -----------------------------------------------------------
   Purpose:    Give a set positions (x,y) as control points, 
               calculate interpolated points which span cubic 
               polynomials between those control points.
   Parameters:
    xyu:       Unscaled sequence of positions (x,y) usually 
               representing a user defined polygon
    nsamples:  Number of spline interpolation points in each segment
               A segment is has two control points as start and end point.
   Returns:    Unscaled interpolated data including the control points.
               As a polygon the interpolated data is not closed so the last
               point is not equal to the first.
   Notes:      -Cubic splines, Mathematical elements for computer graphics, 
               2nd ed., Rogers & Adams
               -The algorithm calculates internal tangents but it needs
               two additional tangents for the first and last control points.
               Here we force smoothness by setting these to tangents to
               the same value. which is the tangent corresponding to the
               first segment.
               -Internally, the data is always scaled in the range [-1,1]
   Examples:   x = [0,1000,2000, 3000, 2500,  1200,  800]
               y = [0,1000,-1000, 0,   1000, 1800, 1700]
               xy = zip(x,y)
               nsamples = 100
               xy_spl = cubicspline(xy, samples)
   -----------------------------------------------------------
   """
   np = len(xyu)
   if np < 3:
      return None                                # Not enough control points to do anything
   xymin = numpy.min(xyu,0)
   xymax = numpy.max(xyu,0)
   scale = float(max(xymax[0]-xymin[0],xymax[1]-xymin[1]))
   if scale == 0.0:
      return None
   xys = numpy.asarray(xyu)/scale
   x, y = zip(*xys)
   x = list(x)
   y = list(y)
   x.append(x[0])                                # Close polygon
   y.append(y[0])
   xy = zip(x,y)
   np += 1                                       # A vertex was added, increase the counter
   #x,y = zip(*xy)
   x1 = numpy.array(x[:-1])
   x2 = numpy.array(x[1:])
   y1 = numpy.array(y[:-1])
   y2 = numpy.array(y[1:])
   t = numpy.hypot(x2-x1,y2-y1)                  # Chord approximation tk. Note that t[0] is t2 in Rogers & Adams
   M = numpy.zeros((np,np))
   M[0,0] = M[-1,-1] = 1.0                       # Add outer parts of matrix
   R = numpy.zeros((np,2))                       # Initialize matrix R in R&A (eq. 5-15)
   for row in range(1,np-1):                     # Fill matrices M and R
      col = row -1
      t_first = t[row-1]; t_next = t[row];
      M[row,col] = t_next
      M[row,col+1] = 2.0*(t_first + t_next)
      M[row,col+2] = t_first
      x1 = xy[row-1][0]; x2 = xy[row][0]; x3 = xy[row+1][0]
      y1 = xy[row-1][1]; y2 = xy[row][1]; y3 = xy[row+1][1]
      R[row,0] = 3.0 * ( ((t_first/t_next)*(x3-x2)) + ((t_next/t_first)*(x2-x1)) )
      R[row,1] = 3.0 * ( ((t_first/t_next)*(y3-y2)) + ((t_next/t_first)*(y2-y1)) )
   
   # Here we need the tangents of the first and last point
   # Assume last point is not equal to first point, i.e. if
   # data is polygon data, then polygon is not closed
   
   """
   # Circle through three points with starting point in the middle
   x1 = x[1]; x2 = x[0]; x3 = x[-2]
   y1 = y[1]; y2 = y[0]; y3 = y[-2]
   ma = (y2-y1)/(x2-x1)
   mb = (y3-y2)/(x3-x2)
   xt = (ma*mb*(y1-y3)+mb*(x1+x2)-ma*(x2+x3))/(2.0*(mb-ma))
   yt = ma*(xt-x1) + y1
   P1 = Plast = (-(xt-x2),(yt-y2))
   """
   P1 = (x[1]-x[0],y[1]-y[0]); Plast = P1  #(x[0]-x[-1],y[0]-y[-1])
   R[0] = P1
   R[-1] = Plast
   Ptan = numpy.matrix(M).I * R                  # Internal tangent vectors
   # Here we create the sub-divisions in a segment. Take the number of samples in  
   # each segment the same. The idea is that the user increaser the distance between
   # control points on polygon parts that look like straight lines. Less spline
   # points are needed to interpolate those segments.
   # The alternative is not implemented. The alternative is to calculate samples
   # that are equidistant on all segments.
   ta = numpy.array(range(nsamples))/float(nsamples)
   ta2 = ta*ta
   ta3 = ta*ta2
   F = numpy.matrix(numpy.zeros((4,len(ta))))
   G = numpy.matrix(numpy.zeros((4,2)))
   F[0] = 2.0*ta3 - 3.0*ta2 + 1.0
   F[1] = 1.0 - F[0]
   F2 = ta*(ta2-2.0*ta+1.0)
   F3 = ta*(ta2-ta)
   for seg in range(np-1):
      F[2] =  F2 * t[seg]
      F[3] =  F3 * t[seg]
      G[0] = xy[seg]
      G[1] = xy[seg+1]
      G[2] = Ptan[seg]
      G[3] = Ptan[seg+1]
      Pspline = F.T * G
      if seg == 0:
         v = Pspline.copy()
      else:
         v = numpy.concatenate((v,Pspline))
   if scale:                                     # Rescale to original size
      v *= scale
   return v
   



class Poly(Polygon):
   def __init__(self, frame, framenr, active, markers, x0, y0, type=None, acolor='b', 
                spline=False, **kwargs):
      # Either initialize object with one marker at x0, y0 or
      # initialize with a sequence of markers copied from an  existing object.
      canvas = frame.figure.canvas
      self.canvas = canvas
      self.active = active
      self.frame = frame
      self.framenr = framenr
      self.kwargs = kwargs
      self.shapecolor = acolor           # Color when active
      self.edgecolor = 'r'
      self.shapetype = type
      self.epsilon = 20                  # A distance in display coordinates
      Polygon.__init__(self, zip([x0],[y0]), closed=True, alpha=0.1, edgecolor='r', **self.kwargs)
      if not spline:
         self.frame.add_patch(self)
      self.markers = Line2D([x0],[y0], marker='o', markerfacecolor='r', color='r', animated=False)
      self.frame.add_line(self.markers)
      self.x0 = x0
      self.y0 = y0
      self.area = None
      self.sum = None
      self.flux = None
      self.spline = None
      if spline:
         self.spline = Polygon(zip([x0],[y0]), closed=True, alpha=0.3, edgecolor='r')
         self.frame.add_patch(self.spline)
      if self.active:
         self.set_active(markers)


   def copy(self, frame, framenr, x0, y0, xy, active, markers):
      newobj = Poly(frame, framenr, active, markers, x0, y0, type=self.shapetype, **self.kwargs)
      newobj.updatexy(xy)
      return newobj


   def updatexy(self, xy):
      self.xy = xy
      self.markers.set_data(zip(*self.xy))
      self.centroid()


   def shiftxy(self, x0, y0):
      x, y = zip(*self.xy)
      dx = x0 - self.x0
      dy = y0 - self.y0
      x = numpy.asarray(x) + dx
      y = numpy.asarray(y) + dy
      return zip(x,y)


   def set_active(self, markers=False):
      self.active = True
      alpha = 0.1
      if self.markers:
         self.markers.set_visible(markers)
      if self.spline != None:
         self.spline.set_facecolor(self.shapecolor)
         self.spline.set_edgecolor(self.edgecolor)
         self.spline.set_alpha(alpha)
      else:
         self.set_facecolor(self.shapecolor)
         self.set_edgecolor(self.edgecolor)
         self.set_alpha(alpha)


   def addvertex(self, x, y, markers=True):
      xyl = list(self.xy)
      xyl.append([x,y])    # Append the new position
      self.xy = xyl
      self.markers.set_data(zip(*self.xy))
      self.markers.set_visible(markers)
      self.centroid()


   def set_markers(self, vis=True):
      if self.markers != None:
         self.markers.set_visible(vis)


   def set_inactive(self):
      self.active = False
      self.set_markers(False)
      alpha = 0.3
      if self.spline == None:
         self.set_facecolor('y')
         self.set_edgecolor(self.edgecolor)
         self.set_alpha(alpha)
      else:
         self.spline.set_facecolor('y')
         self.spline.set_alpha(alpha)
         self.spline.set_edgecolor(self.edgecolor)


   def indexclosestmarker(self, x, y):
      # These coordinates are display pixels!
      xt,yt = zip(*self.frame.transData.transform(self.xy))
      xt = numpy.asarray(xt); yt = numpy.asarray(yt)
      d = (xt-x)*(xt-x) + (yt-y)*(yt-y)
      a2 = numpy.equal(d, numpy.amin(d))
      inds = numpy.nonzero(a2)[0]
      i = int(inds[0])
      if d[i] > self.epsilon:
         i = None
      self.closestindx = i
      return i
      """
      # What is the distance to the center?
      xt, yt = self.frame.transData.transform((p.xcent,p.ycent))
      dc = (xt-x)*(xt-x) + (yt-y)*(yt-y)
      if i == None:
         if dc < self.epsilon:
            i = -1
      else:
         if dc < self.epsilon and dc < d[i]:
            i = -1
      return i
      """

   def deletemarker(self, indx):
      xyl = list(self.xy)
      if len(xyl) == 2:
         return
      if indx == 0:
         del xyl[-1]
         del xyl[0]
         if len(xyl) > 0:
            xyl.append(xyl[0])       # Make polygon closed again
      else:
         del xyl[indx]
      self.xy = xyl
      self.markers.set_data(zip(*self.xy))


   def indexsegmentinrange(self, x, y):
      # Find index of nearest segment
      if not self.spline:
         tra = self.get_transform()
      else:
         tra = self.spline.get_transform()
      xyt = tra.transform(self.xy)
      p = x, y
      ind = None
      dmin = None
      for i in range(len(xyt)-1):
         s0 = xyt[i]
         s1 = xyt[i+1]
         #d = (x-s0[0])*(x-s0[0]) + (y-s0[1])*(y-s0[1]) + (x-s1[0])*(x-s1[0]) + (y-s1[1])*(y-s1[1])
         # Criterion is the orthogonal distance to a line segment.
         d = dist_point_to_segment(p, s0, s1)
         if dmin == None:
            dmin = d
            ind = i
         else:
            if d < dmin:
               dmin = d
               ind = i
      return ind


   def insertmarker(self, x, y, indx):
      if indx == None:
         return
      i = indx + 1
      if i >= len(self.xy):   # After the last segment you can only append
         return
      self.xy = numpy.array(list(self.xy[:i]) + [(x, y)] + list(self.xy[i:]))
      self.markers.set_data(zip(*self.xy))
      self.centroid()


   def delete(self):
      # Remove visible elements
      self.frame.lines.remove(self.markers)
      if self.spline:
         self.frame.patches.remove(self.spline)
      else:
         self.frame.patches.remove(self)
          
          
   def inside(self, x, y):
      # Is this (x,y) position within the polygon?
      #poly = numpy.asarray(zip(p.x, p.y))
      pos = [(x,y)]
      mask = nxutils.points_inside_poly(pos, self.xy)
      return mask[0]
      
      
   def moveall(self, x0, y0):
      dx = x0-self.x0; dy = y0-self.y0
      x, y = zip(*self.xy)
      x = numpy.asarray(x) + dx
      y = numpy.asarray(y) + dy
      self.markers.set_data(x, y)
      self.xy = zip(x, y)
      self.centroid()
      

   def movemarker(self, x, y, indx):
      self.xy[indx] = x,y
      self.markers.set_data(zip(*self.xy))
      self.centroid()


   def centroid(self):
      x, y = zip(*self.xy)
      x = numpy.asarray(x)
      y = numpy.asarray(y)
      self.x0 = x.sum()/len(x)
      self.y0 = y.sum()/len(y)


   def allinsideframe(self):
      x1, x2 = self.frame.get_xlim()
      y1, y2 = self.frame.get_ylim()
      if self.spline:
         xy = self.spline.xy
      else:
         xy = self.xy
      inside = True
      for x, y in xy:
          if x > x2 or x < x1 or y > y2 or y < y1:
             inside = False
             break
      return inside


class Ellipse(Poly):
   def __init__(self, frame, framenr, active, markers, x0, y0, type=None, 
                r1=0.0, r2=0.0, r3=30.0, **kwargs):
      # r1, r2 are extra parameters. For a default ellipse these represent the major cq. minor axes.
      startang, endang, delta = (0.0, 360.0, 1.0)
      self.phi = numpy.arange(startang, endang+delta, delta) * numpy.pi/180.0
      self.pa = r3
      self.x0 = x0
      self.y0 = y0
      self.maj = r1
      self.min = r2
      Poly.__init__(self, frame, framenr, active, markers, x0, y0, type=type, acolor='m', **kwargs)
      self.updatexy(self.getvertices())
   
   
   def getvertices(self):
      pa_rad = self.pa*numpy.pi/180.0
      sinP = numpy.sin(pa_rad)
      cosP = numpy.cos(pa_rad)
      phi = self.phi
      cosA = numpy.cos(phi)
      sinA = numpy.sin(phi)
      min = self.min
      maj = self.maj
      """
      self.xmajor = maj * cosP + self.x0
      self.ymajor = maj * sinP + self.y0
      self.xminor = -min * sinP + self.x0
      self.yminor =  min * cosP + self.y0
      dmin = (self.xminor-self.x0)**2 + (self.yminor-self.y0)**2
      dmaj = (self.xmajor-self.x0)**2 + (self.ymajor-self.y0)**2
      if dmin > dmaj:       # Then swap axes
         self.xmajor, self.xminor = self.xminor, self.xmajor
         self.ymajor, self.yminor = self.yminor, self.ymajor
      """
      d = (min * cosA)**2 + (maj * sinA)**2
      r = numpy.sqrt( (maj**2 * min**2)/d )
      Xell = r * cosA
      Yell = r * sinA
      # Rotate
      Xr = Xell * cosP - Yell * sinP + self.x0
      Yr = Xell * sinP + Yell * cosP + self.y0
      vertices = zip(Xr,Yr)
      return vertices


   def movemarker(self, x, y, indx):
      # indx is a dummy variable. One cannot move a single point in an ellipse,
      # instead, the new position is used to reshape the ellipse 
      pa = numpy.arctan2(y-self.y0, x-self.x0) * 180.0/numpy.pi
      D = abs(self.pa-pa)
      axis = numpy.sqrt((self.x0-x)**2 + (self.y0-y)**2)
      if 45.0 <= D < 135.0:
         #self.pa = pa + 90.0
         self.min = axis
      elif 135.0 <= D < 225.0:
         self.pa = pa + 180.0
         self.maj = axis
      elif 225.0 <= D < 315.0:
         #self.pa = pa + 270.0
         self.min = axis
      else:
         self.pa = pa
         self.maj = axis
      self.xy = self.getvertices()
      self.markers.set_data(zip(*self.xy))


   def copy(self, frame, framenr, x0, y0, xy, active, markers):
      # First a real copy. The values for the axes and angle are dummy
      # The position x0, y0 is the New central position
      newobj = Ellipse(frame, framenr, active, markers, x0, y0, type=self.shapetype, 
                       r1=self.maj, r2=self.min, r3=self.pa, **self.kwargs)
      # Then update vertices with 'updatexy which also estimates shape paramaters
      newobj.updatexy(xy)
      return newobj


   def updatexy(self, xy):
      self.xy = xy
      self.markers.set_data(zip(*self.xy))
      # We updated the ellipse for the new vertices, but what about the the new
      # parameters of this ellipse? In fact an ellipse that is transformed for
      # another image is usually not an ellipse anymore.
      # So we need to estimate the new parameters.
      # We require that the center position (x0, y0) remains constant
      x, y = zip(*self.xy)
      x = numpy.asarray(x); y = numpy.asarray(y)
      # Assume symmetry and estimate ellipse parameters.
      # The indices are based on a sampling of 360 samples.
      n90 = len(xy)/4
      self.maj = numpy.sqrt((x[0]-self.x0)**2 +(y[0]-self.y0)**2)
      self.min = numpy.sqrt((x[n90]-self.x0)**2 +(y[n90]-self.y0)**2)
      self.pa = numpy.arctan2(y[0]-self.y0, x[0]-self.x0) * 180.0/numpy.pi 


class Circle(Poly):
   def __init__(self, frame, framenr, active, markers, 
                x0, y0, type=None, r1=0.0, r2=0.0, **kwargs):
      # r1, r2 are extra parameters. For a default ellipse these represent the major cq. minor axes.
      startang, endang, delta = (0.0, 360.0, 1.0)
      self.phi = numpy.arange(startang, endang+delta, delta) * numpy.pi/180.0
      self.x0 = x0
      self.y0 = y0
      self.radius = r1
      Poly.__init__(self, frame, framenr, active, markers, x0, y0, type=type, acolor='g', **kwargs)
      self.updatexy(self.getvertices())


   def getvertices(self):
      vertices = zip(self.radius*numpy.cos(self.phi)+self.x0, self.radius*numpy.sin(self.phi)+self.y0)
      return vertices


   def movemarker(self, x, y, indx):
      self.radius = numpy.sqrt((self.x0-x)**2 + (self.y0-y)**2)
      self.xy = self.getvertices()
      self.markers.set_data(zip(*self.xy))


   def copy(self, frame, framenr, x0, y0, xy, active, markers):
      """ Create a copy of the current object.
      """
      newobj = Circle(frame, framenr, active, markers, x0, y0, type=self.shapetype, 
                      r1=self.radius, **self.kwargs)
      newobj.updatexy(xy)
      return newobj


   def updatexy(self, xy):
      # Get vertices from another object. Usually these are transformed pixel positions.
      self.xy = xy
      self.markers.set_data(zip(*self.xy))
      # Now estimate the radius
      xr, yr = xy[0]
      self.radius = numpy.sqrt((self.x0-xr)**2 + (self.y0-yr)**2)


class Rectangle(Poly):
   def __init__(self, frame, framenr, active, markers,
                x0, y0, type=None, r1=0.0, r2=0.0, **kwargs):
      # r1, r2 are extra parameters. For a default rectangle these represent 
      # width and the height
      self.x0 = x0
      self.y0 = y0
      self.width = r1
      self.height = r2
      Poly.__init__(self, frame, framenr, active, markers, x0, y0, type=type, acolor='r', **kwargs)
      self.updatexy(self.getvertices())


   def getvertices(self):
      x = [0.0, 0.0, 0.0, 0.0]
      y = [0.0, 0.0, 0.0, 0.0]
      # Rectangle counter clockwise. Start at lower left edge
      x[0] = self.x0 - 0.5*self.width
      y[0] = self.y0 - 0.5*self.height
      x[1] = x[0] + self.width
      y[1] = y[0]
      x[2] = x[1]
      y[2] = y[1] + self.height
      x[3] = x[0]
      y[3] = y[2]
      vertices = zip(x,y)
      return vertices


   def movemarker(self, x2, y2, indx):
      x, y = zip(*self.xy)
      dx = (x2 - x[indx])
      dy = (y2 - y[indx])
      self.x0 += dx/2.0
      self.y0 += dy/2.0
      self.width = 2.0*abs(x2-self.x0)
      self.height = 2.0*abs(y2-self.y0)
      self.xy = self.getvertices()
      self.markers.set_data(zip(*self.xy))


   def copy(self, frame, framenr, x0, y0, xy, active, markers):
      """ User wants a copy of the current active object centered at 
      position x0, y0. The vertices of the new object are in parameter xy.
      Note that these vertices could be transformed pixel positions.
      So possible we have different parameters for the rectangle
      (width, height in pixels).
      """
      newobj = Rectangle(frame, framenr, active, markers, self.x0, self.y0, type=self.shapetype,
                         r1=self.width, r2=self.height, **self.kwargs)
      newobj.updatexy(xy)
      return newobj


   def updatexy(self, xy):
      # Get vertices from another object. Usually these are transformed pixel positions.
      # e.g. the active object has vertices in pixels. These are transformed
      # to world coordinates. Then for a different image, these wcs coordinates are 
      # transformed to pixels again. So with different systems, the flux objects
      # do not maintain their shape. Therefore the new properties of the rectangle
      # are estimates only. It allows a user to propagate a shape exactly in world coordinates
      # and the choice of the exact object is in an arbitrary image (i.e. the image of the 
      # current active object)
      self.xy = xy
      x, y = zip(*self.xy)
      self.markers.set_data(x,y)
      # Now estimate the new width, height and center
      # Note that a rectangle in one image can have arbitrary shape in another image
      # depending on a possible wcs transformation.
      self.width = x[1] - x[0]
      self.height = y[2] - y[1]
      self.x0 = (x[1] + x[0])/2.0
      self.y0 = (y[3] + y[0])/2.0
   
      
class Spline(Poly):
   def __init__(self, frame, framenr, active, markers,
                x0, y0, type=None, r1=0.0, r2=0.0, **kwargs):
      Poly.__init__(self, frame, framenr, active, markers, x0, y0, type=type, acolor='r', 
                    spline=True, **kwargs)

   def copy(self, frame, framenr, x0, y0, xy, active, markers):
      newobj = Spline(frame, framenr, active, markers, x0, y0, type=self.shapetype, **self.kwargs)
      newobj.updatexy(xy)
      return newobj


      
class Shapecollection(object):
   #-----------------------------------------------------------------
   """
   Administration class for a collection of shapes.
   The figure 

   :param images:  In each image a shape can be drawn using mouse-
                   and keyboard buttons. This shape is duplicated
                   either in pixel coordinates or world coordinates in
                   the other images of the list with images.
                   These images have two attributes that are relevant for
                   this module. These are *fluxfie* to define how the
                   flux should be calculated using fixed variables
                   *s* for the sum of the intensities of the pixels
                   in an area and *a* which represents the area.
   :type images:   A list of objects from class :class:`maputils.Annotatedimage`

   :param ifigure: The Matplotlib figure where the images are.
   :type ifigure:  Matplotlib :class:`Figure` object
   
   :param wcs:     The default is *True* which implies that in case of
                   multiple images shapes propagate through world
                   coordinates. If you have images with the same
                   size and WCS, then set *wcs=False* to
                   duplicate shapes in pixel coordinates which is
                   much faster.
   :type wcs:      Boolean
   
   :param inputfilename:
                   Name of file on disk which stores shape information.
                   The objects are read from this file and plotted on
                   all the images in the image list. The coordinates
                   in the file can be either pixel- or world coordinates.
                   You should specify that with parameter *inputwcs*
   :type inputfilename:
                   String

   :param inputwcs: 
                   Set the shape mode for shapes from file to
                   either pixels coordinates (*inputwcs=False*)
                   or to world coordinates (*inputwcs=True*).
   :type inputwcs: Boolean


   This shape interactor reacts to the following keyboard and mouse buttons::

      mouse - left  :  Drag a polygon point to a new position or
                       change the radius of a circle or
                       change the minor axis of an ellipse or
                       change the major axis and position angle of an ellipse
      mouse - middle:  Select an existing object in any frame
      key   - a     :  Add a point to a polygon or spline
      key   - c     :  Copy current object at mouse cursor
      key   - d     :  Delete a point in a polygon or spline
      key   - e     :  Erase active object and associated objects in other images
      key   - i     :  Insert a point in a polygon or spline
      key   - n     :  Start with a new object
      key   - w     :  Write object data in current image to file on disk
      key   - r     :  Read objects from file for current image
      key   - [     :  Next active object in current shape selection
      key   - ]     :  Previous active object in current shape selection
      
      Interactive navigation defined by canvas
      Amongst others:
      key   - f     :  Toggle fullscreen
      key   - g     :  Toggle grid


      Gui buttons:
      'Quit'         :  Abort program
      'plot result'  :  Plot calculated flux as function of shape and image
      'Save result'  :  Save flux information to disk
                        The file names are generated and contain date
                        and time stamp (e.g flux_24042010_212029.dat)
      'Pol.'         :  Select shape polygon. Start with key 'n' for
                        new polygon. Add new points with key 'a'.
      'Ell.'         :  Select shape ellipse. Start with key 'n' for
                        new ellipse. With left mouse button Drag major axis to change
                        size and rotation or, using a point near the
                        center, drag entire ellipse to a new position.
      'Cir.:'        :  Select shape circle. Start with key 'n' for
                        new circle. The radius can be changed by dragging
                        an arbitrary point on the border to a new position.
      'Rec.'         :  Select shape rectangle. Start with key 'n' for
                        new rectangle. Drag any of the four edges to resize
                        the rectangle.
      'Spl.'         :  Like the polygon but the points between two knots
                        follow a spline curve.

   :Notes:

      All shapes are derived from a polygon class. There is one method
      that generates coordinates for all shapes and :meth:`maputils.getflux`
      uses the same routine to calculate whether a pixel in an enclosing
      box is within or outside the shape. For circles and ellipses the
      number of polygon points is 360 and this slows down the calculation
      significantly. Methods which assume a perfect circle or ellipse can
      handle the inside/outside problem much faster, but note that due to different
      WCS's, ellipses and circles don't keep their shape in other images.
      So in fact only a polygon is the common shape. A spline is a polygon
      with an artificially increased number of points.

   :Example:
   
     ::

      fig = plt.figure(figsize=(12,10))
      frame1 = fig.add_axes([0.07,0.1,0.35, 0.8])
      frame2 = fig.add_axes([0.5,0.1,0.43, 0.8])
      im1 = f1.Annotatedimage(frame1)
      im2 = f2.Annotatedimage(frame2)
      im1.Image(); im1.Graticule()
      im2.Image(); im2.Graticule()
      im1.interact_imagecolors(); im1.interact_toolbarinfo()
      im2.interact_imagecolors(); im2.interact_toolbarinfo()
      im1.plot(); im2.plot()
      im1.fluxfie = lambda s, a: s/a
      im2.fluxfie = lambda s, a: s/a
      im1.pixelstep = 0.5; im2.pixelstep = 0.5
      images = [im1, im2]
      shapes = shapes.Shapecollection(images, fig, wcs=True, inputwcs=True)

   """
   #-----------------------------------------------------------------
   def __init__(self, images, ifigure, wcs=True,
                inputfilename=None, inputwcs=False, gipsy=False):
      self.frames = []
      self.images = images
      self.numberofimages = len(images)
      self.inputfilename = inputfilename
      self.inputwcs = inputwcs
      self.gipsy = gipsy
      self.wcs = wcs
      self.activeobject = None
      self.currenttype = 0
      self.canvas = ifigure.canvas
      self.numberoftypes = 5
      self.shapetypes = [self.polygon, self.ellipse, self.circle, self.rectangle, self.spline] = range(self.numberoftypes)
      # Extend the shape types by adding a variable and the number in self.numberoftypes 
      self.maxindx = [-1]*self.numberoftypes  # Initialize index for object sequences to -1
      # A shape is repeated in all images.  For this group of shapes we reserve an index
      self.currentobj = [-1]*self.numberoftypes
      self.shapes = [[]]*self.numberoftypes   # A  list with lists for each shape type
      # Order of elements: shapes[shapetype][currentobj][currentimage]
      self.currentimage = None
      self.canvas.mpl_connect('key_press_event', self.key_pressed_global)
      self.cidmove = ifigure.canvas.mpl_connect('motion_notify_event', self.motion_notify)
      # Here it seems that the order of connecting callbacks is important.
      # The backend.bases.py module defines its own toolbar message for mouse motions
      # if we connect the motion_notify_event after the button_press_event, this backend
      # callback takes precedence over the one we defined in this module.
      # If you don't see a custom toolbar message then this could be a starting point
      # to search for a solution
      self.cidpress = self.canvas.mpl_connect('button_press_event', self.button_press)
      self.cidrelease = self.canvas.mpl_connect('button_release_event', self.button_release)
      self.toolbar = get_current_fig_manager().toolbar
      for im in self.images:           # Separate the frames from the images
         self.frames.append(im.frame)
      self.figure = ifigure
      self.shapedict = {'pol':self.polygon, 'ell':self.ellipse, 'cir':self.circle, 'rec':self.rectangle, 'spl':self.spline}
      # Define some buttons
      #axcolor = 'lightgoldenrodyellow'
      #shaps = self.figure.add_axes([0.93, 0.85, 0.06, 0.14])
      #radio = RadioButtons(shaps, ('pol', 'ell', 'cir', 'rec', 'spl'))
      #radio.on_clicked(self.setshape)
      self.figresult = figure(figsize=(6,5))
      self.frameresult = self.figresult.add_subplot(1,1,1)
      self.frameresult.set_title("Flux as function of shape and image")
      self.results = False

      self.graycol = 'chartreuse'
      quit_button = self.figure.add_axes([0.01, 0.94, 0.11, 0.05])
      b0 = Button(quit_button, 'Quit')
      b0.on_clicked(self.doquit)

      result_button = self.figure.add_axes([0.13, 0.94, 0.11, 0.05])
      b1 = Button(result_button, 'Plot Flux')
      b1.on_clicked(self.plotresults)

      save_button = self.figure.add_axes([0.25, 0.94, 0.11, 0.05])
      b2 = Button(save_button, 'Save Flux')
      b2.on_clicked(self.saveresults)

      pol_button = self.figure.add_axes([0.74, 0.94, 0.05, 0.05])
      b3 = Button(pol_button, 'Pol.')
      b3.on_clicked(self.setpoly)

      ell_button = self.figure.add_axes([0.79, 0.94, 0.05, 0.05])
      b4 = Button(ell_button, 'Ell.')
      b4.on_clicked(self.setellipse)

      cir_button = self.figure.add_axes([0.84, 0.94, 0.05, 0.05])
      b5 = Button(cir_button, 'Cir.')
      b5.on_clicked(self.setcircle)

      rec_button = self.figure.add_axes([0.89, 0.94, 0.05, 0.05])
      b6 = Button(rec_button, 'Rec.')
      b6.on_clicked(self.setrectangle)

      spl_button = self.figure.add_axes([0.94, 0.94, 0.05, 0.05])
      b7 = Button(spl_button, 'Spl.')
      b7.on_clicked(self.setspline)

      self.buttons = [b0, b1, b2, b3, b4, b5, b6, b7]
      for i, b in enumerate(self.buttons):
         if i > 3:
            b.setcolor(self.graycol)
         if i == 3:
            b.setcolor('r')
         b.label.set_fontsize(10)
         
      tdict = dict(color='g', fontsize=10, va='bottom', ha='left')
      helptxt = "SHAPES:\n"
      helptxt += "n=start new object -- a=add point -- d=delete point -- i=insert point\n"
      helptxt += "c=copy object -- e=erase object -- [=next shape in group -- ]=prev in gr.\n"
      helptxt += "w=write shapes to disk -- r=read shapes from disk\n"
      helptxt += "Mouse-left=drag and/or change shape --- Mouse-middle=select shape"
      
      ifigure.text(0.01,0.01, helptxt, tdict)
      helptxt  = "COLOURS:\n"
      """helptxt += "Page-up=change colour map -- page-down=change colour map\n"
      helptxt += "Key 1=linear, 2=log, 3=exp, 4=sqrt, 5=square 9=inverse, 0=reset\n"
      helptxt += "b=change colour bad pixels -- m=save colour map to disk\n"
      helptxt += "Toggle keys h=histogram equalization -- z=smooth"
      """
      helptxt += self.images[0].get_colornavigation_info()
      ifigure.text(0.5,0.01, helptxt, tdict)
      
      """
      print helptxt
      stdout.flush()
      """

   def setpoly(self, event):
      for i in range(3,len(self.buttons)):
         self.buttons[i].setcolor(self.graycol)
      self.buttons[3].setcolor('r')
      self.setshape('pol')
   def setellipse(self, event):
      for i in range(3,len(self.buttons)):
         self.buttons[i].setcolor(self.graycol)
      self.buttons[4].setcolor('r')
      self.setshape('ell')
   def setcircle(self, event):
      for i in range(3,len(self.buttons)):
         self.buttons[i].setcolor(self.graycol)
      self.buttons[5].setcolor('r')
      self.setshape('cir')
   def setrectangle(self, event):
      for i in range(3,len(self.buttons)):
         self.buttons[i].setcolor(self.graycol)
      self.buttons[6].setcolor('r')
      self.setshape('rec')
   def setspline(self, event):
      for i in range(3,len(self.buttons)):
         self.buttons[i].setcolor(self.graycol)
      self.buttons[7].setcolor('r')
      self.setshape('spl')

      
   def setshape(self, label):
   #----------------------------------------------------------
      """
      This is the callback function for the radio button with
      the selection of shapes.
      """
   #----------------------------------------------------------
      newtype = self.shapedict[label]
      if newtype < 0 or newtype >= len(self.shapetypes):   # Does not represent a shape
         return
      oldtype = self.currenttype
      if oldtype == newtype:
         return                                            # Nothing changed
      oldindx = self.currentobj[oldtype]
      newindx = self.currentobj[newtype]                   # Select a new group of shapes
      self.currenttype = newtype
      if self.maxindx[oldtype] >= 0:
         for obj in self.shapes[oldtype][oldindx]:         # Make current group of objects inactive
            if obj.active:
               obj.set_inactive()
      if (self.activeobject):
         frame = self.activeobject.frame
      else:
         frame = None
      if self.maxindx[newtype] >= 0:
         for obj in self.shapes[newtype][newindx]:         # Make the objects of the current shape active
            if obj.frame is frame:
               self.activeobject = obj
               obj.set_active(markers=True)
            else:
               obj.set_active()
      self.canvas.draw()

   
   def getimage(self, event):
      image = None
      for ima in self.images:
         #if ima.frame is frame:
         if ima.frame.contains(event)[0]:
            image = ima
            break
      return image


   def getframenr(self, event):
      nr = None
      for i, fr in enumerate(self.frames):
         #if fr is frame:
         if fr.contains(event)[0]:
            nr = i
            break
      return nr


   def updatesplines(self):
      if self.activeobject == None:
         return
      xy = cubicspline(self.activeobject.xy, 10)
      if xy != None and self.activeobject.spline != None:
         self.activeobject.spline.xy = xy
         cindx = self.currentobj[self.currenttype]
         for obj in self.shapes[self.currenttype][cindx]:
            if not (obj is self.activeobject):
               if self.wcs:
                  proj1 = self.currentimage.projection
                  proj2 = self.images[obj.framenr].projection
                  obj.spline.xy = self.transformXY(self.activeobject.spline.xy, proj1, proj2)
               else:
                  obj.spline.xy = self.activeobject.spline.xy



   def transformXY(self, xy1, proj1, proj2):
   #----------------------------------------------------------
      """
      Given one or a sequence of positions in pixels *xy* that belong
      to an image with world coordinate system *proj1*, we want
      the pixel values in another world coordinate system given by
      *proj2*. This second projection can differ in output sky system.
      We follow the next procedure to transform:

      * If the sky systems are equal then we can transform
        the pixel coordinates without a sky transformation.
      * If the sky systems differ, transform the pixel position
        in world coordinates in the first system with the
        sky system of the second system.
      * Transform the world coordinates into pixels in the
        second system. These world coordinates are now given
        in the sky system of the second world coordinate system. 
      """
   #----------------------------------------------------------
      if proj1.skyout != proj2.skyout:
         proj1.skyout = proj2.skyout
   
      xyworld1 =  proj1.toworld(xy1)
      xy2 = proj2.topixel(xyworld1)
      stdout.flush()
      proj1.skyout = None                           # Reset
      return xy2


   def addnewobject(self, shapetype, x, y, framenr):
      # Add a new shape and copy to all images
      oldgroup = self.maxindx[self.currenttype]
      if oldgroup >= 0:
         for obj in self.shapes[self.currenttype][oldgroup]:
            if obj.active:
               obj.set_inactive()
      self.currenttype = shapetype
      self.maxindx[self.currenttype] += 1                  # Increase index because we add an object
      if self.activeobject:
         self.activeobject.set_markers(False)
      objlist = []                                         # List with copy of flux object for each image
      obj = None;
      currentframe = self.frames[framenr]
      x1, x2 = currentframe.get_xlim()
      y1, y2 = currentframe.get_ylim()
      maj = abs(x2-x1)/5.0
      min = abs(y2-y1)/8.0
      baseobj = None
      # Make single and copy this later for other images
      active = markers = True
      xpb = ypb = 0.0
      if self.currenttype == self.ellipse:
         baseobj = Ellipse(currentframe, framenr, active, markers,
                     xpb, ypb, type=self.currenttype, r1=maj, r2=min)
      elif self.currenttype == self.polygon:
         baseobj = Poly(currentframe, framenr, active, markers,
                     xpb, ypb, type=self.currenttype)
      elif self.currenttype == self.circle:
         baseobj = Circle(currentframe, framenr, active, markers,
                     xpb, ypb, type=self.currenttype, r1=min)
      elif self.currenttype == self.rectangle:
         baseobj = Rectangle(currentframe, framenr, active, markers,
                     xpb, ypb, type=self.currenttype, r1=maj, r2=min)
      elif self.currenttype == self.spline:
         baseobj = Spline(currentframe, framenr, active, markers,
                     xpb, ypb, type=self.currenttype, r1=maj, r2=min)
      baseobj.updatexy(zip(x, y))                # The position to start with
      if self.wcs:
         xyworld = self.images[framenr].projection.toworld(baseobj.xy)
      proj1 = self.images[framenr].projection

      for i in range(self.numberofimages):
         active = True
         markers = False
         if i != baseobj.framenr:
            if self.wcs:
               proj2 = self.images[i].projection
               xy = self.transformXY(baseobj.xy, proj1, proj2)
            else:
               xy = baseobj.xy
            obj = baseobj.copy(self.frames[i], i, xpb, ypb, xy, active, markers)
         else:
            obj = baseobj
            obj.set_active()
         objlist.append(obj)

      if len(objlist) > 0:
         numlists = len(self.shapes[self.currenttype]) #'shapes' is a list of lists of objects
         if numlists == 0:                             # List is still empty
            self.shapes[self.currenttype] = [objlist]
         else:
            self.shapes[self.currenttype].append(objlist)
         self.currentobj[self.currenttype] = numlists

      self.activeobject = baseobj
      self.activeobject.set_markers(True)
      if self.currenttype == self.spline:
         self.updatesplines()
      self.canvas.draw()
   
   
   def key_pressed_global(self, event):
      """This is the event handler for all types of supported shapes for which we want
         statistics"""
      if not event.inaxes:
         return
      if self.toolbar.mode != '':                             # Do nothing while in pan or zoom mode
         return
      self.currentimage = self.getimage(event)
      if self.currentimage == None:
         return
      xpb = event.xdata; ypb = event.ydata
      if self.wcs:
         xw, yw = self.currentimage.projection.toworld((xpb,ypb))
      
      if event.key.isdigit():
         """User pressed a number 1, 2, 3, .. (i.e. it does not start with 0)
            This number corresponds to one of the supported types"""
         newtype = int(event.key) - 1                         # Bring user selected type in range 0..number of shape types -1
         if newtype < 0 or newtype >= len(self.shapetypes):   # Does not represent a shape
            return
         oldtype = self.currenttype
         if oldtype == newtype:
            return                                            # Nothing changed
         oldindx = self.currentobj[oldtype]
         newindx = self.currentobj[newtype]                   # Select a new group of shapes
         self.currenttype = newtype
         if self.maxindx[oldtype] >= 0:
            for obj in self.shapes[oldtype][oldindx]:         # Make current group of objects inactive
               if obj.active:
                  obj.set_inactive()
         if self.maxindx[newtype] >= 0:
            for obj in self.shapes[newtype][newindx]:         # Make the objects of the current shape active
               if obj.frame.contains(event)[0]:
                  self.activeobject = obj
                  obj.set_active(markers=True)
               else:
                  obj.set_active()
         self.canvas.draw()

      elif event.key in ['n', 'c']:
         if event.key == 'c':
            if not self.activeobject:       # User want a copy but from what?
               return
            if not self.activeobject.frame.contains(event)[0]:
               return                       # One can only copy in frame of active object 
            if self.activeobject.shapetype != self.currenttype:
               return                       # Last object was of other type. Cannot copy.

         # Add a new shape and copy to all images
         oldgroup = self.maxindx[self.currenttype]
         if oldgroup >= 0:
            for obj in self.shapes[self.currenttype][oldgroup]:
               if obj.active:
                  obj.set_inactive()
         self.maxindx[self.currenttype] += 1                  # Increase index because we add an object
         if self.activeobject:
            self.activeobject.set_markers(False)
         objlist = []                                         # List with copy of flux object for each image 
         obj = None;
         x1, x2 = event.inaxes.get_xlim()
         y1, y2 = event.inaxes.get_ylim()
         maj = abs(x2-x1)/5.0
         min = abs(y2-y1)/8.0
         framenr = self.getframenr(event)
         baseobj = None
         if event.key == 'n':
            # Make single and copy this later for other images
            active = markers = True
            if self.currenttype == self.ellipse:
               baseobj = Ellipse(event.inaxes, framenr, active, markers, 
                         xpb, ypb, type=self.currenttype, r1=maj, r2=min)
            elif self.currenttype == self.polygon:
               baseobj = Poly(event.inaxes, framenr, active, markers, 
                         xpb, ypb, type=self.currenttype)
            elif self.currenttype == self.circle:
               baseobj = Circle(event.inaxes, framenr, active, markers,
                         xpb, ypb, type=self.currenttype, r1=min)
            elif self.currenttype == self.rectangle:
               baseobj = Rectangle(event.inaxes, framenr, active, markers,
                         xpb, ypb, type=self.currenttype, r1=maj, r2=min)
            elif self.currenttype == self.spline:
               baseobj = Spline(event.inaxes, framenr, active, markers,
                         xpb, ypb, type=self.currenttype, r1=maj, r2=min)
            xyshift = baseobj.xy   # Not shifted here
            if self.wcs:
               xyworld = self.getimage(event).projection.toworld(xyshift)
            
         
         if event.key == 'c':
            # Make single and copy this later for other images
            active = markers = True
            xyshift = self.activeobject.shiftxy(xpb, ypb)
            baseobj = self.activeobject.copy(event.inaxes, framenr, xpb, ypb, xyshift, active, markers)


         for i in range(self.numberofimages):
            active = True
            markers = False
            if not self.frames[i].contains(event)[0]:
               if self.wcs:
                  # NOTE: The current system assumes that the images for which
                  # the shapes should be copied, have all the same sky system.
                  # If we want to extend this to different systems we need a
                  # sky transformation first (using the skysystem of the other image)
                  proj1 = self.images[framenr].projection
                  proj2 = self.images[i].projection
                  xy = self.transformXY(xyshift, proj1, proj2)
                  x0, y0 = self.transformXY((xpb, ypb), proj1, proj2)
               else:
                  xy = xyshift
                  x0, y0 = xpb, ypb
               obj = baseobj.copy(self.frames[i], i, x0, y0, xy, active, markers)
            else:
               obj = baseobj
               obj.set_active()
            objlist.append(obj)
            
         if len(objlist) > 0:
            numlists = len(self.shapes[self.currenttype]) #'shapes' is a list of lists of objects
            if numlists == 0:                             # List is still empty
               self.shapes[self.currenttype] = [objlist]
            else:
               self.shapes[self.currenttype].append(objlist)
            self.currentobj[self.currenttype] = numlists
         self.activeobject = baseobj
         self.activeobject.set_markers(True)
         if event.key == 'c' and self.currenttype == self.spline:
            self.updatesplines()
         self.canvas.draw()


      elif event.key in ['[', ']']:
         # Change active object in current group (only)
         newgroup = oldgroup = self.currentobj[self.currenttype]
         if oldgroup == -1:
            return     # Nothing to do, no object yet available
         if event.key == '[':
            newgroup += 1
            if newgroup > self.maxindx[self.currenttype]:
               newgroup = 0
         else:
            newgroup -= 1
            if newgroup < 0:
               newgroup = self.maxindx[self.currenttype]
         if newgroup == oldgroup:
            return
         if self.activeobject:
            self.activeobject.set_markers(False)
         for obj in self.shapes[self.currenttype][oldgroup]:
            if obj.active:
               obj.set_inactive()
         # Make the objects of the current shape active
         for obj in self.shapes[self.currenttype][newgroup]:
            obj.set_active()
            if obj.frame.contains(event)[0]:
               self.activeobject = obj
               self.activeobject.set_markers(True)
         self.currentobj[self.currenttype] = newgroup
         self.canvas.draw()
      
      elif event.key == 'a':
         """
         Add a vertex for a polygon or spline. For the current
         object the spline interpolation points are calculated and
         they are transformed to pixel coordinates in other images.
         So for transformations between world coordinate systems,
         each spline does not get its own interpolation points.
         but transforms the interpolated positions from the
         active object.
         """
         if self.activeobject:
            if not self.activeobject.frame.contains(event)[0]:
               return
         cindx = self.currentobj[self.currenttype]
         if cindx >= 0 and self.currenttype in (self.polygon, self.spline):
            framenr = self.getframenr(event)
            self.activeobject.addvertex(xpb, ypb, True)
            for obj in self.shapes[self.currenttype][cindx]:
               if not obj is self.activeobject:
                  setmarker = False
                  if self.wcs:
                     proj1 = self.images[framenr].projection
                     proj2 = self.images[obj.framenr].projection
                     xp, yp = self.transformXY((xpb,ypb), proj1, proj2)
                  else:
                     xp, yp = xpb, ypb
                  obj.addvertex(xp, yp, setmarker)
            if self.currenttype == self.spline:
               self.updatesplines()
            self.canvas.draw()


      elif event.key == 'd':
         # Delete the closest marker in all images
         if self.activeobject == None:
            return                                  # Nothing to do
         indx = self.activeobject.indexclosestmarker(event.x, event.y)
         if indx == None:
            return                                  # Could not find a closest point
         cindx = self.currentobj[self.currenttype]
         if cindx >= 0 and self.currenttype in [self.polygon, self.spline]:
            for obj in self.shapes[self.currenttype][cindx]:
               obj.deletemarker(indx)
            if self.currenttype == self.spline:
               self.updatesplines()
            self.canvas.draw()
            
      elif event.key == 'i':
         if self.activeobject == None:
            return           # Nothing to do
         indx = self.activeobject.indexsegmentinrange(event.x, event.y)
         cindx = self.currentobj[self.currenttype]
         if cindx >= 0 and self.currenttype  in [self.polygon, self.spline]:
            framenr = self.getframenr(event)
            for obj in self.shapes[self.currenttype][cindx]:
               if obj == self.activeobject:
                  xp, yp = xpb, ypb
               else:
                  proj1 = self.images[framenr].projection
                  proj2 = self.images[obj.framenr].projection
                  xp, yp = self.transformXY((xpb,ypb), proj1, proj2)
               obj.insertmarker(xp, yp, indx)
            if self.currenttype == self.spline:
               self.updatesplines()
            self.canvas.draw()
            
      elif event.key == 'e':
         # Erase current group of objects
         oldgroup = self.currentobj[self.currenttype]
         if oldgroup < 0:
            return
         #if self.currenttype in [self.polygon, self.ellipse]:
         for obj in self.shapes[self.currenttype][oldgroup]:
            obj.delete()
         del self.shapes[self.currenttype][oldgroup]
         self.maxindx[self.currenttype] -= 1
         if self.maxindx[self.currenttype] < 0:
            # Cannot change to another group because there are no other groups
            self.currentobj[self.currenttype] = -1
            newgroup = None
            for sh in self.shapetypes:
               if sh != self.currenttype:
                  if self.currentobj[sh] != -1:
                     self.currenttype = sh
                     newgroup = self.currentobj[sh]
                     break
            if newgroup == None:
               self.activeobject = None
               self.canvas.draw()
         else:
            newgroup = oldgroup - 1
         if newgroup < 0:
            newgroup = self.maxindx[self.currenttype]
         self.currentobj[self.currenttype] = newgroup
         if newgroup >= 0:
            for obj in self.shapes[self.currenttype][newgroup]:
               obj.set_active()
               if obj.frame.contains(event)[0]:
                  self.activeobject = obj
                  self.activeobject.set_markers(True)
         self.canvas.draw()

      elif event.key == 'w':
         # All markers of current image to one file. The first column sets the number of
         # the polygon it belongs. The format is:
         # polygon-number  x-pixels  y-pixels  x-wcs  y-wcs
         stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
         filename = "shapes_" + stamp + ".dat"
         f = open(filename, 'w')
         stamp = datetime.now().strftime("! Saved at %A %d/%m/%Y %H:%M:%S\n")
         f.write(stamp)
         iminfo = "! Data saved for image: %s\n" % self.currentimage.basename
         f.write(iminfo)
         f.write('! Format: Shape - object number - x pixel - y pixel - x world - y world\n')
         for sh in self.shapes:
            for ol, objlist in enumerate(sh):
               for obj in objlist:
                  #if obj.frame is self.currentimage.frame:
                  if self.framesequal(obj.frame, self.currentimage.frame):
                     im = self.images[obj.framenr]
                     for mx, my in obj.xy:   # Note that this is also valid for splines
                        if (im.mixpix == None):
                           xw, yw = im.projection.toworld((mx, my))
                        else:
                           xw, yw, dum = im.projection.toworld((mx, my, im.mixpix))
                        f.write('%d %d %12f %12f %12f %12f\n' % (obj.shapetype, ol, mx, my, xw, yw))
         f.close()
         mes = "Wrote shape data to file: %s" % filename
         self.toolbar.set_message(mes)


      elif event.key == 'r':
         # Read data from file. Select between pixel- or world
         # coordinates. First number in the file is an index for
         # the polygon the data belongs to.
         if self.inputfilename == None:
            self.toolbar.set_message("No input file name specified!")
            return
         im = self.currentimage
         framenr = self.getframenr(event)
         t = tabarray.tabarray(self.inputfilename)
         if not self.inputwcs:
            shapenr, polynr, x, y = t.columns((0, 1, 2, 3))
         else:
            shapenr, polynr, xw, yw = t.columns((0, 1, 4, 5))
            if (im.mixpix == None):
               x, y = im.projection.topixel((xw, yw))
            else:
               x, y, dum = im.projection.topixel((xw, yw, im.mixpix))
         smax = shapenr.max()
         omax = polynr.max()
         for sh in range(self.numberoftypes):
            for ob in range(int(omax)+1):      # Number of object is unknown
               xlist = []
               ylist = []
               for s, o, x1, y1 in zip(shapenr, polynr, x,y):
                  if s == sh and o == ob:
                     xlist.append(x1)
                     ylist.append(y1)
               if len(xlist):
                  self.addnewobject(sh, xlist, ylist, framenr) 
         mes = "Read data from file %s" % self.inputfilename
         self.toolbar.set_message(mes)

   def button_press(self, event):
      # -A position pointed with the mouse could be within a polygon
      # Then activate that polygon and the associated polygons (group)
      if not event.inaxes: 
         return
      if self.toolbar.mode != '':         # Must be in zoom or pan mode, so do nothing
         return
      if not self.activeobject:           # There is not an object selected to have interaction with
         return
      if event.button == 2:               # Middle button
         for i, fr in enumerate(self.frames):
            if fr.contains(event)[0]:
               currentframe = i
               break
         x = event.xdata; y = event.ydata
         newgroup = None
         oldgroup = self.currentobj[self.currenttype]
         oldtype = self.currenttype
         self.activeobject.closestindx = self.activeobject.indexclosestmarker(event.x, event.y)
         oldactiveobject = self.activeobject
         for sh in self.shapes:
            # Loop over all shape types
            for group, objlist in enumerate(sh):
               obj = objlist[currentframe]
               if obj.inside(x, y):
                  newgroup = group
                  newtype = obj.shapetype
                  self.activeobject = obj
                  break
            if newgroup != None:
              break
         if newgroup == None:
            return

         for obj in self.shapes[oldtype][oldgroup]:
            if obj.active:
               obj.set_inactive()
         for obj in self.shapes[newtype][newgroup]:
            markers = obj == self.activeobject
            obj.set_active(markers)
         self.currentobj[newtype] = newgroup
         self.currenttype = newtype
         self.canvas.draw()

      if event.button == 1:
         self.activeobject.closestindx = self.activeobject.indexclosestmarker(event.x, event.y)
         if self.activeobject.closestindx == None:
            if self.activeobject.inside(event.xdata, event.ydata):
               self.activeobject.closestindx = -1             # Not a marker, index can be used to move all markers
         return          # Nothing changed




   def motion_notify(self, event):
      """ Move one marker or the whole object"""
      if not event.inaxes: 
         return
      if self.toolbar.mode != '':
         return
      currentimage = self.getimage(event)
      if not self.activeobject:
         return

      # Remember: the shapes array is shapes[shapetype][currentobj][currentimage]
      # If we are in a different frame than the one of the active object
      # then we switch to the corresponding object in the current image.
      # 
      if not self.activeobject.frame.contains(event)[0]:
         group = self.currentobj[self.currenttype]
         if group < 0:
            return
         framenr = self.getframenr(event)
         if framenr == None:                     # Perhaps a frame of a button etc.
            return
         self.activeobject.set_inactive()
         self.activeobject = self.shapes[self.currenttype][group][framenr]
         self.activeobject.set_active(markers=True)
         self.canvas.draw()
         return
      
      self.currentimage = self.getimage(event)
      if self.currentimage == None:
         return

      if event.button == 1:
         xp = event.xdata; yp = event.ydata
         group = self.currentobj[self.currenttype]
         indx = self.activeobject.closestindx
         # currentimage = self.activeobject.image
         if indx != None and indx != -1:
            self.activeobject.movemarker(xp, yp, indx)
         #if self.wcs:
            #xw, yw = self.currentimage.projection.toworld((xp,yp))
            #if self.currenttype != self.polygon:
               #xyworld = self.images[self.activeobject.framenr].projection.toworld(self.activeobject.xy)
         if group < 0:
            return
         for obj in self.shapes[self.currenttype][group]:        # Copy to other images
            if self.wcs:                                         # Is a transformation needed?
               proj1 = self.currentimage.projection
               proj2 = self.images[obj.framenr].projection
               xp, yp = self.transformXY((event.xdata,event.ydata), proj1, proj2)
            if indx == -1:
               # This was a position inside the polygon but not close enough to a marker.
               # Then move all markers
               obj.moveall(xp, yp)
            elif indx != None and not (obj is self.activeobject): #(obj.frame is self.currentimage.frame):  # Skip the active object
               # A position close enough to a marker was found. 
               # Move this marker if the shape is an irregular polygon type. For other types
               # it has a different meaning.
               if not self.wcs or self.currenttype in (self.polygon, self.spline):
                  obj.movemarker(xp, yp, indx)
               else:
                  proj1 = self.currentimage.projection
                  proj2 = self.images[obj.framenr].projection
                  xy = self.transformXY(self.activeobject.xy, proj1, proj2)
                  obj.updatexy(xy)
         if self.currenttype == self.spline:
            self.updatesplines()
         self.canvas.draw()


   def button_release(self, event):
      if not event.inaxes: 
         return
      if self.toolbar.mode != '':
         return
      if not self.activeobject:           # There is not an object selected to have interaction with
         return
      currentimage = self.getimage(event)
      if currentimage == None:
         return
      if event.button in [1,3]:
         self.activeobject.closestindx = None
      if event.button == 1 and self.currenttype == self.spline:
         self.updatesplines()
         self.canvas.draw()

   def framesequal(self, fr1, fr2):
      if fr1.get_position().x0 != fr2.get_position().x0:
         return False
      if fr1.get_position().x1 != fr2.get_position().x1:
         return False
      if fr1.get_position().y0 != fr2.get_position().y0:
         return False
      if fr1.get_position().y0 != fr2.get_position().y0:
         return False
      return True

   def plotresults(self, event):
      # Plot these values. Both as markers and with connecting lines
      markerlist = ['+' , ',' , '.' , '1' , '2' , '3' , '4', '<' , '>' , 'D' , 'H' , '^' , 'd', 'h' , 'o', 'p' , 's' , 'v' , 'x' , '|']
      # For efficiency we need to calculate properties for all objects in one image
      # and repeat this for all images
      mes = "\nObject properties:"
      if self.gipsy and gipsymod:
         anyout(mes)
      else:
         print mes
      fluxlist = []
      for i, im in enumerate(self.images):
         for sh in self.shapes:
            for ol, objlist in enumerate(sh):
               for obj in objlist:
                  if self.framesequal(obj.frame, im.frame):
                     if obj.allinsideframe():
                        if obj.spline:
                           xy = obj.spline.xy
                        else:
                           xy = obj.xy
                        obj.area, obj.sum = im.getflux(xy)
                        obj.flux = im.fluxfie(obj.sum, obj.area)
                        mes = "Object %d with shape %d in image %d has area=%g, sum=%g, flux=%g" % (ol, obj.shapetype, i, obj.area, obj.sum, obj.flux)
                        if self.gipsy and gipsymod:
                           anyout(mes)
                        else:
                           print mes
                        fluxlist.append(obj.flux)
                     else:
                        mes = "Object %d with shape %d in image %d has pixels outside frame" % (ol, obj.shapetype, i)
                        if self.gipsy and gipsymod:
                           anyout(mes)
                        else:
                           print mes
                        obj.area = obj.sum = obj.flux = None


      # Next we have the freedom to sort the data as we want. For a plot we want
      # to show the properties of each object as function of the image.
      mindx = 0
      frameresult = self.frameresult
      frameresult.clear()
      for sh in self.shapes:
         for objlist in sh:
            x = []; y = []
            for i, obj in enumerate(objlist):
               x.append(i)
               y.append(obj.flux)
            frameresult.plot(x,y, marker=markerlist[mindx], color='r', label=str(mindx))
            frameresult.plot(x,y, '-', color='k')
            mindx += 1
            if mindx == len(markerlist)-1:
               mindx = 0

      if len(fluxlist) == 0:
         self.figresult.canvas.draw()
         self.results = False
         return        # No objects found
      fluxlist  = numpy.asarray(fluxlist)
      ymin = fluxlist.min()
      ymax = fluxlist.max()
      d = (ymax-ymin)/20.0
      frameresult.set_ylim(ymin-d, ymax+d)
      frameresult.set_xlim(-0.5, self.numberofimages-1+0.5)
      frameresult.set_title("Flux as function of shape and image")
      xticks = range(self.numberofimages)  # Only the image numbers as labels
      frameresult.set_xticks(xticks)
      frameresult.set_xlabel("Image number")
      frameresult.set_ylabel("Flux")
      frameresult.legend()
      self.results = True
      self.figresult.canvas.draw()

   def doquit(self, event):
      if self.gipsy and gipsymod:
         finis()
      else:
         exit()     # Exit program

   def saveresults(self, event):
      # Save the flux results to file on disk
      if not self.results:
         self.plotresults(event)
         #print "Calculate results first with button 'plot results'"
         #return
      stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
      filename = "flux_" + stamp + ".dat"
      f = open(filename, 'w')
      stamp = datetime.now().strftime("! Saved at %A %d/%m/%Y %H:%M:%S\n")
      f.write(stamp)
      f.write('! sh: 0=polygon  1=ellipse  2=circle  3=rectangle  4=spline\n')
      f.write('! obj: object number\n')
      for i, im in enumerate(self.images):
         iminfo = "! im %d = %s\n" % (i, im.basename)
         f.write(iminfo)
      f.write('!\n')
      f.write("! %4s %4s %4s %16s %16s %16s\n" % ("sh", "obj", "im", "sum", "area", "flux"))
      line = '!' + '='*78 + '\n'
      f.write(line)
      for sh in self.shapes:
         for ol, objlist in enumerate(sh):
            for obj in objlist:
               if obj.area != None:
                  f.write('  %4d %4d %4d %16g %16g %16g\n' % (obj.shapetype, ol, obj.framenr, obj.sum, obj.area, obj.flux))
      f.close()
      mes = "Wrote flux results to file: %s" % filename
      self.toolbar.set_message(mes)
      if self.gipsy and gipsymod:
         anyout(mes)


# Adding areas for which we want to calculate flux:
# Starting point is a canvas with a number of images. Each image is associated with
# an Axes object (frame). A user either starts with no flux objects at all
# or reads objects from a file. Assume there are no flux objects yet. Then we need 
# a 'key_pressed' callback that creates an object of a certain type. e.g.
# 1) Start polygon: notify user and change key_pressed callback to that of the polygon object
# 2) Start ellipse: display a default ellipse which can be moved and modified,
#    notify user and change key_pressed callback to that of the ellipse object
# 3) Spline -in fact a polygon with more vertices than control points
# 4) Rectangle -in fact a polygon with 4 vertices
#
# Note that we need to identify the frame in which the polygon is drawn. For that frame the
# coordinate system is pixels. A change in a polygon must propagate into the other frames.
# If these frames have different wcs's (world coordinate system), we need to convert
# a pixel position into a world coordinate and to update the polygon in the other frames we
# need to convert the world coordinate to pixels in the system of the other images.
# Therefore each shape should know to which image it belongs and which projection object
# (to transform pixels to world coordinates vv) it should use.
#
# As soon we initialized a flux object, the key-pressed callback changes.
# In this stage a shape contains zero, one, two or more than two vertices.
# Then there must be keys and buttons to modify this shape. These can be different for different
# shapes.
# Also the key-pressed callback that belongs to the flux object should allow a user to select
# another object in the same frame or in another frame with an image.
 
# Next part sets the environment (figure, frame) and shows the images


def main():
   fig = figure(figsize=(10,10), facecolor="#fff07e")
   frames = [None, None, None, None]
   frames[0] = fig.add_subplot(2,2,1, aspect=1, adjustable='box', autoscale_on=False)
   frames[1] = fig.add_subplot(2,2,2, aspect=1, adjustable='box', autoscale_on=False)
   frames[2] = fig.add_subplot(2,2,3, aspect=1, adjustable='box', autoscale_on=False)
   frames[3] = fig.add_subplot(2,2,4, aspect=1, adjustable='box', autoscale_on=False)
   
   frames[0].set_xlim(0,10)
   frames[0].set_ylim(0,10)
   frames[1].set_xlim(0,8)
   frames[1].set_ylim(0,7)
   frames[2].set_xlim(0,8)
   frames[2].set_ylim(0,11)
   frames[3].set_xlim(0,8)
   frames[3].set_ylim(0,7)
   
   names = ['m101', 'M1', 'L1', 'NGC2323']
   
   class Projection(object):
      def __init__(self, name):
         self.name = name
         
      def toworld(self, xy):
         if self.name != 'L1':
            xyw = numpy.asarray(xy)*2.0
         else:
            xyw = xy
         return xyw
      
      def topixel(self, xy):
         if self.name != 'L1':
            xyp = numpy.asarray(xy)/2.0
         else:
            xyp = xy
         return xyp   
   
   class Image(object):
      def __init__(self, name, frame):
         self.name = name
         self.frame = frame
         self.projection = Projection(name)
      def positionmessage(self, x, y):
         return "%g %g" % (x, y)
      def getflux(xy,pixelstep=0.2):
         return 10, 3    # Just dummies
         
   images = []
   for n,f in zip(names,frames):
      im = Image(n,f)
      images.append(im)
      
   shapes = Shapecollection(images, fig, wcs=True)
   show()


if __name__ == "__main__":
    main()