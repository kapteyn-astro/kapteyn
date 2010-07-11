#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    rulers.py
# PURPOSE: Provide methods to plot a ruler showing great circle offsets
#          with respect to a given starting point.
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    April 17, 2010
# UPDATE:  April 17, 2010
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

Module rulers
===============
This module defines a class for drawing rulers.

.. autoclass:: Ruler

"""

import numpy
from kapteyn.positions import str2pos, unitfactor


class Ruler(object):
   #-----------------------------------------------------------------
   """
   Draws a line between two spatial positions
   from a start point (x1,y1) to an end point (x2,y2)
   with labels indicating a constant offset in world
   coordinates. The positions are either in pixels
   or in world coordinates. The ruler is a straight
   line but the ticks are usually not equidistant
   because projection effects make the offsets non linear
   (e.g. the TAN projection diverges while the CAR projection
   shows equidistant ticks).
   By default, the zero point is exactly in the middle of
   the ruler but this can be changed by setting a
   value for *lambda0*.  The step size
   for the ruler ticks in units of the spatial
   axes is entered in parameter *step*.
   At least one of the axes in the plot needs to be
   a spatial axis.

   :param projection:    The Projection object which sets the WCS for the ruler.
   :type projection:     A :class:`wcs.Projection` object

   :param mixpix:        The pixel of the missing spatial axis in a Position-Velocity
                         image.
   :type mixpix:         Integer

   :param pxlim:         Limit in pixel coordinates for the x-axis.
   :type pxlim:          Tuple or list with two integers.

   :param pylim:         Limit in pixel coordinates for the y-axis.
   :type pylim:          Tuple or list with two integers.

   :param aspectratio:   The aspect ratio is defined as *pixel height / pixel width*.
                         The value is needed to draw tick mark perpendicular
                         to the ruler line for images where the pixels are not square
                         in world coordinates. Its default is 1.0.
   :type aspectratio:    Float

   :param pos1:          Position information for the start point. This info overrules
                         the values in x1 and y1.
   :type pos1:           String

   :param pos1:          Position information for the end point. This info overrules
                         the values in x2 and y2.
   :type pos1:           String

   :param x1:            X-location of start of ruler either in pixels or world coordinates
                         Default is lowest pixel coordinate in x.
   :type x1:             None or Floating point number

   :param y1:            Y-location of start of ruler either in pixels or world coordinates
                         Default is lowest pixel coordinate in y.
   :type y1:             None or Floating point number

   :param x2:            X-location of end of ruler either in pixels or world coordinates
                         Default is highest pixel coordinate in x.
   :type x2:             None or Floating point number

   :param y2:            Y-location of end of ruler either in pixels or world coordinates
                         Default is highest pixel coordinate in y.
   :type y2:             None or Floating point number

   :param lambda0:       Set the position of label which represents offset 0.0.
                         Default is lambda=0.5 which represents the middle of the ruler.
                         If you set lambda=0 then offset 0.0 is located at the start
                         of the ruler. If you set lambda=1 then offset 0.0 is located at the
                         end of the ruler.
   :type lambda0:        Floating point number

   :param step:          Step size of world coordinates in degrees or in units
                         entered in *units*.
   :type step:           Floating point number

   :param world:         Set ruler mode to world coordinates (default is pixels)
   :type world:          Boolean

   :param angle:         Set angle of tick marks in degrees. If omitted then a default
                         is calculated (perpendicular to ruler line) which applies
                         to all labels.
   :type angle:          Floating point number

   :param addangle:      Add a constant angle in degrees to *angle*.
                         Only useful if *angle* has its default
                         value. This parameter is used to improve layout.
   :type adangle:        Floating point number

   :param fmt:           Format of the labels. See example.
   :type fmt:            String

   :param fun:           Format ruler values according to this function (e.g. to convert
                         degrees into arcminutes). The output is always in degrees.
   :type fun:            Python function or Lambda expression

   :param units:         Rulers ticks are labeled in a unit that is compatible
                         with degrees. The units are set by the step size used to
                         calculate the position of the tick marks. You can
                         set these units explicitely with this parameter.
                         Note that values for *fun* and *fmt*
                         cannot be set because these are set automatically if
                         *units* has a value. Note that *units* needs only
                         a part of a complete units string because a
                         case insensitive minimal match
                         is applied. Usually one will use something like
                         *units=arcmin* or *units=Arcsec*.

                         Note: If a value for *units* is entered, then this method
                         expects the step size is given in the same units.
   :type units:          String

   :param fliplabelside: Choose other side of ruler to draw labels.
   :type fliplabelside:  Boolean

   :param mscale:        A scaling factor to create more or less distance between
                         the ruler and its labels. If *None* then this method calculates
                         defaults. The values are usually less than 5.0.

   :type mscale:         Floating point number

   :param `**kwargs`:    Set keyword arguments for the labels.
                         The attributes for the ruler labels are set with these keyword arguments.
   :type `**kwargs`:     Matplotlib keyword argument(s)

   :Raises:
      :exc:`Exception`
         *Rulers only suitable for maps with at least one spatial axis!*
         These rulers are only for plotting offsets as distances on
         a sphere for the current projection system. So we need at least
         one spatial axis and if there is only one spatial axis in the plot,
         then we need a matching spatial axis.
      :exc:`Exception`
         *Cannot make ruler with step size equal to zero!*
         Either the input of the step size is invalid or a wrong default
         was calculated (perhaps end point is equal to start point).
      :exc:`Exception`
         *Start point of ruler not in pixel limits!*
      :exc:`Exception`
         *End point of ruler not in pixel limits!*

   :Returns:      A ruler object of class ruler which is added to the plot container
                  with Plotversion's method :meth:`Plotversion.add`.
                  This ruler object has two methods to change the properties
                  of the line and the labels:

                  * `setp_line(**kwargs)` -- Matplotlib keyword arguments for changing
                     the line properties.
                  * `setp_labels(**kwargs)` -- Matplotlib keyword arguments for changing
                     the label properties.

   :Notes:        A bisection is used to find a new marker position so that
                  the distance to a previous position is *step*..
                  We use a formula of Thaddeus Vincenty, 1975, for the
                  calculation of a distance on a sphere accurate over the
                  entire sphere.

   :Examples:     Create a ruler object and change its properties

                  ::

                     ruler2 = annim.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=0.5, step=2.0,
                                          fmt='%3d', mscale=-1.5, fliplabelside=True)
                     ruler2.setp_labels(ha='left', va='center', color='b')

                     ruler4 = annim.Ruler(pos1="23h0m 15d0m", pos2="22h0m 30d0m", lambda0=0.0,
                                          step=1, world=True,
                                          fmt=r"$%4.0f^\prime$",
                                          fun=lambda x: x*60.0, addangle=0)
                     ruler4.setp_line(color='g')
                     ruler4.setp_labels(color='m')

                     # Force step size and labeling to be in minutes of arc.
                     annim.Ruler(pos1='0h3m30s 6d30m', pos2='0h3m30s 7d0m',
                                 lambda0=0.0, step=5.0,
                                 units='arcmin', color='c')

   .. automethod:: setp_line
   .. automethod:: setp_label
   """
   #-----------------------------------------------------------------
   def __init__(self, projection, mixpix, pxlim, pylim, aspectratio=1.0,
                pos1=None, pos2=None,
                x1=None, y1=None, x2=None, y2=None, lambda0=0.5, step=None,
                world=False, angle=None, addangle=0.0,
                fmt=None, fun=None, units=None, fliplabelside=False, mscale=None,
                labelsintex=True, **kwargs):
      self.ptype = "Ruler"
      self.x1 = None
      self.y1 = None
      self.x2 = None
      self.y2 = None
      self.x = []
      self.y = []
      self.xw = []
      self.yw = []
      self.stepsizeW = None
      self.label = []
      self.offsets = []      # Store the offsets in degrees
      self.angle = None
      self.kwargs = {'clip_on' : True}   # clip_on is buggy for plot() in MPL versions <= 0.98.3 change later
      self.tickdx = None
      self.tickdy = None
      self.mscale = None
      self.fun = None
      self.fmt = None
      self.linekwargs = {'color' : 'k'}
      self.kwargs.update(kwargs)    # These are the kwargs for the labels

      # Recipe:
      # Are the start and endpoint in world coordinates or pixels?
      # Convert to pixels.
      # Calculate the central position in pixels
      # Calculate the central position in world coordinates (Xw,Yw)
      # Find a lambda in (x,y) = (x1,y1) + lambda*(x2-x1,y2-x1)
      # so that, if (x,y) <-> (xw,yw), the distance D((Xw,Yw), (xw,yw))
      # is the step size on the ruler.
      def bisect(offset, lambda_s, Xw, Yw, x1, y1, x2, y2):
         """
         We are looking for a value mu so that mu+lambda_s sets a
         pixel which corresponds to world coordinates that are
         'offset' away from the start point set by lambda_s
         If lambda_s == 0 then we are in x1, x2. If lambda_s == 1
         we are in x2, y2
         """
         mes = ''
         if offset >= 0.0:
            a = 0.0; b = 1.1
         else:
            a = -1.1; b = 0.0
   
         f1 = getdistance(a, lambda_s, Xw, Yw, x1, y1, x2, y2) - abs(offset)
         f2 = getdistance(b, lambda_s, Xw, Yw, x1, y1, x2, y2) - abs(offset)
         validconditions = f1*f2 < 0.0
         if not validconditions:
            mes = "Found interval without a root for this step size"
            return  None, mes
   
         tol = 1e-12   # Tolerance. Stop iteration if (b-a)/2 < tol
         N0  = 50      # Stop output with error message if number of iterations
                        # exceeds this number
         # Initialize
         i = 0
         fa = getdistance(a, lambda_s, Xw, Yw, x1, y1, x2, y2) - abs(offset)
         # The iteration itself
         while i <= N0:
            # The bisection
            p = a + (b-a)/2.0
            fp = getdistance(p, lambda_s, Xw, Yw, x1, y1, x2, y2) - abs(offset)
            # Did we find a root?
            i += 1
            if fp == 0.0 or (b-a)/2.0 < tol:
               # print 'Root is: ', p, fp          # We found a root
               # print "Iterations: ", i
               break                         # Success..., leave the while loop
            if fa*fp > 0:
               a = p
               fa = fp
            else:
               b = p
         else:
            mes = 'Ruler bisection failed after %d iterations!' % N0
            p = None
         return p, mes
   
   
      def DV(l1, b1, l2, b2):
         # Vincenty, Thaddeus, 1975, formula for distance on sphere accurate over entire sphere
         fac = numpy.pi / 180.0
         l1 *= fac; b1 *= fac; l2 *= fac; b2 *= fac
         dlon = l2 - l1
         a1 = numpy.cos(b2)*numpy.sin(dlon)
         a2 = numpy.cos(b1)*numpy.sin(b2) - numpy.sin(b1)*numpy.cos(b2)*numpy.cos(dlon)
         a = numpy.sqrt(a1*a1+a2*a2)
         b = numpy.sin(b1)*numpy.sin(b2) + numpy.cos(b1)*numpy.cos(b2)*numpy.cos(dlon)
         d = numpy.arctan2(a,b)
         return d*180.0/numpy.pi
   
   
      def tolonlat(x, y):
         # This function also sorts the spatial values in order
         # longitude, latitude
         if mixpix == None:
            xw, yw = projection.toworld((x,y))
            xwo = xw     # Store originals
            ywo = yw
         else:
            W = projection.toworld((x, y, mixpix))
            xw = W[projection.lonaxnum-1]
            yw = W[projection.lataxnum-1]
            xwo = xw; ywo = yw
            if projection.lonaxnum > projection.lataxnum:
               xwo, ywo = ywo, xwo    # Swap
         return xw, yw, xwo, ywo
   
   
      def topixel2(xw, yw):
         # Note that this conversion is only used to convert
         # start and end position, given in world coordinates,
         # to pixels.
         if mixpix == None:
            x, y = projection.topixel((xw,yw))
         else:
            unknown = numpy.nan
            wt = (xw, yw, unknown)
            pixel = (unknown, unknown, mixpix)
            (wt, pixel) = projection.mixed(wt, pixel)
            x = pixel[0]; y = pixel[1]
         return x, y
   
   
      def getdistance(mu, lambda_s, Xw, Yw, x1, y1, x2, y2):
         lam = lambda_s + mu
         x = x1 + lam*(x2-x1)
         y = y1 + lam*(y2-y1)
         xw, yw, xw1, yw1 = tolonlat(x,y)
         return DV(Xw, Yw, xw, yw)
   
   
      def nicestep(x1, y1, x2, y2):
         # Assume positions in pixels
         xw1, yw1, dummyx, dummyy = tolonlat(x1,y1)
         xw2, yw2, dummyx, dummyy = tolonlat(x2,y2)
         step = None
         length = DV(xw1, yw1, xw2, yw2)
         # Nice numbers for dms should also be nice numbers for hms
         sec = numpy.array([30, 20, 15, 10, 5, 2, 1])
         minut = sec
         deg = numpy.array([60, 30, 20, 15, 10, 5, 2, 1])
         nicenumber = numpy.concatenate((deg*3600.0, minut*60.0, sec))
         fact = 3600.0
   
         d = length * fact
         step2 = 0.9*d/3.0          # We want at least four offsets on our ruler
         for p in nicenumber:
            k = int(step2/p)
            if k >= 1.0:
               step2 = k * p
               step = step2
               break           # Stop if we have a candidate
   
         # d = x2 - x1
         # If nothing suitable then try something else
         if step == None:
            f = int(numpy.log10(d))
            if d < 1.0:
               f -= 1
            D3 = numpy.round(d/(10.0**f),0)
            if D3 == 3.0:
               D3 = 2.0
            elif D3 == 6:
               D3 = 5.0
            elif D3 == 7:
               D3 = 8
            elif D3 == 9:
               D3 = 10
            if D3 in [2,4,8]:
               k = 4
            else:
               k = 5
            step = (D3*10.0**f)/k
         return step/fact

      spatial = projection.types[0] in ['longitude', 'latitude'] or projection.types[1] in ['longitude', 'latitude']
      if not spatial:
         raise Exception, "Rulers only suitable for maps with at least one spatial axis!"
   
      if pos1 != None:
         poswp = str2pos(pos1, projection, mixpix=mixpix)
         if poswp[3] != "":
            raise Exception, poswp[3]
         # The result of the position parsing of str2pos is stored in 'poswp'
         # Its second element are the returned pixel coordinates.
         # (poswp[1]).
         # Note we required 1 position. Then the pixel coordinate we want is
         # poswp[1][0]. If we omit the last index then we end up with a sequence (of 1)
         # which cannot be processed further. Finally the pixel coordinate represents a
         # position in 2-dim. So the first element represents x (poswp[1][0][0]).
         pix =  poswp[1][0]
         x1 = pix[0]
         y1 = pix[1]
      else:
         if x1 == None: x1 = pxlim[0]; world = False
         if y1 == None: y1 = pylim[0]; world = False
         if world:
            x1, y1 = topixel2(x1, y1)
   
      if pos2 != None:
         poswp = str2pos(pos2, projection, mixpix=mixpix)
         if poswp[3] != "":
            raise Exception, poswp[3]
         pix =  poswp[1][0]
         x2 = pix[0]
         y2 = pix[1]
      else:
         if x2 == None: x2 = pxlim[1]; world = False
         if y2 == None: y2 = pylim[1]; world = False
         if world:
            x2, y2 = topixel2(x2, y2)
   
      #print "DV", DV(23*15,15, 22*15, 30)*60.0
   
      # Get a step size for nice offsets
      if step == None:
         stepsizeW = nicestep(x1, y1, x2, y2)
      else:
         stepsizeW = step
      if step == 0.0:
         raise Exception, "Cannot make ruler with step size equal to zero!"
   
   
      # Look for suitable units (degrees, arcmin, arcsec) if nothing is
      # specified in the call. Note that 'stepsizeW' is in degrees.
      uf = None
      if fun == None and fmt == None:
         if units != None:
            uf, errmes = unitfactor('degree', units)
            if uf == None:
               raise ValueError(errmes)
            if uf != 1.0:
               fun = lambda x: x*uf
               # Input in 'units' but must be degrees for further processing
               stepsizeW /= uf
            if uf == 1.0:
               if labelsintex:
                  fmt = r"%4.0f^{\circ}"
               else:
                  fmt = u"%4.0f\u00B0"
            elif uf == 60.0:
               # Write labels in arcmin
               if labelsintex:
                  fmt = r"%4.0f^{\prime}"
               else:
                  fmt = r"%4.0f'"
            elif uf == 3600.0:
               # Write labels in arcsec
               if labelsintex:
                  fmt = r"%4.0f^{\prime\prime}"
               else:
                  fmt = r"%4.0f''"
            else:
               raise ValueError("Only Degrees, arcmin and arcsec allowed")
         else:
            if labelsintex:
               fmt = r"%4.0f^{\circ}"
            else:
               fmt = u"%4.0f\u00B0"
            if abs(stepsizeW) < 1.0:
               # Write labels in arcmin
               fun = lambda x: x*60.0
               if labelsintex:
                  fmt = r"%4.0f^{\prime}"
               else:
                  fmt = r"%4.0f'"
            if abs(stepsizeW) < 1.0/60.0:
               # Write labels in arcsec
               fun = lambda x: x*3600.0
               if labelsintex:
                  fmt = r"%4.0f^{\prime\prime}"
               else:
                  fmt = r"%4.0f''"
      elif fmt == None:          # A function but not a format. Then a default format
         fmt = '%g'
      # Check whether the start- and end point of the ruler are inside the frame
      start_in = (pxlim[0]-0.5 <= x1 <= pxlim[1]+0.5) and (pylim[0]-0.5 <= y1 <= pylim[1]+0.5)
      if not start_in:
         raise Exception, "Start point of ruler not in pixel limits!"
      end_in = (pxlim[0]-0.5 <= x2 <= pxlim[1]+0.5) and (pylim[0]-0.5 <= y2 <= pylim[1]+0.5)
      if not end_in:
         raise Exception, "End point of ruler not in pixel limits!"
   
      # Ticks perpendicular to ruler line. Prependicular is with respect to
      # square pixels, so correct these first for their aspect ratio to find
      # the right angle.
      defangle = 180.0 * numpy.arctan2(y2-y1, (x2-x1)/aspectratio) / numpy.pi - 90.0
   
      l1 = pxlim[1] - pxlim[0] + 1.0; l1 /= 50.0
      l2 = pylim[1] - pylim[0] + 1.0; l2 /= 50.0
      ll = max(l1,l2)
      dx = ll*numpy.cos(defangle*numpy.pi/180.0)*aspectratio
      dy = ll*numpy.sin(defangle*numpy.pi/180.0)
      if fliplabelside:
         dx = -dx
         dy = -dy
   
      if angle == None:
         phi = defangle
      else:
         phi = angle
      phi += addangle
      defkwargs = {'fontsize':10, 'rotation':phi}
      if defangle+90.0 in [270.0, 90.0, -90.0, -270.0]:
         if fliplabelside:
            defkwargs.update({'va':'center', 'ha':'right'})
         else:
            defkwargs.update({'va':'center', 'ha':'left'})
         if mscale == None:
            mscale = 1.5
      elif defangle+90.0 in [0.0, 180.0, -180.0]:
         if fliplabelside:
            defkwargs.update({'va':'bottom', 'ha':'center'})
         else:
            defkwargs.update({'va':'top', 'ha':'center'})
         mscale = 1.5
      else:
         defkwargs.update({'va':'center', 'ha':'center'})
         if mscale == None:
            mscale = 2.5
      defkwargs.update(kwargs)
      #ruler = Rulerstick(x1, y1, x2, y2, defangle, dx, dy, mscale, **defkwargs)
      self.x1 = x1; self.x2 = x2; self.y1 = y1; self.y2 = y2
      self.angle = defangle
      self.tickdx = dx; self.tickdy = dy
      self.mscale = mscale
      self.kwargs.update(defkwargs) 
      self.fmt = fmt
      self.fun = fun
   
      lambda_s = lambda0
      x0 = x1 + lambda_s*(x2-x1)
      y0 = y1 + lambda_s*(y2-y1)
      Xw, Yw, xw1, yw1 = tolonlat(x0, y0)
      self.append(x0, y0, 0.0, fmt%0.0)
      self.appendW(xw1, yw1)         # Store in original order i.e. not sorted
      self.stepsizeW = stepsizeW     # Needed elsewhere so store as an attribute
   
   
      # Find the mu on the straight ruler line for which the distance between
      # the position defined by mu and the center point (lambda0) is 'offset'
      # Note that these distances are calculated on a sphere
      for sign in [+1.0, -1.0]:
         mu = 0.0
         offset = 0.0
         lamplusmu = lambda_s + mu
         while mu != None and (0.0 <= lamplusmu <= 1.0):
            offset += sign*stepsizeW
            mu, mes = bisect(offset, lambda_s, Xw, Yw, x1, y1, x2, y2)
            if mu != None:
               lamplusmu = lambda_s + mu
               if 0.0 <= lamplusmu <= 1.0:
                  x = x1 + (lamplusmu)*(x2-x1)
                  y = y1 + (lamplusmu)*(y2-y1)
                  if fun != None:
                     off = fun(offset)
                  else:
                     off = abs(offset)
                  self.append(x, y, offset, fmt%off, labelsintex)
                  xw, yw, xw1, yw1 = tolonlat(x, y)
                  self.appendW(xw1, yw1)
            elif sign == -1.0:
               break
               # raise Exception, mes
      self.pxlim = pxlim
      self.pylim = pylim


   def setp_line(self, **kwargs):
      #-----------------------------------------------------------------
      """
      Set the ruler line properties. The keyword arguments are Matplotlib
      keywords for :class:`Line2D` objects.

      :param kwargs: Keyword argument(s) for changing the default properties
                     of the ruler line. This line is a :class:`Line2D`
                     Matplotlib object with attributes like
                     *linewidth*, *color* etc.
      :type kwargs:  Python keyword arguments
      """
      #-----------------------------------------------------------------
      self.linekwargs.update(kwargs)


   def setp_label(self, **kwargs):
      #-----------------------------------------------------------------
      """
      Set the ruler label properties. The keyword arguments are Matplotlib
      keywords for :class:`Text` objects. Note that the properties
      apply to all labels. It is not possible to address a separate label.

      :param kwargs: Keyword argument(s) for changing the default properties
                     of the ruler labels. This line is a :class:`Text`
                     Matplotlib object with attributes like
                     *fontsize*, *color* etc.
      :type kwargs:  Python keyword arguments
      """
      #-----------------------------------------------------------------
      self.kwargs.update(kwargs)


   def append(self, x, y, offset, label, labelsintex=True):
      self.x.append(x)
      self.y.append(y)
      self.offsets.append(offset)
      if labelsintex:
         label = r"$%s$"%label
      self.label.append(label)


   def appendW(self, xw, yw):
      self.xw.append(xw)
      self.yw.append(yw)


   def plot(self, frame):
      """
      Plot one ruler object in the current frame
      """
      ruler = self
      frame.plot((ruler.x1,ruler.x2), (ruler.y1,ruler.y2), '-', **ruler.linekwargs)
      dx = ruler.tickdx
      dy = ruler.tickdy
      #self.frame.plot( [ruler.x1, ruler.x1+dx], [ruler.y1, ruler.y1+dy], '-', **ruler.linekwargs)
      #self.frame.plot( [ruler.x2, ruler.x2+dx], [ruler.y2, ruler.y2+dy], '-', **ruler.linekwargs)
      for x, y, label in zip(ruler.x, ruler.y, ruler.label):
         frame.plot( [x, x+dx], [y, y+dy], '-', color='k')
         frame.text(x+ruler.mscale*dx, y+ruler.mscale*dy, label, **ruler.kwargs)
