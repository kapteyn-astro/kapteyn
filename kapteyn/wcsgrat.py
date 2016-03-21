#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    wcsgrat.py
# PURPOSE: Provide classes and methods to produce graticule data 
#          for WCS labeling and plotting grid lines.
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    August 17, 2008
# UPDATES: Version 0.2, September 09, 2008
#          Version 0.3, October 17, 2008
#          Version 1.0, June 17, 2009
#          Version 1.1, December 05, 2012 (bug fixes only)
#          Version 1.2, December 14, 2014; replaced == None etc.
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#----------------------------------------------------------------------

"""
Module wcsgrat
==============
A graticule is a system of crossing lines on a map representing
positions of which one coordinate is constant.
For a spatial map it consists of parallels of latitude and
meridians of longitude as defined by a given projection.

This module is used to set up such graticules and labels for the
selected world coordinate system. It plots the results with
plotting library
`Matplotlib <http://matplotlib.sourceforge.net/index.html>`_.

Besides spatial axes, it supports also spectral axes and a mix of both
(e.g. position-velocity diagrams). It deals with data dimensions > 2
by allowing arbitrary selections of two axes.
The transformations between pixel coordinates and world coordinates
are based on module 
:mod:`wcs` which is a Python binding for Mark R. Calabretta's library
`WCSLIB <http://www.atnf.csiro.au/people/mcalabre/WCS>`_.
>From *WCSLIB* we use only the core transformation routines.
Header parsing is done with module :mod:`wcs`.

Axes types that are not recognized by this software is treated as being linear.
The axes types correspond with keywords *CTYPEn* in a FITS file.
The information from a FITS file is retrieved by module
`PyFITS <http://www.stsci.edu/resources/software_hardware/pyfits>`_

.. seealso:: Tutorial material with code examples:
   
     * Tutorial maputils module
       which contains many examples with source code,
       see :ref:`maputils_tutorial`.

     * Figure gallery 'all sky plots'
       with many examples of Graticule constructors,
       see :ref:`allsky_tutorial`.
   
.. moduleauthor:: Martin Vogelaar <gipsy@astro.rug.nl>



Module level data
-----------------

:data:`left, bottom, right, top`
   The variables *left*, *bottom*, *right* and *top* are
   equivalent to the strings *"left"*, *"bottom"*, *"right"* and *"top"* 
   and are used as identifiers for plot axes.
:data:`native, notnative, bothticks, noticks`
   The variables *native*, *notnative*, *bothticks*, *noticks* 
   correspond to the numbers 0, 1, 2 and 3 and represent modes 
   to make ticks along an axis visible or invisible. Ticks along an axis
   can represent both world coordinate types (e.g. when a map is rotated). Sometimes
   one wants to allow this and sometimes not.

   .. tabularcolumns:: |p{25mm}|p{125mm}|

   ========= =============================================
   Tick mode Description
   ========= =============================================
   native    Show only ticks that are native to the 
             coordinate axis. Do not allow ticks 
             that correspond to the axis for which 
             a constant value applies. So, for example,
             in a RA-DEC
             map which is rotated 45 degrees we want only
             Right Ascensions along the x-axis.
   notnative Plot the ticks that are not native to the
             coordinate axis. So, for example, in a RA-DEC
             map which is rotated 45 degrees we want only
             Declinations along the x-axis.
   bothticks Allow both type of ticks along a plot axis
   noticks   Do not allow any tick to be plotted.
   ========= =============================================

Functions
---------

.. index:: Convert degrees to separate field of the hms/dms format
.. autofunction:: gethmsdms
.. index:: Convert hms or dms values to text or LaTeX representation
.. autofunction:: makelabel


Class Graticule
---------------

.. autoclass:: Graticule
.. autoclass:: WCStick

Class Insidelabels
--------------------

.. autoclass:: Insidelabels

"""
from sys import stdout
from kapteyn import wcs        # The Kapteyn Python binding to WCSlib, including celestial transformations
from kapteyn.positions import str2pos, parsehmsdms, unitfactor
from kapteyn import rulers
from kapteyn.celestial import skyparser
import six
from string import ascii_letters as letters
from random import choice
import numpy


__version__ = '1.1'
(left,bottom,right,top) = list(range(4))                 # Names of the four plot axes
(native, notnative, bothticks, noticks) = list(range(4))

# A variable to change the vertical size of a TeX box
# For more information, have a look at method
# _update_metrics() at the end of this code.
tweakhms = 0

def issequence(obj):
  return isinstance(obj, (list, tuple, numpy.ndarray))


def parseplotaxes(plotaxes):
   """
   -----------------------------------------------------------
   Purpose:      It is possible to specify axes by an integer
                 or by a string.
                 The function is used internally to allow
                 flexible input of numbers to identify one of
                 the four plot axes.
   Parameters:
      plotaxes - Scalar or sequence with elements that are either
                 integers or strings or a combination of those.

   Returns:      A list with unique numbers between 0 and 3

   Notes:        - Order is unimportant
                 - Input can be a scalar or a sequence (tuple, list)
                 - Scalers are upgraded to a list.
                 - The result has only unique numbers
   -----------------------------------------------------------
   """
   if not issequence(plotaxes):
      plotaxes = [plotaxes,]
   if isinstance(plotaxes, tuple):
      plotaxes = list(plotaxes)
   for i,pa in enumerate(plotaxes):
      if isinstance(pa, six.string_types):
         if "LEFT".find(pa.upper()) == 0:
            plotaxes[i] = left
         elif "BOTTOM".find(pa.upper()) == 0:
            plotaxes[i] = bottom
         elif "RIGHT".find(pa.upper()) == 0:
            plotaxes[i] = right
         elif "TOP".find(pa.upper()) == 0:
            plotaxes[i] = top
         else:
            raise ValueError("[%s] Cannot identify this plot axis!" % pa)
   for pa in plotaxes:                           # Check validity
      if pa < 0 or pa > 3:
         raise ValueError("Cannot identify this plot axis!")
   aset = {}
   list(map(aset.__setitem__, plotaxes, []))
   return list(aset.keys())



def parsetickmode(tickmode):
   """
   -----------------------------------------------------------
   Purpose:
   Parameters:
      tickmode - Scalar or sequence with elements that are either
                 integers or strings or a combination of those.

   Returns:      A list with unique numbers between 0 and 3

   Notes:        

   -----------------------------------------------------------
   """
   #(native, notnative, bothticks, noticks) = range(4)
   if isinstance(tickmode, six.string_types):
      if "NATIVE_TICKS".find(tickmode.upper()) == 0:
         tickmode = native
      elif "SWITCHED_TICKS".find(tickmode.upper()) == 0:
         tickmode = notnative
      elif "ALL_TICKS".find(tickmode.upper()) == 0:
         tickmode = bothticks
      elif "NO_TICKS".find(tickmode.upper()) == 0:
         tickmode = noticks
      else:
         raise ValueError("[%s] Cannot identify this tick mode!" % tickmode)
   if tickmode < 0 or tickmode > 3:
      raise ValueError("%d does not correspond to a supported tick mode!" % tickmode)
   return tickmode




def gethmsdms(a, prec, axtype, skysys, eqlon=None):
   #---------------------------------------------------------------------------
   """
   Given a number in degrees and an axis type in *axtype*
   equal to 'longitude' or 'latitude',
   calculate and return the parts of its sexagesimal representation, i.e.
   hours or degrees, minutes and seconds. Also return the fractional seconds
   and the sign if the input was a value at negative latitude.
   The value for *skysys* sets the formatting to hours/minutes/seconds
   if it represents an equatorial system.
    

   :param a:
      The longitude or latitude in degrees.
   :type a:
      Floating point
   :param prec:
      The number of decimals in the seconds
   :type prec:
      Integer
   :param axtype:
      One of 'longitude' or 'latitude'
   :type axtype:
      String
   :param skysys:
      The sky system
   :type skysys:
      Integer

   :Returns:
      
      tuple: (Ihours, Ideg, Imin, Isec, Fsec, sign)
      which represent Integer values for the hours, degrees, minutes
      and seconds. *Fsec* is the fractional part of the seconds.
      Element *sign* is -1 for negative latitudes and +1 for positive
      latitudes.

      
   """
   #---------------------------------------------------------------------------
   if not axtype in ['longitude', 'latitude']:
      return None
   sign = 1
   if not issequence(skysys): # Usually a number, sometimes a tuple
      skysys = [skysys]
   if eqlon is None:
      eqlon = (axtype == 'longitude') and (wcs.equatorial in skysys)
   if axtype == 'longitude':
      degs = numpy.fmod(a, 360.0)  # Now in range -360, 360
      if degs < 0.0:
         degs += 360.0
   else:
      if a > 90.0 or a < -90.0:
         return None
      if a < 0.0:
         sign = -1
      degs = sign * a  # Make positive

   if prec < 0:
      prec = 0
   # How many seconds is this? Round to 'prec'
   if eqlon:
      sec = numpy.round(degs*240.0, prec)
      sec = numpy.fmod(sec, 360.0*240.0)     # Rounding can result in 360 deg. again, so correct
   else:
      sec = numpy.round(degs*3600.0, prec)
   AIsec = numpy.int(sec)     # Integer seconds
   Fsec = sec - AIsec     # Fractional remainder
   X = AIsec / 3600.0     # Either hours or degrees
   IX = numpy.int(X)
   secleft = AIsec - IX*3600.0
   Imin = int(secleft/60.0)
   Isec = secleft - Imin*60.0
   if eqlon:
      Ihours = IX; Ideg = None
   else:
      Ideg = IX; Ihours = None
   return Ihours, Ideg, Imin, Isec, Fsec, sign


def makeexplab(fmt, x):
#-------------------------------------------------------------------------------
   """
   Format exponentials
   """
#-------------------------------------------------------------------------------
   # The format string contains two formats. 
   if 1: #try:
      c = fmt.rfind('%')
      fmt1 = fmt[:c]
      fmt2 = fmt[c:]
      n = fmt2[1:fmt2.index('e')]
      ex = int(n)
      val = x/10.**ex
      lab = fmt1%val+"\cdot10^{%s}"%(str(ex)) + fmt2[fmt2.index('e')+1:]
   else: #except:
      raise TypeError("Wrong format (%s) for numbers in exponential LaTeX format"%fmt)
   return lab


def makelabel(hmsdms, Hlab, Dlab, Mlab, Slab, prec, fmt, tex):
   #-------------------------------------------------------------------------------
   """
   From the output of function *gethmsdms* and some Booleans, this function
   creates a label in plain text or in TeX. The Booleans set a flag whether
   a field (hours, degrees, minutes or seconds) should be printed or not.
   The *fmt* parameter is used if it does not contain the percentage character (%)
   but instead contains characters from the set HDMS. A capital overules the
   corresponding Boolean value, so if *fmt='HMS'*, the values for *Hlab*, *Mlab* and
   *Slab* are all set to True.

   :param hmsdms:
      The output of function :func:`wcsgrat.gethmsdms`
   :type hmsdms:
      Tuple with integer and floating point numbers
   :param Hlab:
      If False, there is no need to print the hours
   :param Dlab:
      If False, there is no need to print the degrees
   :param Mlab:
      If False, there is no need to print the minutes
   :param Slab:
      If False, there is no need to print the seconds
   :param fmt:
      String containing a combination of the characters
      ['H', 'D', 'M', 'S', '.', 'h', 'd', 'm', 's'] 
      A capital sets the corresponding input Boolean (Hlab, Dlab, etc.)
      to True. A dot starts to set the precision. The number of characters
      after the dot set the precision itself.
      A character that is not a capital sets the corresponding input
      Boolean (Hlab, Dlab, etc.) to False. This is a bit dangerous because
      with this option one can suppress fields to be printed that contain
      a value unequal to zero. It is applied if you want to suppress
      e.g. seconds if all the seconds in your label are 0.0.
      The suppression of printing minutes is overruled if hours (or degrees)
      and seconds are required. Otherwise we could end up with non
      standard labels (e.g. 2h30s).
   :type fmt:
      String
   :param tex:
      If True, then format the labels in LaTeX.
   :type tex:
      Boolean

   :Returns:
      *lab*, a label in either hms or dms in plain text or in LaTeX format.


   :Examples:

      >>> # Set the format in Hours, minutes and seconds with a precision
      >>> # of three. The suppression of minutes will not work here:
      >>> grat.setp_tick(wcsaxis=0, fmt="HmS.SSS")

      >>> # The same effect is obtained with:
      >>> grat.setp_tick(wcsaxis=0, fmt="HmS.###")

      >>> # Let the system determine whether seconds are printed
      >>> # but make sure that degrees and minutes are included:
      >>> grat.setp_tick(wcsaxis=1, fmt="DM")

      >>> # If we know that all minutes and seconds in our labels are 0.0
      >>> # and we want only the hours to be printed, then use:
      >>> grat.setp_tick(wcsaxis=0, fmt="Hms")

      >>> grat.setp_tick(wcsaxis=0, fmt="Dms")
      >>> # Plot labels in Degrees even if the axis is an equatorial longitude.
   """
   #-------------------------------------------------------------------------------
      
   Ihours, Ideg, Imin, Isec, Fsec, sign = hmsdms
   hms = False
   if Ihours is not None:
      hms = True
   if not (fmt is None):
      if fmt.find('%') == -1:   # Not a common format, must be a HMS/DMS format
         if fmt.find('H') != -1:
            Hlab = True
         if fmt.find('D') != -1:
            Dlab = True
         if fmt.find('M') != -1:
            Mlab = True
         if fmt.find('S') != -1:
            Slab = True
         if fmt.find('h') != -1:
            Hlab = False
         if fmt.find('d') != -1:
            Dlab = False
         if fmt.find('m') != -1:
            Mlab = False
         if fmt.find('s') != -1:
            Slab = False
         s2 = fmt.split('.')
         if len(s2) > 1:
            prec = len(s2[1])
   lab = ""

   # Minutes must be printed if hours and seconds are required
   # otherwise one can end up with a minimal label with hours/degs and
   # no minuts in between
   if hms:
      if Hlab:
         if tex:
            lab += r"%d^{\rm h}"%Ihours
         else:
            lab += "%dh"%Ihours
         if Slab:
            Mlab = True
   else:
      if Dlab:
         si = ' '
         if sign == -1:
            si = '-'
         if tex:
            lab += r"%c%d^{\circ}"%(si,Ideg)
         else:
            lab += "%c%dd"%(si,Ideg)
         if Slab:
            Mlab = True
   if Mlab:
      if tex:
         if hms:
            lab += r"%.2d^{\rm m}"%Imin
         else:
            lab += r"%.2d^{\prime}"%Imin
      else:
         lab += "%.2dm"%Imin
   if Slab:
      lab += "%.2d"%Isec
      if prec > 0:
         fsec = ".%*.*d" % (prec, prec, int(round(Fsec*10.0**prec,0)))
         lab += fsec
      if not tex:
         lab += 's'
      else:
         if hms:
            lab += r"^{\rm s}"
         else:
            lab += r"^{\prime\prime}"
   if sign == -1 and not Dlab:
      lab = "-"+lab

   return lab


class WCStick(object):
   #-------------------------------------------------------------------------------
   """
   A WCStick object is an intersection of a parallel or meridian (or equivalent
   lines with one constant world coordinate) with one of 
   the axes of a rectangle in pixels. The position of that intersection is 
   stored in pixel coordinates and can be used to plot a (formatted) label
   showing the position of the constant world coordinate of the graticule line.
   This class is only used in the context of the Graticule class.
   """
   #-------------------------------------------------------------------------------
   def __init__(self, x, y, axisnr, labval, wcsaxis, offset, fun=None, fmt=None):
      """
      -----------------------------------------------------------
      Purpose:     Store tick properties
      Parameters: 
       x -         pixel coordinate of position in x-direction
       y -         pixel coordinate of position in y-direction
       axisnr -    number between 0 and 4 representing
                   axes: left,bottom,right,top
       labval -    A (formatted) string representation
                   of the graticule line to which the tick belongs.
       wcsaxis -   0 for the first WCS axis, 1 for the second WCS axis
       offset -    Was it an offset?
      Returns:     WCStick object with kwargs which sets the 
                   plot properties for a tick
      Notes:       The keyword arguments attribute contains 
                   information about plot properties.These 
                   properties are (plot-)package specific.
                   We standardized on Matplotlib. If a plot
                   method is added for another plotting system,
                   one has to translate those properties to
                   the equivalents of that other system.
                   Have a look at the code in method 'plot' to
                   explore how such conversions should be done. 
      -----------------------------------------------------------
      """
      self.x = x              # A tick is an intersection with a rectangle at x, y
      self.y = y
      self.axisnr = axisnr    # Which axis did it intersect?
      self.labval = labval    # What is the value of the world coordinate?
      self.offset = offset    # Is it an offset?
      self.wcsaxis = wcsaxis  # To which (wcs) type of axis does it belong?
      self.fmt = fmt          # Python format string for conversion number to strings
      self.fun = fun          # Convert a tick (wcs) value with this function
      self.axtype = None      # Is it a long. or lat. axis?
      self.skysys = None      # And does it belong to an equatorial system?
      self.label = ""         # The text as it will appear in a plot
      self.prec = None        # The precision of the seconds in a hms/dms format
      self.delta = None       # The step between two ticks for this wcs axis
      self.tex = None         # Format label in TeX?
      self.texsexa = None     # Format the sexagesimal numbers
      self.kwargs = {}        # Keyword arguments to set plot attributes
      self.markkwargs = {}    # Keyword arguments to set plot attributes for the markers


def createlabels(Tlist):
   """
   -------------------------------------------------------------------------------
   This function creates the text labels for a list of tick which belong to
   the same axis and the same wcs axis.
   -------------------------------------------------------------------------------
   """
   global tweakhms
   Hprev = Dprev = Mprev = Sprev = signprev = None
   dirprev = None; Labprev = None
   H = D = M = S = True
   first = True
   for t in Tlist:
      # There are some conditions for plotting labels in hms/dms:
      stdout.flush()
      if t.axtype in ['longitude', 'latitude'] and t.offset == False and t.fun is None and\
         (t.fmt is None or t.fmt.find('%') == -1):

         # Each tick label has its own fontsize. We take the first label's fontsize as
         # the value for which we tweak the characters 's' and 'm' in the TeX labels
         if 'fontsize' in t.kwargs and t.tex:
            if tweakhms == 0:
               tweakhms = t.kwargs['fontsize']
            
         if t.axtype == 'longitude' and t.fmt is not None and 'D' in t.fmt.upper():
            # This is an equatorial longitude axis for which one wants
            # DMS formatting not HMS formatting. Useful for "all sky" plots
            eqlon = False
         else:
            eqlon = None
         hmsdms = gethmsdms(t.labval, t.prec, t.axtype, t.skysys, eqlon)
         Ihours, Ideg, Imin, Isec, Fsec, sign = hmsdms
         if first:
            hmsdms1 = hmsdms
            first = False
         # Take care of the fact that the first label is not at the end
         # so possibly the direction of labeling can be switched.
         if Labprev is not None:
            direction = t.labval > Labprev
            if direction != dirprev and dirprev is not None:
               Hprev, Dprev, Mpriv, Sprev, Fsecprev, sign = hmsdms1
               Sprev += Fsecprev
            dirprev = direction
         H = (Ihours != Hprev); D = (Ideg != Dprev); M = (Imin != Mprev); S = (Isec+Fsec != Sprev)
         Hprev = Ihours; Dprev = Ideg; Mprev = Imin; Sprev = Isec+Fsec; signprev = sign
         Labprev = t.labval
         texsexa = t.texsexa
         if texsexa is None:
            if t.tex:
               texsexa = True
            else:
               texsexa = False
         lab = makelabel(hmsdms, H, D, M, S, t.prec, t.fmt, texsexa)
      else:
         if t.fun is None:
            val = t.labval
         else:
            val = t.fun(t.labval)
         if t.fmt is None:
            lab = "%g" % val
         else:
            if t.tex and t.fmt.count("%") == 2:
               # There are two formats given so the user wants an exponential format
               lab = makeexplab(t.fmt, val)
            else:
               try:
                  lab = t.fmt % val
               except:
                  raise TypeError("Wrong format (%s) for numbers"%t.fmt)
      if t.tex:
         lab = r"$%s$" % lab
      t.label = lab


class Gratline(object):
   #-----------------------------------------------------------------
   """
   This class is used to find a set of coordinates that defines 
   (part of) a graticule line for which one of the world coordinates
   is a constant. It stores the coordinates of the intersections
   with the box and a corresponding label for annotating purposes.
   """
   #-----------------------------------------------------------------
   def __init__(self, wcsaxis, constval, gmap, pxlims, pylims, wxlims, wylims, 
                linesamples, mixgrid, skysys, addx=None, addy=None, addpixels=False, offsetlabel=None,
                fun=None, fmt=None):
      #-----------------------------------------------------------------
      """
      Purpose:     Initialize a 'grid line' which is part of a 
                   graticule. The method should be called within the 
                   context of the Graticule class only.

      Parameters:
       wcsaxis:    One of the values 0 or 1 for the first and second
                   world coordinate type. Or a number > 1 which is 
                   the id of a graticule line representing a border.
       constval:   For a graticule line, one of the world coordinates
                   is a constant. The other coordinate varies within
                   certain limits. The value of the constant is given by 
                   this parameter.
       gmap:       The projection object for these wcs axes.
       pxlims:     The lower and upper limits of an image in grids along
                   the x-axis.
       pylims:     Same for the y-axis
       wxlims:     Lower and upper values of the image in world
                   coordinates.
       wylims:     Same as wxlims but now for the y-axis
       linesamples:This parameter sets the number of samples with
                   which we build a graticule line. In fact it is
                   equivalent to a step size for straight lines.
                   Therefore it is also used to identify jumps
                   in longitudes and latitudes.
                   We want to avoid jumps in a plot and therefore
                   break up the graticule line in pieces.
       mixgrid:    For images with only one spatial axis, we need
                   to supply a pixel value for the matching spatial
                   axis which is more or less hidden but essential
                   in the coordinate transformation.
       skysys:     The skysystem for this graticule line is used
                   to format position labels in hour, minutes, 
                   seconds or degrees, minutes and seconds.
       addx:       If not set to None, it implies that we supplied
                   this method with known positions in world
                   coordinates or pixels. If world coordinates
                   the coordinate transformations from
                   pixel- to world coordinates are not necessary.
                   The parameters addx and addy can be used if you have
                   coordinates that define a border of an 
                   all-sky map, but to avoid drawing the border
                   outside the limits in pixels, we treat the line
                   as a graticule line so it is clipped correctly.
                   Note that if you want to use addx and addy then
                   the value of wcsaxis should not be 0 or 1.
       addy:       The same as addx but now for y
       addpixels:  True or False.
                   The values in addx and addy are either pixel
                   coordinates or world coordinates.
       offsetlabel:Text label for an offset axis
       fun:        Change the offset values using a function.
       fmt:        Format the offset using a Python format.

      Returns:     A graticule line which consists of one or more
                   line pieces with default (plot) attributes,
                   a sequence of ticks (each with a position, label
                   and default plot attributes)
 
      Notes:       Special graticule lines are those which sets
                   the limb of the graticule. Not all projections
                   behave in a way that limbs can be found with 
                   boundaries in world coordinates. Sometimes
                   we have a boundary in world coordinates but
                   it turns out to be not as accurate as possible.
                   For those situations we have  a rather crude
                   method to find boundaries in pixels. To avoid
                   jumps and to apply clipping at the edges we
                   process these pixels like other graticule lines.
      """
      #-----------------------------------------------------------------
      # Helper functions
      def __inbox(x, y, box):
         """
         --------------------------------------------------------------
         Purpose:     Is a position (x,y) in pixels within the 
                      limits of the image?
         Parameters:  
          x:          X-coordinate of pixel
          y:          Y-coordinate of pixel
          box:        Tuple of 4 numbers (xlo, ylo, xhi, yhi)

         Returns:     False or True

         Notes:       Note that internally the box is increased with
                      1/2 pixel in all directions
         --------------------------------------------------------------
         """
         if box[0] <= box[2]:
            if x >= box[2] or x <= box[0]:
               return False
         else:
            if x >= box[0] or x <= box[2]:
               return False
         if box[1] <= box[3]:
            if y >= box[3] or y <= box[1]:
               return False
         else:
            if y >= box[1] or y <= box[3]:
               return False
         return True


      def __handlecrossing(box, x1, y1, x2, y2):
         """
         -----------------------------------------------------------
         Purpose:    Return properties of intersection of 
                     graticule line and enclosing rectangle
         Parameters: 
          box:       As in helper function __inbox
          x1, y1:    First position in pixels inside or outside box
          x2, y2:    Second position outside or inside box

         Returns:    The axis number and the position of the intersection

         Notes:      Given the boundaries of a rectangle in grids,
                     and two positions of which we know that one 
                     is inside the box and the other is outside the
                     box, calculate the intersections of the line
                     through these points and all of the axes of
                     the box. Note that we used the following
                     axis-index relation.
                     0: left 
                     1: bottom 
                     2: right
                     3: top
         -----------------------------------------------------------
         """
         for axisnr in range(4):
            if axisnr == left or axisnr == right:
               # This is the left and right Y-axis for which x2 != x1
               # otherwise we would not have an intersection
               if x2-x1 != 0.0:
                  lamb = (box[axisnr]-x1)/(x2-x1)
               else:
                  lamb = -1.0;
               if  0.0 <= lamb <= 1.0:
                  ycross = y1 + lamb*(y2-y1)
                  xlab = box[axisnr]; ylab = ycross
                  # pylab.plot([box[axisnr]], [ycross], 'ro')
                  return axisnr, xlab, ylab
            else:
               # Check intersection with y axis
               if y2-y1 != 0.0:
                  lamb = (box[axisnr]-y1)/(y2-y1)
               else:
                  lamb = -1.0
               if  0.0 <= lamb <= 1.0:
                  xcross = x1 + lamb*(x2-x1)
                  ylab = box[axisnr]; xlab = xcross
                  # pylab.plot([xcross], [box[axisnr]], 'yo')
                  return axisnr, xlab, ylab


      #-------------------------------------------------
      # Start definition of __init__() 
      #-------------------------------------------------
      if wcsaxis == 0:              # constant value on X axis
         xw = numpy.zeros(linesamples) + constval
         dw = (wylims[1] - wylims[0])/1000.0  # Ensure that we cross borders
         yw = numpy.linspace(wylims[0]-dw, wylims[1]+dw, linesamples)
      elif wcsaxis == 1:            # constant value on Y axis
         dw = (wxlims[1] - wxlims[0])/1000.0
         xw = numpy.linspace(wxlims[0]-dw, wxlims[1]+dw, linesamples)
         yw = numpy.zeros(linesamples) + constval
      else:  # A grid line without one of the coordinates being constant e.g. a border
         xw = addx
         yw = addy
      if (mixgrid is None):  # Probably matching axis pair or two independent axes
         world = (xw, yw)
         if wcsaxis in [0,1]:
            pixel = gmap.topixel(world)
         else:              # This should be an 'added' grid line
            if addpixels:   # A set with pixels for a border is already provided
               pixel = (addx, addy)
            else:           # A set with world coordinates for a border is provided.
               pixel = gmap.topixel(world)
      else:                 # wcs coordinate pair with one extra grid coordinate
         unknown = numpy.zeros(linesamples)
         unknown += numpy.nan
         zp = numpy.zeros(linesamples) + mixgrid
         world = (xw, yw, unknown)
         pixel = (unknown, unknown, zp)
         (world, pixel) = gmap.mixed(world, pixel, iter=10)
         #print "WCSGRAT world=",world
         #print "WCSGRAT pixel=",pixel
      if wcsaxis == 0 or wcsaxis == 1:
         self.axtype = gmap.types[wcsaxis]
      else:
         # E.g. grid lines made without a constant value do not belong to an axis
         # so we use a dummy type 'border' 
         self.axtype = 'border'
         
      self.skysys = skysys
      self.wcsaxis = wcsaxis

      if wcsaxis in [0,1]:
         if gmap.types[wcsaxis] == 'longitude':
            # This must be a longitude axis. We want the range to be limited to [0,360>
            while constval < 0.0:
               constval += 360.0
            while constval >= 360.0:
               constval -= 360.0
      self.constval = constval
      self.linepieces = []
      # For each plot axis we store the ticks that belong to that axis.
      # A tick also belongs to a wcs axis (e.g. a longitude or latitude)
      # but this is then an attribute of the tick
      self.ticks = []
      self.kwargs = {}
      # Special care for 'jumps' in a plot. A jump can occur for example near 180 deg where
      # for an all sky plot 180 deg is a position left in the box while a world
      # coordinate > 180 deg is a position at the right side of the box. 
      # When stepping near these (plot) discontinuities, we want to avoid to connect 
      # positions that are more than factor*step size separated from each other. 
      lastx = lasty = None                 # Flag for reset of counters is set after a 'jump'
      countin = 0; countout = 0
      lastinside = False
      stepy = (pylims[1] - pylims[0] + 1.0)/ linesamples
      stepx = (pxlims[1] - pxlims[0] + 1.0)/ linesamples
      box = (pxlims[0]-0.5, pylims[0]-0.5, pxlims[1]+0.5, pylims[1]+0.5)
      for p in zip(pixel[0], pixel[1]):     # Works for 2 and 3 element world tuples
         xp = p[0]; yp = p[1]
#         print "WCSGRAT xp, yp", xp, yp
         #if wcsaxis == 0 and numpy.isnan(xp):
         #   print xp, yp, box, constval
         if not numpy.isnan(xp) and not numpy.isnan(yp):  # NaN's can occur with divergent projections
            currentinside = __inbox(xp, yp, box)
            if lastx is not None  and (lastinside or currentinside):
               # These jump conditions are somewhat arbitrary.
               # If the projection diverges then these steps sizes may be
               # not enough to have a full coverage.
               jump = abs(lastx-xp) > abs(10.0*stepx) or abs(lasty-yp) > abs(10.0*stepy)
               if jump:
                  # print "JUMP: ", lastx, lasty, xp, yp, abs(lastx-xp), abs(lasty-yp), stepx, stepy
                  if countin > 0:
                     self.linepieces.append( (x,y) )   # Close current line piece
                  countout = 0; countin = 0
                  lastx = lasty = None;
            if countin == 0:
               x = []; y = []
            crossing2in = crossing2out = False
            if currentinside:
               if countout > 0:
                  # We are going from outside to inside
                  crossing2in = True
                  countout = 0
               else:
                  x.append(xp); y.append(yp)
               countin += 1
            else:
               if countin > 0:
                  # We are crossing from inside to outside
                  crossing2out = True
                  countin = 0
               countout += 1            
            if crossing2in or crossing2out:
               axisnr, xlab, ylab = __handlecrossing(box, lastx, lasty, xp, yp)
               if self.axtype != 'border':    # Border lines sometimes do not have a constant world coordinate
                  if offsetlabel is not None:
                     labelvalue = offsetlabel
                     offs = True
                  else:
                     labelvalue = constval
                     offs = False
                  tick = WCStick(xlab, ylab, axisnr, labelvalue, wcsaxis, offs, fun=fun, fmt=fmt)
                  # Set a new default for the font size for the tick labels
                  if wcsaxis == 0:
                     tick.kwargs.update({'fontsize':'10'})
                  else:
                     tick.kwargs.update({'fontsize':'10'})
                  self.ticks.append(tick)
               x.append(xlab); y.append(ylab)  # Add this intersection as element of the graticule line
            if crossing2out:
               self.linepieces.append( (x,y) )
            lastx = xp; lasty = yp
            lastinside = currentinside

      if countin > 0:                   # Store what is left over
         self.linepieces.append( (x,y) )



class PLOTaxis(object):
   """
   -------------------------------------------------------------------------------
   Each (plot) axis can have different properties related to the ticks,
   and the labels. Labels can be transformed using an external function
   ------------------------------------------------------------------------------
   """
   def __init__(self, axisnr, mode, label, **kwargs):
      """
      -----------------------------------------------------------
      Purpose:      Create object that represents an (plot) axis
                    and store default attributes. Only its 
                    attributes should be used by a user/programmer.
      Parameters:
       axisnr -     0: left 
                    1: bottom 
                    2: right
                    3: top
       mode -       What should this axis do with the tick
                    marks and labels?
                    0: ticks native to axis type only
                    1: only the tick that is not native to axis type
                    2: both types of ticks (map could be rotated)
                    3: no ticks
       label -      An annotation of the current axis
      Returns:      Object with attributes 'axisnr', 'mode',
                    'label' and 'kwargs'
      Notes:        Each plot axis is associated with a PLOTaxis
                    instance.
      -----------------------------------------------------------
      """
      self.axisnr = axisnr         # left=0, bottom=1, right=2, top=3
      self.mode = mode             # Set which (native/not native) labels should be stored for this axis 
      self.label = label           # Default title for this axis
      self.kwargs = kwargs         # Keyword aguments for the axis labels
      self.xpos = None             # Position of label (normalized dev. coords)
      self.ypos = None



class Insidelabels(object):
   """
   A small utility class for wcs labels inside a plot with a graticule.
   Useful for all sky plots.

   .. automethod:: setp_label
   """
   class Ilabel(object):
      def __init__(self, Xp, Yp, Xw, Yw, labval, rots, axtype,
                   skysys=None, fun=None, fmt=None,
                   offset=False, prec=0, tex=True, texsexa=True, **kwargs):
         self.x = Xp
         self.y = Yp
         self.xw = Xw
         self.yw = Yw
         self.label = ""
         self.labval = labval
         self.rotation = rots
         self.axtype = axtype
         self.skysys = skysys
         self.offset = offset
         self.fmt = fmt
         self.fun = fun
         self.prec = prec
         self.tex = tex
         self.texsexa = texsexa
         if not tex:
            self.texsexa = False
         self.kwargs = kwargs


   def __init__(self, wcsaxis):
      self.labels = []
      self.ptype = "Insidelabels"
      self.pxlim = None
      self.pylim = None
      self.wcsaxis = wcsaxis


   def append(self, Xp, Yp, Xw, Yw, labval, rots, axtype, skysys=None,
              fun=None, fmt=None, offset=False, prec=0, tex=True, texsexa=True,
              **kwargs):
      #-----------------------------------------------------------------
      """
      Append a new 'Ilabel' object to the list
      """
      #-----------------------------------------------------------------
      if not tex:           # Overrule input if LaTeX is not used
         texsexa = False
      ilab = self.Ilabel(Xp, Yp, Xw, Yw, labval, rots, axtype,
                         skysys=skysys, fun=fun, fmt=fmt,
                         offset=offset, prec=prec, tex=tex, texsexa=texsexa,
                         **kwargs)
      self.labels.append(ilab)


   def setp_label(self, position=None,
                  tol=1e-12, fmt=None, fun=None, tex=None, texsexa=None,
                  **kwargs):
      #-----------------------------------------------------------------
      """
      This method handles the
      properties of the 'inside' labels, which are :class:`Text` objects
      in Matplotlib. The most useful properties are *color*, *fontsize*
      and *fontstyle*. One can change the label values using an external
      function and/or change the format of the label.
      
      :param position:
            Accepted are None, or one or more values representing
            the constant value of the graticule line in
            world coordinates. These positions are used to identify
            individual graticule lines so that each line can have its
            own properties. If no position is entered, then the changes
            are applied to all the labels in the current object.
            The input can also be a string that represents a sexagesimal number.
      :type position: *None* or one or a sequence of floating point numbers
      
      :param tol:
            If a value > 0 is given, the gridline with the
            constant value closest to a given position within
            distance 'tol' gets updated attributes.
      :type tol: Floating point number
      
      :param fmt:
            A string that formats the tick value
            e.g. ``fmt="%10.5f"`` in the Python way, or a string
            that contains no percentage character (%) but a format
            to set the output of sexagesimal numbers e.g.
            fmt='HMs'. The characters in the format either force
            (uppercase) a field to be printed, or it suppresses
            (lowercase) a field to be printed.
            See also the examples at :func:`wcsgrat.makelabel`.
      :type fmt: String
      
      :param fun:
            An external function which will be used to
            convert the tick value e.g. to convert
            velocities from m/s to km/s. See also
            example 2_ below.
      :type fun: Python function or Lambda expression

      :param tex:
            If True then format the tick label in LaTeX. This is the
            default. If False then standard text will be applied.
            Some text properties cannot be changed if LaTeX is
            in use.
      :type tex: Boolean

      :param texsexa:
            If False and parameter *tex* is True, then format the
            tick label without superscripts for sexagesimal labels.
            This option can be used if superscripts result in 'jumpy' labels.
            The reason is that in Matplotlib the TeX labels at the bottom
            of a plot are aligned at a baseline at the top of the characters and
            not at the bottom, while the height between LaTeX boxes may vary.
      :type tex: Boolean
      
      :param `**kwargs`:
            Keyword arguments for plot properties like *color*,
            *visible*, *rotation* etc. The plot attributes are standard
            Matplotlib attributes which can be found in the
            Matplotlib documentation.
      :type `**kwargs`: Matplotlib keyword arguments
      

      :note:
            Some projections generate labels that are very close
            to each other. If you want to skip labels then you can
            use keyword/value *visible=False*. Note that *visible*
            is a parameter of Matplotlib's plot functions.
      """
      #-----------------------------------------------------------------
      if not tex is None:
         if not tex:
            texsexa = False
      if position is not None:
         if not issequence(position):
            posn = [position]
         else:
            posn = position

      for label in self.labels:
         if position is None:
            if not fmt is None:
               label.fmt = fmt
            if not fun is None:
               label.fun = fun
            if not tex is None:
               label.tex = tex
            if not texsexa is None:
               label.texsexa = texsexa
            label.kwargs.update(kwargs)
         else:      # One or more positions are given. Find the right index.
            for pos in posn:
               if isinstance(pos, six.string_types):
                  C = pos.upper()
                  if 'H' in C or 'D' in C or 'M' in C or 'S' in C:
                     pos, err = parsehmsdms(pos)
                     if err != '':
                        raise Exception(err)
                     else:
                        pos = pos[0]
                  else:
                     raise ValueError("[%s] is entered as a string but does not represent valid HMS or DMS"%pos)
               if self.wcsaxis == 0:
                  d = abs(label.xw - pos)
               else:
                  d = abs(label.yw - pos)
               if d <= tol:
                  if not fmt is None:
                     label.fmt = fmt
                  if not fun is None:
                     label.fun = fun
                  if not tex is None:
                     label.tex = tex
                  if not texsexa is None:
                     label.texsexa = texsexa
                  label.kwargs.update(kwargs)


   def plot(self, frame):
      """
      -----------------------------------------------------------
      Purpose:         Plot world coordinate labels inside a plot
      Parameters:      'frame', a Matplotlib Axes object

      Returns:         --
      Notes:
      -----------------------------------------------------------
      """
      createlabels(self.labels)
      # With Insidelabels there is no problem with the alignment
      # with TeX labels, because the alignment is not 'top'.
      # There is no need to tweak the HMS labels here.
      for inlabel in self.labels:
         frame.text(inlabel.x, inlabel.y, inlabel.label,
                    clip_on=True, **inlabel.kwargs)

      # Set limits
      xlo = self.pxlim[0]-0.5
      ylo = self.pylim[0]-0.5
      xhi = self.pxlim[1]+0.5
      yhi = self.pylim[1]+0.5
      frame.set_xlim((xlo,xhi))
      frame.set_ylim((ylo,yhi))


def getlambda(pixel, lo, hi):
#-----------------------------------------------------------------------------
   """
   Small utility to calculate lambda on a line for given position
   in pixels
   """
#-----------------------------------------------------------------------------
   if pixel is None:
      return 0.5
   delta = hi - lo
   if delta == 0.0:
      return 0.5
   return (pixel-lo)/(delta)


def getStarts(startvals, proj, i, axnum, wcstype, mixpix=None):
#------------------------------------------------------------------------------
   """
   Helper function for Graticule contructor.
   This function converts user supplied positions on an axis to numbers
   if the values are given as one string. It extends the functionality
   of function str2pos(). Its main purpose is to set one or more values for
   axis labels if a user does not want the default values as calculated
   in the constructor of class Graticule.

   Notes:

   The projection object 'proj' represents two axes of the map if
   the map is either spatial or does not have any spatial axis.
   If the map has only one spatial axis then a third axis is added
   to the projection object and the valye of mixpix is set. Note that
   function str2pos() diminishes the dimensionality with one is
   the value of mixpix is set.

   So distinguish three options:
   1) This function is called with an axis that can be represented by
      a projection object with one axis like a spectral axis or a linear axis.
   2) This function is called with a spatial axis and the other axis in the map
      is not spatial. The value of mixpix should be set.
   3) This function is called with a spatial axis and the other axis in the map
      is also spatial. The value of mixpix is not set in this case.
  
   """
#------------------------------------------------------------------------------
   if startvals is None:
      # This is the default. Do nothing
      return startvals, None, ""

   if not isinstance(startvals, six.string_types):
      # Perhaps the values are one or more floating point numbers
      return startvals, None, ""

   spatial = wcstype in ['lo', 'la']
   if spatial:
      # If the is a value for mixpix then the map had only one spatial axis.
      # The missing spatial axis is then always the last.
      if not mixpix is None:
         subaxnum = (i+1, 3)
      else:
         # But we need a mixpix here and None was given.
         if i == 0:
            mixpix = proj.crpix[1]
            subaxnum = (1,2)
         else:
            mixpix = proj.crpix[0]
            subaxnum = (2,1)
   else:
      # We need only one. i is either 0 (first axis) or 1 (second axis)
      subaxnum = (i+1,)   # Make tuple
      mixpix = None       # Prevent str2pos() to decrease dimensionality !
   subproj = proj.sub(subaxnum)
   world, pixels, units, errmes = str2pos(startvals, subproj, mixpix=mixpix)
   if errmes != '':
      return None, None, errmes
   #w = world.flatten()
   w = world[:,0]  # Get rid of extra spatial if there is one
   p = pixels[:,0]
   if len(w) == 1:
      w = w[0]         # No longer a sequence but a real start value.
      p = p[0]
   return w, p, ""



class Graticule(object):
#-----------------------------------------------------------------
   """
Creates an object that defines a graticule
A (spatial) graticule consists of parallels and  meridians. We extend this to
a general grid so we can cover every type of map (e.g. position velocity maps).

:param header:    Is a Python dictionary or dictionary-like object
                  containing FITS-style keys and values, e.g. a
                  header object from PyFITS.
                  Python dictionaries are used for debugging,
                  or plotting experiments or when you need to
                  define a projection system from scratch.
:type header:     Python dictionary or FITS header object (pyfits.NP_pyfits.HDUList)

:param graticuledata: This is a helper object. It can be any object as long it
                  has attributes:

                  * header
                  * axnum
                  * pxlim
                  * pylim
                  * mixpix
                  * spectrans

                  Software that interfaces with a user to get data
                  and relevant properties could/should produce objects which
                  have at least values for the attributes listed above.
                  Then these objects could be used as a shortcut parameter.
:type graticuledata:  Object with some required attributes

:param axnum:     This parameter sets which FITS axis corresponds
                  to the x-axis of your graticule plot rectangle
                  and which one corresponds to the y-axis
                  (see also description at *pxlim* and *pylim*).
                  The first axis in a FITS file is axis 1.
                  If *axnum* set to *None* then the default
                  FITS axes will be 1 and 2.
                  With a sequence you can set different FITS axes
                  like ``axnum=(1,3)`` Then the input is a tuple
                  or a list.
:type axnum:      None, Integer or sequence of Integers

:param wcstypes:  List with the type of the used axes. These types are
                  derived from the projection object axis types (attribute
                  wcstype) but are translated into a string: The strings
                  are 'lo' for a longitude axis, 'la' for a latitude axis,
                  'sp; for a spectral axis and 'li_xxx' for a linear axis
                  where 'xxx' is the ctype for that axis.
:type wcstypes:   List of strings

:param pxlim:     The values of this parameter together with
                  the values in pylim define a rectangular frame.
                  The intersections of graticule lines with this
                  frame are the positions where want
                  to plot a tick mark and write a label that
                  gives the position as a formatted string.
                  Further, the limits in pixels are used to set
                  the step size when a graticule line is sampled.
                  This step size then is used to distinguish
                  a valid step from a jump (e.g. from 180-delta
                  degrees to 180+delta degrees which can jump from one side
                  in the plot to the other side).
                  To prevent a jump in a plot, the graticule
                  line is splitted into line pieces without jumps.
                  The default of *pxlim* is copied from the header
                  value. FITS
                  data starts to address the pixels with 1 and the last pixel
                  is given by FITS keyword *NAXISn*.
                  Note that internally the enclosing rectangle
                  in pixels is enlarged with 0.5 pixel in all
                  directions. This enables a correct overlay on an
                  image where the pixels have a size.
:type pxlim:      *None* or exactly 2 Integers

:param pylim:     See description at pxlim. The range is along the
                  y-axis.
:type pylim:      *None* or exactly 2 Integers

:param mixpix:    For maps with only 1 spatial coordinate we need to
                  define the pixel that sets the spatial value
                  on the matching spatial axis. If its value is
                  *None* then the value of *CRPIXn* of the matching
                  axis from the header is taken as default.
:type mixpix:     *None* or 1 Integer

:param spectrans: The spectral translation. For spectral axes
                  it is usually possible to convert to another
                  representation. For instance one can 'translate'
                  a frequency into a velocity which is one of
                  the types: VOPT-F2W, VRAD, VELO-F2V
                  (for optical, radio and radial velocities).
                  See also the article
                  `Representations of spectral coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/scs.pdf>`_
                  by Greisen, Calabretta, Valdes & Allen.
                  Module *maputils* from the Kapteyn Package provides
                  a method that creates a list with possible spectral
                  translations given an arbitrary header.
                  The spectral translation should be followed by
                  a code (e.g. as in 'VOPT-F2W') which sets the
                  conversion algorithm. If you
                  don't know this beforehand, you can either append
                  the string '-???' or try your translation without this
                  coding. Then this module tries to find the appropriate
                  code itself.
:type spectrans:  String

:param skyout:    A single number or a tuple which specifies
                  the celestial system.
                  The tuple is laid out as follows:
                  ``(sky system, equinox, reference system,
                  epoch of observation)``.
                  Predefined are the systems:

                     * wcs.equatorial
                     * wcs.ecliptic,
                     * wcs.galactic
                     * wcs.supergalactic

                  or the minimal matched string versions of these values.
                  
                  Predefined reference systems are:

                     * wcs.fk4,
                     * wcs.fk4_no_e,
                     * wcs.fk5,
                     * wcs.icrs,
                     * wcs.j2000

                  or the minimal matched string versions of these values.

                  Prefixes for epoch data are:

                     =============  =================== ======================================
                     Prefix         Description         Example
                     =============  =================== ======================================
                     B              Besselian epoch     'B 1950', 'b1950', 'B1983.5', '-B1100 
                     J              Julian epoch        'j2000.7', 'J 2000', '-j100.0'       
                     JD             Julian Date         'JD2450123.7'                        
                     MJD            Modified Julian Day 'mJD 24034', 'MJD50123.2'            
                     RJD            Reduced Julian Day  'rJD50123.2', 'Rjd 23433'            
                     F              DD/MM/YY (old FITS) 'F29/11/57'                          
                     F              YYYY-MM-DD          'F2000-01-01'                         
                     F              YYYY-MM-DDTHH:MM:SS 'F2002-04-04T09:42:42.1'             
                     =============  =================== ======================================
 
                
                  See the documentation of module :mod:`celestial` for more details.
                  Example of a sky definition::
   
                     skyout = (wcs.equatorial, wcs.fk4_no_e, 'B1950')
                  
:type skyout:     *None*, one Integer or a tuple with a sky definition

:param alter:     A character from 'A' through 'Z', indicating
                  an alternative WCS axis description from a FITS header.
:type alter:      Character

:param wxlim:     Two numbers in units of the x-axis. For spatial
                  axes this is usually in degrees. The numbers
                  are the limits of an interval for which
                  graticules will be calculated. If these values
                  are omitted, defaults will be calculated.
                  Then random positions in pixels are converted to
                  world coordinates and the greatest gap in
                  these coordinates is calculated. The end- and
                  start point of the gap are the start- and end point
                  of the range(s) in world coordinates. It is not enough
                  to transform only the limits in pixels because a maximum
                  or minimum in world coordinates could be located
                  on arbitrary pixel positions depending on the projection.
:type wxlim:      *None* or exactly two floating point numbers

:param wylim:     See wxlim, but now applied for the y-axis
:type wylim:      *None* or exactly two floating point numbers

:param boxsamples: Number of random pixel positions within a box
                   with limits *pxlim* and *pylim* for which world
                   coordinates are calculated to get an estimate of
                   the range in world coordinates (see description
                   at wxlim). The default is listed in the argument list
                   of this method. If speed is essential one can try smaller
                   numbers than the default.
:type boxsamples:  Integer

:param startx:    If one value is given then this is the
                  first graticule line that has a constant
                  x **world coordinate** equal to *startx*.
                  The other values will be
                  calculated, either with distance *deltax*
                  between them or
                  with a default distance calculated by this method.
                  If *None* is set, then a suitable value will be
                  calculated.
                  The input can also be a string which is parsed by
                  the positions module. This enables the use of
                  units etc.
                  Examples (see also module :mod:`positions`:
                  
                    * For a frequency axis: startx="linspace(1.4240,1.4250,4) Ghz"
                    * For a frequency axis: startx="arange(1.4240,1.4250,0.0005) Ghz"
                    * For a spectral translation to WAVE: startx="'0.2105, 0.2104' m"
                    * Two labels on a longitude axis: startx="3h00m20s 3h00m30s"


:type startx:     *None* or 1 floating point number or a sequence
                  of floating point numbers or a string.

:param starty:    [None, one value, sequence]
                  Same for the graticule line with constant
                  y world coordinate equal to starty.
:type starty:     *None* or 1 floating point number or a sequence
                  of floating point numbers or a string.

:param deltax:    Step in **world coordinates** along the x-axis
                  between two subsequent graticule lines.
                  It can also be a string with an expression and optionally
                  a unit. Note that the expression cannot contain any spaces.
                  Example:
                  
                    * deltax = 5*6/6 dmsmin
                    
:type deltax:     *None* or a floating point number or a string

:param deltay:    Same as deltax but now as step in y direction.
                  It can also be a string with an expression and optionally
                  a unit.
:type deltay:     *None* or a floating point number or a string.

:param skipx:     Do not calculate the graticule lines with the
                  constant world coordinate that is associated
                  with the x-axis.
:type skipx:      Boolean

:param skipy:     The same as skipx but now associated with
                  the y-axis.
:type skipy:      Boolean

:param gridsamples: Number of positions on a graticule line for which
                    a pixel position is calculated and stored as part
                    of the graticule line. If *None* is set then the
                    default is used (see the argument list of this method).
:type gridsamples:  Integer

:param labelsintex: The default is that all tick labels are formatted for LaTeX.
                    These are not the axes labels. If you want to format these in
                    LaTeX then you need to set them explicitly as in:
   
                    >>> grat.setp_axislabel("bottom",
                        label=r"$\mathrm{Right\ Ascension\ (2000)}$",
                        fontsize=14)``

                    Printing your axis labels in LaTeX
                    limits the number of Matplotlib properties that one
                    can set.

:type labelsintex:  Boolean

:param offsetx:     Change the default mode which sets either plotting
                    the labels for the given -or calculated world coordinates
                    or plotting labels which represent constant offsets
                    with respect to a given starting point.
                    The offset mode is default for plots with mixed axes,
                    i.e. with only one spatial axis. In spatial maps
                    this offset mode is not
                    very useful to plot the graticule lines because these lines
                    are plotted at a constant world coordinate and do not know
                    about offsets.
                    The offset axes correspond to the pixel positions of
                    start- and endpoint of
                    the left and bottom axes and the default start point of
                    the offsets (value 0) is at the centre of the axis.
                    One can change this start point with *startx*, *starty*.
:type offsetx:      *None* or Boolean

:param offsety:     Same as *offsetx* but now for the left plot axis.
:type offsety:      *None* or Boolean

:param unitsx:      Units for first axis. Applies both to regular and offset axes.
                    If this parameter sets a unit other than the default,
                    then a conversion function will be used to display the
                    labels in the new units. The unit in the default axis label
                    will be replaced by the new units.
:type unitsx:       String

:param unitsy:      Units for second axis.
:type unitsy:       String


:raises:
   :exc:`ValueError` *Could not find enough (>1) valid world coordinates in this map!*
      User wanted to let the constructor estimate what the ranges in
      world coordinates are for this header, but only zero or one
      coordinate could be found.
   
   :exc:`ValueError` *Need data with at least two axes*
      The header describes zero or one axes. For a graticule
      plot we need at least two axes.
   
   :exc:`ValueError` *Need two axis numbers to create a graticule*
      The *axnum* parameter needs exactly two values.
   
   :exc:`ValueError` *Need two different axis numbers*
      A user/programmer entered two identical axis numbers.
      Graticules need two different axes.
   
   :exc:`ValueError` *pxlim needs to be of type tuple or list*
      Check type.
   
   :exc:`ValueError` *pxlim must have two elements*
      Number must be exactly 2.
   
   :exc:`ValueError` *pylim needs to be of type tuple or list*
      Check type.
   
   :exc:`ValueError` *pylim must have two elements*
      Number must be exactly 2.
   
   :exc:`ValueError` *Could not find a grid for the missing spatial axis*
      The specification in *axnum* corresponds to a map with only one
      spatial axis. If parameter *mixpix* is omitted then the constructor
      tries to find a suitable value from the (FITS) header. It
      reads *CRPIXn* where n is the appropriate axis number. If nothing
      could be found in the header then this exception will be raised.

   :exc:`ValueError` *Could not find a matching spatial axis pair*
      The specification in *axnum* corresponds to a map with only one
      spatial axis. A We need the missing spatial axis to find a
      matching world coordinate, but a matching axis could not be found
      in the header.
   
   
   :exc:`ValueError` *wxlim needs to be of type tuple or list*
      Check type.
      
   :exc:`ValueError` *wxlim must have two elements*
      Number must be exactly 2.
      
   :exc:`ValueError` *wylim needs to be of type tuple or list*
      Check type.
      
   :exc:`ValueError` *wylim must have two elements*
      Number must be exactly 2.
      
   :exc:`ValueError` *boxsamples < 2: Need at least two samples to find limits*
      There is a minimum number of random positions we have to
      calculate to get an impression of the axis limits in world coordinates.

   :exc:`ValueError` *Number of samples along graticule line must be >= 2 to avoid a step size of zero*
      The value of parameter *gridsamples* is too low. Low values give
      distorted graticule lines. Higher values (like the default) give
      smooth results.


:Returns:         A graticule object. This object contains the line
                  pieces needed to draw the graticule and the
                  ticks (positions, text and axis number).
                  The basis method to reveal this data (necessary
                  if you want to make a plot yourself) is described in the
                  following example::

                     graticule = wcsgrat.Graticule(header)
                     for gridline in graticule:
                     print "\\nThis gridline belongs to axis", gridline.wcsaxis
                     print "Axis type: %s.  Sky system %s:" % (gridline.axtype, gridline.skysys)
                     for t in gridline.ticks:
                        print "tick x,y:", t.x, t.y
                        print "tick label:", t.labval
                        print "tick on axis:", t.axisnr
                     for line in gridline.linepieces:
                        print "line piece has %d elements" % len(line[0])

.. Note::
                  A Graticule object has a string representation and can
                  therefore be easily inspected with Python's **print** statement. 

**Attributes:**
                        
.. attribute::    axes
                        
                  Read the PLOTaxis class documentation.
                  Four PLOTaxis instances, one for each axis of the
                  rectangular frame in pixels set by *xplim* and *pylim*
                  If your graticule object is called **grat** then
                  the four axes are accessed with:
                        
                     * grat.axes[wcsgrat.left]
                     * grat.axes[wcsgrat.bottom]
                     * grat.axes[wcsgrat.right]
                     * grat.axes[wcsgrat.top]
                  
                  Usually these attributes are set with method :meth:`setp_plotaxis()`.

                  Examples:

                  ::
                        
                     grat.axes[wcsgrat.left].mode = 1
                     grat.axes[wcsgrat.bottom].label = 'Longitude / Latitude'
                     grat.axes[wcsgrat.bottom].mode = 2
                     grat.axes[wcsgrat.right].mode = 0

                  ::
                        
                     PLOTaxis modes are:
                         
                     0: ticks native to axis type only
                     1: Only the tick that is not native to axis type
                     2: both types of ticks (map could be rotated)
                     3: no ticks
                  
                  The default values depend on how many ticks, native
                  to the plot axis, are found. If this is < 2 then
                  we allow both native and not native ticks along 
                  all plot axes.

.. attribute:: pxlim

                  The limits of the map in pixels along the x-axis.
                  This value is either set in the constructor or 
                  calculated. The default is *[1,NAXISn]*.
                  The attribute is meant as a read-only attribute.

.. attribute:: pylim:

                  Same for the y-axis.

.. attribute:: wxlim

                  The limits of the map in world coordinates for the
                  x-axis either set in the constructor or calculated
                  (i.e. estimated) by this method. The attribute is
                  meant as a read-only attribute.

.. attribute:: wylim

                  Same for the y-axis

.. attribute:: xaxnum

                  The (FITS) axis number associated with the x-axis
                  Note that axis numbers in FITS start with 1. If these
                  numbers are not given as argument for the 
                  constructor then *xaxnum=1* is assumed.
                  The attribute is
                  meant as a read-only attribute.

.. attribute:: yaxnum

                  Same for the y-axis.
                  Default: *yaxnum=2*

.. attribute:: wcstypes

                  List with strings that represent the wcs axis type
                  of the axes.
                  
.. attribute:: gmap

                  The wcs projection object for this graticule.
                  See the *wcs* module document for more information.

.. attribute:: mixpix

                  The pixel on the matching spatial axis for maps
                  with only one spatial axis. This attribute is 
                  meant as a read-only attribute. 

.. attribute:: xstarts

                  World coordinates associated with the x-axis
                  which set the constant value of a graticule line
                  as calculated when the object is initialized.
                  This attribute is meant as a read-only attribute.

.. attribute:: ystarts
   
                  Same for the y-axis

.. attribute:: skyout

                  Unformatted copy of input parameter *skyout*

.. attribute:: spectrans

                  Unformatted copy of input parameter *spectrans*


:Examples:        Example to show how to use a custom made header to
                  create a graticule object. Usually one uses this option
                  to create **all sky** plots. It is also a useful tool
                  for experiments.::

                     #1. A minimal header for an all sky plot
                     header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                               'CTYPE1' : 'RA---AZP', 'CRVAL1' :0, 
                               'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                               'CTYPE2' : 'DEC--AZP',
                               'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg',
                               'CDELT2' : 5.0,
                               'PV2_1'  : mu, 'PV2_2'  : gamma,
                              }
                     grat = wcsgrat.Graticule(header)

                  Use module `PyFITS <http://www.stsci.edu/resources/software_hardware/pyfits>`_
                  to read a header from a FITS file::

                     #2. A header from a FITS file 'test.fits'
                     import pyfits
                     hdulist = pyfits.open('test.fits')
                     header = hdulist[0].header
                     grat = wcsgrat.Graticule(header)

                  Select the axes for the graticules. Note that the order
                  of the axes should be the same as the order of axes in
                  the image where you want to plot the graticule. If necessary
                  one can swap the graticule plot axes with input parameter
                  *axnum*::

                     #3. Swap x and y- axis in a FITS file
                     grat = wcsgrat.Graticule(header, axnum= (2,1))

                  For data with more than two axes, one can select the axes
                  with input parameter *axnum*::

                     #4. For a FITS file with axes (RA,DEC,FREQ) 
                     #  create a graticule for the FREQ,RA axes:
                     grat = wcsgrat.Graticule(header, axnum=(3,1))

                  Use sexagesimal numbers for *startx*/*starty*::

                     #5. Sexagesimal input
                     grat = wcsgrat.Graticule(...., startx="7h59m30s", starty="-10d0m30s')

**Methods which set (plot) attributes:**

.. automethod::   setp_tick
.. automethod::   setp_plotaxis
.. automethod::   setp_lineswcs0
.. automethod::   setp_lineswcs1
.. automethod::   setp_gratline
.. automethod::   setp_axislabel
.. automethod::   setp_tickmark
.. automethod::   setp_ticklabel
.. automethod::   set_tickmode

**Methods that deal with special curves like borders:**

.. automethod::   scanborder
.. automethod::   addgratline
.. automethod::   setp_linespecial

**Methods related to plotting derived elements:**

.. automethod::   Insidelabels
.. automethod::   Insidelabels.setp_label

**Utility methods:**

.. automethod::   get_aspectratio
   """
   #-----------------------------------------------------------------

   @staticmethod
   def __bisect(direct, const, var1, var2, gmap, tol):
      """
      -----------------------------------------------------------
      Purpose:     Apply bisection to find the position of 
                   a limb (border in a plot).
      Parameters: 
       direct:     direct=0: Apply bisection in y-direction
                   direct=1: Apply bisection in x-direction
       const:      Pixel position of the axis along which 
                   a bisection applied
       var1, var2: Lower and upper limits in pixels of the 
                   interval along which bisection is applied.
       gmap:       The wcs projection object (to apply methods
                   toworld and topixel
       tol:        The tolerance in pixels used as a stop 
                   criterion for the bisection.

      Returns:     The pixel position in range [var1, var2] 
                   which represent a position on the border.

      Notes:       A limb in terms of plotting is a border
                   which defines the regions where conversions 
                   from and to world coordinates is possible.
                   Those positions where this is not possible
                   are represented by NaN's. So we try to find 
                   the position where there is a transition
                   between a number and a NaN within a certain
                   precision given by parameter 'tol'.
                   The bisection is applied either in x- or
                   y-direction at value 'const'
                   and the start interval is between var1 and var2.
      -----------------------------------------------------------
      """
      Nmax = 100
      if direct == 0:  # Vertical bisection
         xw1, yw1 = gmap.toworld((const, var1))
         xw2, yw2 = gmap.toworld((const, var2))
         vw1 = yw1; vw2 = yw2
      else:
         xw1, yw1 = gmap.toworld((var1, const))
         xw2, yw2 = gmap.toworld((var2, const))
         vw1 = xw1; vw2 = xw2

      # One position must be a number and the other must be a NaN
      if (not (numpy.isnan(vw1) or numpy.isnan(vw2))) or (numpy.isnan(vw1) and numpy.isnan(vw2)):
         return None;
      if numpy.isnan(vw1):
         vs = var1
         ve = var2
      else:
         vs = var2
         ve = var1
      i = 0
      while i <= Nmax:
         vm = (vs + ve)/2.0
         if direct == 0: 
            xw, yw = gmap.toworld((const, vm))
            v = yw
         else:
            xw, yw = gmap.toworld((vm, const))
            v = xw
         if numpy.isnan(v):
            vs = vm
         else:
            ve = vm
         if abs(ve-vs) <= tol:
            break
         i += 1
      return vs


   # @staticmethod
   def sortticks(self, tex=False):
      """
      ----------------------------------------------------------
      Purpose:    Collect ticks for each plot axis
                  Format the labels if appropriate.
      Parameters: 
       tex:       True or False. If True then render the labels
                  in TeX.

      Returns:    tickpos, ticklab, tickkwa which are all 4 lists
                  of with tick information per plot axis.

      Notes:      The ticks are sorted per axis because then one
                  can set properties for all ticks that belong
                  to one axis.
      -----------------------------------------------------------
      """
      global tweakhms
      ticks    = [[],[],[],[]]
      tickpos  = [[],[],[],[]]
      ticklab  = [[],[],[],[]]
      tickkwa  = [[],[],[],[]]
      tickMkwa = [[],[],[],[]]
      for gridline in self.graticule:
          wcsaxis = gridline.wcsaxis
          for t in gridline.ticks:
             anr = t.axisnr
             mode = self.axes[anr].mode
             skip = False
             if mode == 0:  # Include a tick. Select ticks along axis with axis 'mode'.
                # Plot only tick labels near the relevant axes
                if  wcsaxis == 0 and (anr == left or anr == right):
                   skip = True
                if  wcsaxis == 1 and (anr == bottom or anr == top):
                   skip = True
             elif mode == 1:
                # Plot only the 'not native' tick labels near the relevant axes
                if  wcsaxis == 1 and (anr == left or anr == right):
                   skip = True
                if  wcsaxis == 0 and (anr == bottom or anr == top):
                   skip = True
             # Mode == 2 allows all ticks for this axis
             elif mode == 3:
                # No ticks at all option
                skip = True

             if not skip:
                if anr in [left,right]:
                   tickpos[anr].append(t.y)
                else:
                   tickpos[anr].append(t.x)
                tickkwa[anr].append(t.kwargs)
                tickMkwa[anr].append(t.markkwargs)
                #ticklab[anr].append(t.label)
                # Give this tick some extra attributes necessary for formatting its label
                t.axtype = gridline.axtype
                t.skysys = gridline.skysys
                t.prec = self.prec[wcsaxis]
                t.delta = self.delta[wcsaxis]
                # Tex mode could have been changed by a set properties method
                if t.tex is None:
                   t.tex = tex
                ticks[anr].append(t)

      # We reached the stage that we assembled all ticks that belong
      # to an axis. This implies that the ticks can belong to different
      # wcs axes. For the formatting of the labels one needs to distinguish
      # those.
       
      for anr in range(0,4):
         for wcsaxis in [0,1]:
            Tlist = []
            for t in ticks[anr]:
               if t.wcsaxis == wcsaxis:
                  Tlist.append(t)
            # Modify the labels. Note that the ticklab list points to the
            # label attribute of a WCStick object. The createlabels() function
            # returns a list so in fact if this function changes a label for
            # a tick object, then
            # the label will automatically be updated in the list also.
            createlabels(Tlist)
         #if anr == bottom:
            # Only bottom axes can have TeX labels that are not aligned very well
            # With this tweak parameter this should improve
            #update_metrics(tweakhms)
      for anr in range(0,4):
         for t in ticks[anr]:
            ticklab[anr].append(t.label)

      return tickpos, ticklab, tickkwa, tickMkwa


   @staticmethod
   def __adjustlonrange(lons):
      """
      -----------------------------------------------------------
      Purpose:    Find minimum and maximum of range in longitudes
                  with gaps. E.g. with a crval near zero, one expects
                  values both negative as positive. However the wcs
                  routines return longitudes always in range [0,360>.
                  So -10 appears as 250 in the list. This results
                  in the wrong min and max of this range. This 
                  algorithm calculates the two array values for which 
                  the gap is the biggest. It then returns the correct
                  min and max of the range with the min always
                  as first parameter (allowing for negative values).

      Parameters:
       lons       A (numpy) 1-dim array of longitudes 
                  (world coordinates)

      Returns:    min, max of range of input longitudes, excluding
                  the biggest gap smaller than or equal to 180
                  (degrees) in the range.

      Examples:   1) Longitudes:  [270, 220, 88, 12, 90, 0, 289, 180, 300, 2, 3, 4]
                  Sorted longitudes:  [0, 2, 3, 4, 12, 88, 90, 180, 220, 270, 289, 300]
                  Biggest gap, start longitude, end longitude 90 -180.0 90
                  min, max: -180.0 90
                  2) Longitudes:  [1, 3, 355, 2, 5, 7, 0, 359, 350, 10, 11, 349]
                  Sorted longitudes:  [0, 1, 2, 3, 5, 7, 10, 11, 349, 350, 355, 359]
                  Biggest gap, start longitude, end longitude 22.0 -11.0 11
                  min, max: -11.0 11
      ------------------------------------------------------------
      """
      def __diff_angle(a, b):
         if b < a:
            result = b + 360 - a
         else:
            result = b - a
         if result > 180.0:
            result -= 360.0
         return result

      # Eliminate largest gap
      lons.sort()
      gap = 0.0
      prev  = lons[-1]
      brkpt = 0
 
      for i, lon in enumerate(lons):
         diff = __diff_angle(prev, lon)
         if abs(diff)>gap:
            gap = abs(diff)
            brkpt = i
         prev = lon
      # 'gap' contains now the largest gap
 
      lon1 = lons[brkpt]; lon2 = lons[brkpt-1]
      if lon1 > lon2:
         lon1 -= 360.0

      return lon1, lon2


   @staticmethod
   def __nicenumbers(x1, x2, start=None, delta=None, axtype=None, skysys=None, offset=False):
      """
      -----------------------------------------------------------
      Purpose:    Find suitable numbers to define graticule lines
                  in interval [x1,x2]. Process longitudes and
                  latitudes in seconds.
                  Also return a list with the same length which
                  contains offsets only.

      Parameters: 
       x1, x2:    A start and an end value representing an interval
       start:     If not None, then include this value in the list
                  of nice numbers.
       delta:     If not None, use this value as step size
                  If both 'start' and 'delta' are not None, then use
                  these values to get all the values in the given 
                  interval [x1,x2] with start equal to 'start' and
                  step size equal to 'delta'
       axtype:    None or one of ('longitude', latitude, 'spectral').
                  Value is used to distinguish lons and lats from
                  other data.
       skysys:    For longitudes distinguish ranges 0,360 in hours 
                  (equatorial system) or degrees.

      Returns:    A tuple with:
                 -A NumPy array with 'nice' numbers
                 -The precision in seconds
                 -The proposed step size

      Notes:     Spatial intervals are first multiplied by the
                 appropriate factor to get an interval in seconds.
                 For numbers >= 10 seconds, a 'nice' step size is
                 drawn from a predefined list with numbers.
                 For other intervals, the length of the interval
                 is scaled to an interval between [0,10>. The
                 scaled length sets the number of divisions and
                 the final step size. The output step size is in degrees.

      Examples:  a=3; b = 11 
                 print  nicenumbers(a, b)
                 print  nicenumbers(a, b, start=10)
                 print  nicenumbers(a, b, start=10, delta=1)
                 print  nicenumbers(a, b, delta=2)
                 (array([  6.,   8.,  10.]), 0, 2.0)
                 (array([ 10.,   8.,   6.,   4.]), 0, 2.0)
                 (array([ 10.,   9.,   8.,   7.,   6.,   5.,   4.]), 0, 1.0)
                 (array([ 4.,  6.,  8.]), 0, 2.0)
                 Other examples:
                 nicenumbers(a, b, axtype='longitude', skysys=wcs.equatorial)
                 nicenumbers(a, b, axtype='longitude', skysys=wcs.ecliptic)
                 nicenumbers(a, b, axtype='latitude', skysys=wcs.galactic)
      -----------------------------------------------------------
      """
      prec = 0
      step = None
      #nc = None
      fact = 1.0
      if x2 < x1:          # Then swap because we want x2 > x1
         x1,x2 = x2,x1
      x1orig = x1; x2orig = x2
      dedge = (x2-x1) / 80.0     # 80 is a 'magic' number that prevents 
                                 # graticule lines too close to the borders
      if not issequence(skysys):
         skysys = [skysys]
      if delta is None:
         if axtype in ['longitude', 'latitude']:
            # Nice numbers for dms should also be nice numbers for hms 
            sec = numpy.array([30, 20, 15, 10, 5, 2, 1])
            minut = sec
            deg = numpy.array([60, 30, 20, 15, 10, 5, 2, 1])
            nicenumber = numpy.concatenate((deg*3600.0, minut*60.0, sec))
            if wcs.equatorial in skysys and axtype == 'longitude':
               fact = 240.0
            else:
               fact = 3600.0

            x1 *= fact; x2 *= fact
            d = x2 - x1
            step2 = 0.9*d/3.0         # We want at least two graticule lines in this range
            for p in nicenumber:
               k = int(step2/p)
               if k >= 1.0:
                  step2 = k * p
                  step = step2
                  # nc = nicenumber
                  break           # Stop if we have a candidate

         if step is None:
            d = x2 - x1
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
            
      else:
         step = delta
         
          

      if step == 0.0:        # Return empty list. Cannot divide by zero
         return [], 0,0      # To get just one point, use delta > x2 - x1

      xxm = step*numpy.round(((x1+x2)/2.0)/step)
      xxm /= fact; step /= fact
      # Now both parameters are in the original units.
      # Calculate a precision for the seconds along spatial axes.
      pstep = step
      if axtype in ['longitude', 'latitude']:
         if wcs.equatorial in skysys and axtype == 'longitude':
            pstep *= 240.0
         else:
            pstep *= 3600.0
      # Set the precision for the seconds
      # We found an optimum precision by trial and error
      f0 = numpy.floor( numpy.log10(pstep) )
      if f0 < 0.0:
         prec = int(abs(f0))
      else:
         prec = 0
      startx = None
      if start is not None:
         startx = start
      elif x1orig+dedge < 0.0 < x2orig-dedge:
         startx = 0.0
      else:
         startx = xxm

      l1 = numpy.arange(startx, x1orig+0.9*dedge, -step)
      n1 = len(l1) 
      o1 = numpy.arange(0.0, -(n1-0.01)*step, -step)
      l2 = numpy.arange(startx+step, x2orig-0.9*dedge, step)
      n2 = len(l2)
      o2 = numpy.arange(0.0+step, (n2+0.01)*step, step)
      nice = numpy.concatenate( (l1,l2) )
      offsets = numpy.concatenate( (o1,o2) )
      
      return nice, offsets, prec, step



   def __estimateLonLatRanges(self, nrandomsamples):
      """
      ----------------------------------------------------------
      Purpose:     Given the current pixel limits, find the
                   limits along the x and y directions in 
                   world coordinates.

      Parameters:  
       nrandomsamples:  Number of random pixel position samples

      Returns:     wxmin, wxmax, wymin, wymax
                   The limits in world coordinates

      Notes:       Given the current ranges in pixel coordinates
                   (box), estimate the ranges in world coordinates.
                   The complication is that we are dealing with
                   many different WCS coordinate transformations.
                   Therefore we sample the grid with 
                   'nrandomsamples' random positions for which
                   we assume that the converted longitudes and
                   latitudes can be used to estimate a close
                   indication for these ranges. The edges of the 
                   box are also included in the calculations.
                   If a projection behaves in a way that the edges
                   are also the limits in world coordinates then
                   we automatically get the maximum limits, 
                   otherwise it is an approximation and the quality
                   of this approximation increases if the number
                   of samples increases. 
      -----------------------------------------------------------
      """
      xlo = self.pxlim[0]-0.5; ylo = self.pylim[0]-0.5; 
      xhi = self.pxlim[1]+0.5; yhi = self.pylim[1]+0.5
      # Dx = (xhi - xlo + 1)/10.0; Dy = (yhi - ylo + 1)/10.0
      Dx = Dy = 0.0
      xr = numpy.random.uniform(xlo-Dx, xhi+Dx, nrandomsamples+4)
      yr = numpy.random.uniform(ylo-Dy, yhi+Dy, nrandomsamples+4)
      # Always include the edges of the frame
      xr[0] = xlo; xr[1] = xhi; xr[2] = xlo; xr[3] = xhi;
      yr[0] = ylo; yr[1] = ylo; yr[2] = yhi; yr[3] = yhi;
      if self.mixpix is None:
         pixels = (xr, yr)
      else:
         zr = numpy.zeros(nrandomsamples+4) + self.mixpix
         pixels = (xr, yr, zr)

      world = self.gmap.toworld(pixels)
      # The world coordinates we want are always the first and second
      # element of the result tuple.
      wx = numpy.ma.masked_where(numpy.isnan(world[0]), world[0])
      wy = numpy.ma.masked_where(numpy.isnan(world[1]), world[1])
      if numpy.ma.count_masked(wx) > len(wx)-2:
         raise Exception("Could not find enough (>1) valid world coordinates in this map!")
      wxmin = wx.min(); wxmax = wx.max(); wymin = wy.min(); wymax = wy.max()

      # If one of the axes is a 'longitude' type axis then 
      # we can have 'jumpy' behaviour near 0.0. What we need then is a routine that
      # finds the greatest contiguous sequence in a sorted array.
      # Then we need to filter the NaN's from the list. A masked array cannot
      # be sorted (yet), so compress it to an array with only valid positions
      # and convert to a numpy array.

      if self.gmap.types[0] == 'longitude':
         wx = numpy.asarray(numpy.ma.masked_where(numpy.isnan(world[0]), world[0]).compressed())
         wxmin, wxmax = self.__adjustlonrange(wx)
      if self.gmap.types[1] == 'longitude':
         wy = numpy.asarray(numpy.ma.masked_where(numpy.isnan(world[1]), world[1]).compressed())
         wymin, wymax = self.__adjustlonrange(wy)
      return wxmin, wxmax, wymin, wymax


   def __init__(self, header=None, graticuledata=None, axnum=None,
                wcstypes=None,
                pxlim=None, pylim=None, 
                mixpix=None, spectrans=None, skyout=None, alter='', 
                wxlim=None, wylim=None,
                boxsamples=5000, 
                startx=None, starty=None, deltax=None, deltay=None,
                skipx=False, skipy=False,
                gridsamples=1000,
                labelsintex=True,
                offsetx=None, offsety=None,
                unitsx=None, unitsy=None):
      """
     -----------------------------------------------------------
      Purpose:    Creates an object that defines a graticule
                  See class documentation above.
      -----------------------------------------------------------
      """
      self.ptype = 'Graticule'
      if graticuledata is not None:
         if header is None:
            header = graticuledata.hdr
         axnum  = graticuledata.axperm
         pxlim  = graticuledata.pxlim
         pylim  = graticuledata.pylim
         mixpix = graticuledata.mixpix
         wcstypes = graticuledata.wcstypes
         # Allow these to be overwritten
         if spectrans is None:
            spectrans = graticuledata.spectrans
         if skyout is None:
            skyout = graticuledata.skyout
         if alter =='':
            alter = graticuledata.alter
            
      self.frame = None
      self.frame2 = None
      self.skyout = skyout
      self.spectrans = spectrans
      self.labelsintex = labelsintex
      if wcstypes is None:
         raise Exception("Need a list with wcs types for these axes")
      self.wcstypes = wcstypes
      # Try to get two axis numbers if none are given
      if axnum is None:
         naxis = header['NAXIS']
         if naxis < 2:
            raise Exception("Need data with at least two axes")
         else:
            self.xaxnum = 1
            self.yaxnum = 2
      else:
         if len(axnum) != 2:
            raise Exception("Need two axis numbers to create a graticule")
         else:
           self.xaxnum = axnum[0]
           self.yaxnum = axnum[1]
      if (self.xaxnum == self.yaxnum):
         raise Exception("Need two different axis numbers")
      
      # Get the axes limits in pixels. The default is copied from
      # the values of NAXISn in the header. Note that there are no alternative
      # keywords for NAXIS.
      if pxlim is None:
         self.pxlim = (1, header['NAXIS' +str(self.xaxnum)])
      else:
         if not issequence(pxlim):
            raise Exception("pxlim needs to be of type tuple or list")
         elif len(pxlim) != 2:
            raise Exception("pxlim must have two elements")
         else: 
            self.pxlim = pxlim
      if pylim is None:
         self.pylim = (1, header['NAXIS' +str(self.yaxnum)])
      else:
         if not issequence(pylim):
            raise Exception("pylim needs to be of type tuple or list")
         elif len(pylim) != 2:
            raise Exception("pylim must have two elements")
         else:
            self.pylim = pylim


      # wcs.debug=True
      # Create the wcs projection object      
      proj = wcs.Projection(header, skyout=skyout, alter=alter)
      if spectrans:
         # Spectral transformations must be applied to the Projection object
         # with all her axes because we need it both for plotting
         # graticule label in the right spectral type (one of the image
         # axes is spectral) or we want the output of the pixels on the
         # repeat axis in the right spectral coordinates (if that axis is
         # spectral). If we postpone this translation until we have a
         # 2-D image, the spectral translation will not work for the
         # repeat axis.
         st2 = spectrans.split('-')
         if len(st2) == 1:
            spectrans += '-???'
         proj = proj.spectra(spectrans)
      
      # If one of the selected axes is a spatial axis and the other
      # is not, then try to find the axis number of the matching axis.
      mix = False
      if self.xaxnum == proj.lonaxnum and self.yaxnum != proj.lataxnum:
         self.matchingaxnum = proj.lataxnum
         mix = True
      elif self.xaxnum == proj.lataxnum and self.yaxnum != proj.lonaxnum:
         self.matchingaxnum = proj.lonaxnum 
         mix = True
      if self.yaxnum == proj.lonaxnum and self.xaxnum != proj.lataxnum:
         self.matchingaxnum = proj.lataxnum
         mix = True
      elif self.yaxnum == proj.lataxnum and self.xaxnum != proj.lonaxnum:
         self.matchingaxnum = proj.lonaxnum 
         mix = True
      if mix:
         if mixpix is None:
            mixpix = proj.source['CRPIX'+str(self.matchingaxnum)+proj.alter]
         if mixpix is None:
            raise Exception("Could not find a grid for the missing spatial axis")
         ok = proj.lonaxnum is not None and proj.lataxnum is not None 
         if not ok:
            raise Exception("Could not find a matching spatial axis pair")
         gmap = proj.sub([self.xaxnum, self.yaxnum, self.matchingaxnum])
         self.mixpix = mixpix
      else:
         gmap = proj.sub([self.xaxnum, self.yaxnum])
         self.mixpix = None

      self.gmap = gmap
      self.gmap.allow_invalid = True

      # We arrived at the stage that we have a Projection object (called 'gmap')
      # which represents the current projection system and/or spectral system.
      # For maps with only one spatial axis, a matching axis is given. We already
      # know the pixel limits for the axes in a map (pxlim, pylim).
      # The positions module provides a useful function (str2pos) to convert
      # strings to positions.
      
      # Note that the getStarts function gets a Projection object called
      # 'gmap' which has always a missing spatial axis added if the image
      # has only one spatial axis.
      startx, startpixX, errmes = getStarts(startx, self.gmap, 0, axnum, wcstypes[0], self.mixpix)
      if errmes != '':
         raise Exception(errmes)
      starty, startpixY, errmes = getStarts(starty, self.gmap, 1, axnum, wcstypes[1], self.mixpix)
      if errmes != '':
         raise Exception(errmes)

      # The next line is added at 28-08-2010 and is necessary because
      # the sky system does not need to be an integer number anymore.
      # It can also be a string. Many tests however are based on the integer
      # number so we convert the sky system first to a number.
      self.__skysys, refin, epochin, epobs = skyparser(proj.skyout)

      # Now we have a projection object available and we want the limits of the axes in 
      # world coordinates. If nothing is specified for the constructor, we have
      # to calculate estimates of these ranges.
      if wxlim is not None:
         if not issequence(wxlim):
            raise Exception("wxlim needs to be of type tuple or list")
         elif len(wxlim) != 2:
            raise Exception("wxlim must have two elements")
         else: 
            self.wxlim = wxlim
      if wylim is not None:
         if not issequence(wylim):
            raise Exception("wylim needs to be of type tuple or list")
         elif len(wylim) != 2:
            raise Exception("wylim must have two elements")
         else: 
            self.wylim = wylim
      if wxlim is None or wylim is None:
         if boxsamples < 2:
            raise Exception("boxsamples < 2: Need at least two samples to find limits")
         minmax = self.__estimateLonLatRanges(boxsamples)
         if wxlim is None:
            self.wxlim = (minmax[0], minmax[1])
         if wylim is None:
            self.wylim = (minmax[2], minmax[3])

      # At this point we need to know for which constant positions we need to find a graticule
      # line. We distinguish the following situations:
      # A) The world coordinate is not spatial
      # 1. A set of start values for the graticules are entered. The step (delta) is a dummy.
      # 2. The start values are not given, but a delta is --> find suitable start values
      # 3. Neither start values nor a delta is entered --> find suitable values for both
      # 4. The user wants the labels to be plotted as offsets --> set output in offsets
      #    i.e. plot n*deltax/y instead of the world coordinate in 'startx/y'
      # 
      # B) The world coordinate is spatial the other is not
      # Set defaults. If only one coordinate is spatial then offsets are the default.
      # 1) Values for startx/y are entered --> ignore
      # 2) Value for deltax/y is entered --> use as delta and calculate world coordinates
      #    that correspond to the given delta.
      # Note that this situation is not trivial. We label for a constant missing spatial axis
      # which is given in grids. Then the world coordinate that corresponds to this grid
      # needs not to be constant because the other spatial coordinate varies. We use
      # methods from the ruler class to find world coordinates that represent
      # these offsets as distances on a sphere.
      #
      # C) Both world coordinates are spatial
      # The default should not be an offset. If the user overrides this, then offsets
      # are offsets on the graticule and do not represent distances on a sphere.

      # Check the offsets

      self.offsetx = offsetx
      self.offsety = offsety
      spatialx = self.gmap.types[0] in ['longitude', 'latitude']
      spatialy = self.gmap.types[1] in ['longitude', 'latitude']
      if self.offsetx is None:
         self.offsetx = spatialx and not spatialy
      if self.offsety is None:
         self.offsety = spatialy and not spatialx

      self.prec  = [0, 0]

      # Handle the values for the step in the labels/graticule lines.
      self.delta = [None, None]
      axisunits = self.gmap.units[0]
      if not deltax is None and isinstance(deltax, six.string_types):
         # Not empty and the type is a string. Then a unit can follow.
         parts = deltax.split()
         if len(parts) > 1:
            uf, errmes = unitfactor(parts[1], axisunits)
            if uf is None:
               raise ValueError(errmes)
         else:
            uf =  1.0
         deltax = uf * eval(parts[0])
         
      axisunits = self.gmap.units[1]
      if not deltay is None and isinstance(deltay, six.string_types):
         # Not empty and the type is a string. Then a unit can follow.
         parts = deltay.split()
         if len(parts) > 1:
            uf, errmes = unitfactor(parts[1], axisunits)
            if uf is None:
               raise ValueError(errmes)
         else:
            uf =  1.0
         deltay = uf * eval(parts[0])

      self.offsetvaluesx = None
      self.offsetvaluesy = None
      self.funx = self.funy = None
      self.fmtx = self.fmty = None
      self.radoffsetx = self.radoffsety = False
      if self.offsetx and spatialx:
         # Then the offsets are distances on a sphere
         xmax = self.pxlim[1] + 0.5
         xmin = self.pxlim[0] - 0.5
         ymin = self.pylim[0] - 0.5
         x1 = xmin; x2 = xmax; y1 = y2 = ymin
         Lambda = getlambda(startpixX, xmin, xmax)
         rulerx = self.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=Lambda, step=deltax, units=unitsx)
         #print "WCSGRAT ruler xw, yw", rulerx.xw, rulerx.yw, rulerx.stepsizeW, rulerx.offsets
         #print "WCSGRAT ruler x, y", rulerx.x, rulerx.y
         
         self.xstarts = rulerx.xw
         self.prec[0] = 0
         self.delta[0] = rulerx.stepsizeW
         self.offsetvaluesx = rulerx.offsets
         self.funx = rulerx.fun
         self.fmtx = rulerx.fmt
         self.radoffsetx = True
      elif issequence(startx):
         self.xstarts = startx
         if len(startx) >= 2:
            self.delta[0] = startx[1] - startx[0]  # Assume this delta also for not equidistant values in startx
      else:
         # startx is a scalar or None
         if startx is None and self.offsetx:
            startx = (self.wxlim[1] + self.wxlim[0]) / 2.0
         self.xstarts, self.offsetvaluesx, self.prec[0], self.delta[0] = self.__nicenumbers(self.wxlim[0],self.wxlim[1],
                                                        start=startx, 
                                                        delta=deltax, 
                                                        axtype=self.gmap.types[0], 
                                                        skysys=self.__skysys)

      if self.offsety and spatialy:
         stdout.flush()
         # Then the offsets are distances on a sphere
         ymax = self.pylim[1] + 0.5
         xmin = self.pxlim[0] - 0.5
         ymin = self.pylim[0] - 0.5
         x1 = xmin; x2 = xmin; y1 = ymin; y2 = ymax
         Lambda = getlambda(startpixY, ymin, ymax)
         rulery = self.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=Lambda, step=deltay, units=unitsy)         
         if spatialx and spatialy:
            self.ystarts = rulery.yw
         else:
            self.ystarts = rulery.xw    # NOTE: do not change in yw!
         self.prec[1] = 0
         self.delta[1] = rulery.stepsizeW
         self.offsetvaluesy = rulery.offsets
         self.funy = rulery.fun
         self.fmty = rulery.fmt
         self.radoffsety = True
      elif issequence(starty):
         self.ystarts = starty
         if len(starty) >= 2:
            self.delta[1] = starty[1] - starty[0]  # Assume this delta also for not equidistant values in startx
      else:
         # starty is a scalar
         if starty is None and self.offsety:
            starty = (self.wylim[1] + self.wylim[0]) / 2.0
         self.ystarts, self.offsetvaluesy, self.prec[1], self.delta[1] = self.__nicenumbers(self.wylim[0],self.wylim[1], 
                                                        start=starty, 
                                                        delta=deltay, 
                                                        axtype=self.gmap.types[1], 
                                                        skysys=self.__skysys)
      # We have two sets of lines. For spatial maps, these are the
      # meridians and the parallels. If other axes are involved, it is 
      # better to use the name grid line with constant x or constant
      # y to identify the graticule lines.
      if (gridsamples < 2):
         raise Exception("Number of samples along graticule line must be >= 2 to avoid a step size of zero")

      # Create the plot axes. The defaults are: plot native ticks to axis
      # for the left and bottom axis and omit ticks along right and top axis.
      
      # Get the equinox so that it can be inserted into the axis title.
      # This equinox has been parsed earlier with skyparser()
      epoch = str(epochin)
      annot = ['']*2

      for aa in [0,1]:
         units = self.gmap.units[aa]
         if aa == 0 and unitsx is not None:
            units = unitsx
         if aa == 1 and unitsy is not None:
            units = unitsy
         if (aa == 0 and self.offsetx) or (aa == 1 and self.offsety):
            annot[aa] = "Offset " 
         if self.gmap.types[aa] in [None, 'spectral']:
            annot[aa] += str(self.gmap.ctype[aa]).split('-')[0] + ' (' + str(units) + ')'
         else:        # Must be spatial
            if (aa == 0 and (self.radoffsetx or self.offsetx)) or\
               (aa == 1 and (self.radoffsety or self.offsety)):
               if self.gmap.types[aa] == 'longitude':
                  olab = 'Radial offset lon.'
               else:
                  olab = 'Radial offset lat.'
               if (aa == 0):
                  if unitsx:
                     olab += '('+unitsx+')'
                  annot[aa] = olab
               else:
                  if unitsy:
                     olab += '('+unitsy+')'
                  annot[aa] = olab
            else:
               if self.gmap.types[aa] == 'longitude':
                  if self.__skysys == wcs.equatorial:
                     annot[aa] = 'R.A.' + ' (' + epoch + ')'
                  elif self.__skysys == wcs.ecliptic:
                     annot[aa] = 'Ecliptic longitude'  + ' (' + epoch + ')'
                  elif self.__skysys == wcs.galactic:
                     
                     annot[aa] = 'Galactic longitude'
                  elif self.__skysys == wcs.supergalactic:
                     annot[aa] = 'Supergalactic longitude'
               else:
                  if self.__skysys == wcs.equatorial:
                     annot[aa] = 'Dec.' + ' (' + epoch + ')'
                  elif self.__skysys == wcs.ecliptic:
                     annot[aa] = 'Ecliptic latitude' + ' (' + epoch + ')'
                  elif self.__skysys == wcs.galactic:
                     annot[aa] = 'Galactic latitude'
                  elif self.__skysys == wcs.supergalactic:
                     annot[aa] = 'Supergalactic latitude'

      self.graticule = []   # Initialize the list with graticule lines
      if not skipx:
         axisunits = self.gmap.units[0]
         if not unitsx is None:
            units = unitsx
            uf, errmes = unitfactor(axisunits, units)
            if uf is None:
               raise ValueError(errmes)
            fie = lambda x: x*uf
            fmt = "%g"
            self.funx = fie
            self.fmtx = fmt

         for i, x in enumerate(self.xstarts):
            offsetlabel = None
            fie = fmt = None
            if self.radoffsetx:
               offsetlabel = self.offsetvaluesx[i]
               fie = self.funx
               fmt = self.fmtx
            elif self.offsetx:
               offsetlabel = self.offsetvaluesx[i]
               if self.gmap.types[0] in ['longitude', 'latitude'] and self.labelsintex:
                  fmt = "%g^{\circ}"
               else:
                  fmt = "%g"
            elif not unitsx is None:
               fie = self.funx
               fmt = self.fmtx
            gridl = Gratline(0, x, self.gmap,
                             self.pxlim, self.pylim, 
                             self.wxlim, self.wylim, 
                             gridsamples,self.mixpix, self.__skysys,
                             offsetlabel=offsetlabel,
                             fun=fie, fmt=fmt)
            # Special case. Both axes are spatial but a user required
            # to label the x axis in offsets. The graticule line however is
            # calculated using the value of 'startx' and a non constant value
            # for the other axis (e.g. the declination axis). This is the way
            # we calculate the usual graticule lines, but we required an offset
            # which is not constant along the graticule line. As a trick we
            # replace the pixel coordinates of the x coordinate, by the x
            # value of the ruler. This value is the pixel value of 'startx'.
            # If this graticule line is curved and does not cross e.g. the top
            # x axis, then we will see a partial graticule line. Usually
            # this will be an exception.
            if offsetx and spatialx and spatialy:
               xoff0 = rulerx.x[i]
               for lp in gridl.linepieces:
                  for ll in range(len(lp[0])):
                     lp[0][ll] = xoff0
            self.graticule.append(gridl)
            gridl.kwargs = {'color': '0.75', 'lw': 1}

      if not skipy:
         axisunits = self.gmap.units[1]
         if not unitsy is None:
            units = unitsy
            uf, errmes = unitfactor(axisunits, units)
            if uf is None:
               raise ValueError(errmes)
            fie = lambda x: x*uf
            fmt = "%g"
            self.funy = fie
            self.fmty = fmt

         for i, y in enumerate(self.ystarts):
            offsetlabel = None
            fie = fmt = None
            if self.radoffsety:
               offsetlabel = self.offsetvaluesy[i]
               fie = self.funy
               fmt = self.fmty
            elif self.offsety:
               offsetlabel = self.offsetvaluesy[i]
               if self.gmap.types[1] in ['longitude', 'latitude'] and self.labelsintex:
                  fmt = "%g^{\circ}"
               else:
                  fmt = "%g"
            elif not unitsy is None:
               fie = self.funy
               fmt = self.fmty
            gridl = Gratline(1, y, self.gmap,
                             self.pxlim, self.pylim, 
                             self.wxlim, self.wylim, 
                             gridsamples,self.mixpix, self.__skysys,
                             offsetlabel=offsetlabel,
                             fun=fie, fmt=fmt)
           
            if offsety and spatialx and spatialy:
               yoff0 = rulery.y[i]    
               for lp in gridl.linepieces:
                  for ll in range(len(lp[1])):
                     lp[1][ll] = yoff0
            gridl.kwargs = {'color': '0.75', 'lw': 1}
            self.graticule.append(gridl)

      # Set properties for the rectangle axes.
      xnumticks = 0
      ynumticks = 0
      for gridline in self.graticule:
          wcsaxis = gridline.wcsaxis
          for t in gridline.ticks:
             anr = t.axisnr
             if wcsaxis == 1 and anr == 0:
                ynumticks += 1
             if wcsaxis == 0 and anr == 1:
                xnumticks += 1
      x1mode = 0
      x2mode = 3
      y1mode = 0
      y2mode = 3
      if xnumticks < 2:
         x1mode = 2
         x2mode = 2 
      if ynumticks < 2:
         y1mode = 2
         y2mode = 2

      axes = []
      # keyword arguments for the tick labels
      kwargs1 = {'fontsize':11}
      kwargs2 = {'fontsize':11}
      kwargs3 = {'fontsize':11, 'rotation':'270', 'visible':False}
      kwargs4 = {'fontsize':11, 'visible':False}
      axes.append( PLOTaxis(left,   y1mode, annot[1], **kwargs1) )
      axes.append( PLOTaxis(bottom, x1mode, annot[0], **kwargs2) )
      axes.append( PLOTaxis(right,  y2mode, annot[1], **kwargs3) )
      axes.append( PLOTaxis(top,    x2mode, annot[0], **kwargs4) )
      self.axes = axes

      dummy = self.get_aspectratio() # Set default values for aspect ratio
      self.objlist = []              # Initialize list with Graticules & derived objects


   def plot(self, framebase, frame=None, frame2=None):
      """
      -----------------------------------------------------------
      Purpose:      Plot the graticule lines and labels
                    Labels are either along the plot axes or 
                    inside the plot.
      Parameters:
        framebase   An Axes object to which the graticules are
                    scaled. Two new Axes objects are created to hold two
                    independent graticules.
        frame       If not None, this is an Axes object, previously used
                    to hold a graticule. So one can re-use this frame,
                    which is in fact a requirement if one wants to
                    zoom/dezoom images with graticule and be able
                    to restore different zoom stages with the toolbar buttons.
        frame2      See frame.
        
      Returns:      --
      Notes:        In Sep 2012 we changed this method to be able to
                    interactively zoom an image with a graticule.
                    There is no method to (re)set the pixel limits of
                    a graticule object, so for zooming one should
                    destroy an old graticule and make a new one.
                    This is not an ideal situation, but it works.
                    However it is often dangerous to add Axes objects while
                    zooming, so we added the option to reuse existing frames.
                    We copy an example from module maputils:


                    frame1 = cube.grat.frame
                    frame2 = cube.grat.frame2
                    cube.grat = currentimage.Graticule(pxlim=newpxlim, pylim=newpylim)
                    cube.grat.plot(currentimage.frame, frame1, frame2)

                    TODO: We should check other plot methods. If they are
                    also used in zoom/dezoom/pan actions, then they should also
                    have the option to reuse existing frames.
      -----------------------------------------------------------
      """
      # We need to sort and format the ticks for the 4 plot axes
      graticule = self
      tex = graticule.labelsintex
      pos, lab, kwargs, Mkwargs = graticule.sortticks(tex=tex)
      aspect = framebase.get_aspect()
      adjust = framebase.get_adjustable()

      # Each graticule holds two frames for two times two axes
      if not frame:
         framelabel= 'FR'+"".join([choice(letters) for i in range(8)])
         try:
            r,c,n = framebase.get_geometry()
            frame = framebase.figure.add_subplot(r, c, n,
                                       aspect=aspect,
                                       adjustable=adjust,
                                       autoscale_on=False,
                                       frameon=False,
                                       label=framelabel)

            # If a colorbar is requested then the original frame (framebase) is
            # altered. The add_subplot method does not see this altered position.
            frame.set_position(framebase.get_position())
         except:
            frame = framebase.figure.add_axes(framebase.get_position(),
                                       aspect=aspect,
                                       adjustable=adjust,
                                       autoscale_on=False,
                                       frameon=False,
                                       label=framelabel)

      graticule.frame = frame  # Store frame so that frame can be identified with events
      # In Matplotlib, it is possible that the upper pixel limit is
      # lower than the lower value. To get graticules for entire pixels in your image
      # we add 0.5, but as an extension. So if hi < lo, we should change the sign of
      # the delta we want to add. However... we work with FITS files or FITS related
      # files where always hi > lo.
      # This is important for the following special case.
      # Assume a user zoomed an area and the area does not include enire border
      # pixels. Then we have a different situation than that of standard image limits
      # which are always in integer pixels. Then one has to avoid the 0.5 correction
      # by adding/subtracting this value before calling the constructor.
      # Assume now that we have an area smaller than 1 pixel. Then this correction
      # is the cause that hi < lo and we add the wrong delta's which increases
      # the area instead of keeping it constant. So we omit the special case
      # that hi < lo.
      deltax = deltay = 0.5
      """
      if graticule.pxlim[0] <= graticule.pxlim[1]:
         deltax = 0.5
      else:
         deltax = -0.5
      if graticule.pylim[0] <= graticule.pylim[1]:
         deltay = 0.5
      else:
         deltay = -0.5
      """
      xlo = graticule.pxlim[0]-deltax; ylo = graticule.pylim[0]-deltay
      xhi = graticule.pxlim[1]+deltax; yhi = graticule.pylim[1]+deltay
      frame.set_yticks(pos[left])
      frame.set_yticklabels(lab[left])
      #print "WCSGRAT ticks=", str(frame.yaxis.get_major_ticks())
      for tick, kw, mkw in zip(frame.yaxis.get_major_ticks(), kwargs[left], Mkwargs[left]):
         tick.label1.set(**kw)
         if len(mkw) != 0:
            tick.tick1line.set(**mkw)
         tick.tick2on = False
         tick.tick2line.set_visible(False)

      frame.set_xticks(pos[bottom])
      frame.set_xticklabels(lab[bottom])

      for tick, kw, mkw in zip(frame.xaxis.get_major_ticks(), kwargs[bottom], Mkwargs[bottom]):
         tick.label1.set(**kw)
         if len(mkw) != 0:
            tick.tick1line.set(**mkw)
         tick.tick2on = False
         tick.tick2line.set_visible(False) 

      if not frame2:
         framelabel= 'SFR'+"".join([choice(letters) for i in range(8)])
         try:
            r,c,n = framebase.get_geometry()
            frame2 = framebase.figure.add_subplot(r, c, n,
                                       aspect=aspect,
                                       adjustable=adjust,
                                       autoscale_on=False,
                                       frameon=False,
                                       label=framelabel)
            frame2.set_position(framebase.get_position())
         except:
            frame2 = framebase.figure.add_axes(frame.get_position(),
                                       aspect=aspect,
                                       adjustable=adjust,
                                       autoscale_on=False,
                                       frameon=False,
                                       label=framelabel)

      # frame2 = framebase.figure.add_axes(frame.get_position(), frameon=False, label=framelabel)
      graticule.frame2 = frame2  # Add new attribute
      # axis sharing is not an option because then also the ticks are
      # shared and we want independent ticks along all 4 axes. For most 
      # projections the axes are not related.
      frame2.yaxis.set_label_position('right')
      frame2.xaxis.set_label_position("top")
      frame2.yaxis.tick_right()
      frame2.xaxis.tick_top()
      frame2.set_xlim(xlo,xhi)
      frame2.set_ylim(ylo,yhi)
      frame2.set_aspect(aspect=aspect, adjustable=adjust)

      frame2.set_xticks(pos[top])
      frame2.set_xticklabels(lab[top])
      for tick, kw, mkw in zip(frame2.xaxis.get_major_ticks(),kwargs[top], Mkwargs[top]):
         tick.label2.set(**kw)
         if len(mkw) != 0:
            tick.tick2line.set(**mkw)
         tick.tick1line.set_visible(False)
      
      frame2.set_yticks(pos[right])
      frame2.set_yticklabels(lab[right])
      for tick, kw, mkw in zip(frame2.yaxis.get_major_ticks(),kwargs[right], Mkwargs[right]):
         tick.label2.set(**kw)
         if len(mkw) != 0:
            tick.tick2line.set(**mkw)
         tick.tick1line.set_visible(False)
      


      # Adjust label position if one is given
      for anr in [left, right]:
         xpos = graticule.axes[anr].xpos
         ypos = graticule.axes[anr].ypos
         if xpos is not None or ypos is not None:
            if xpos is None and ypos is not None:
               graticule.axes[anr].kwargs.update({'y':ypos})
            else:
               if ypos is None: ypos = 0.5
               # xpos is not None then use method
               if anr == left:
                  fr = frame
               else:
                  fr = frame2
               fr.yaxis.set_label_coords(xpos, ypos)

      for anr in [bottom, top]:
         xpos = graticule.axes[anr].xpos
         ypos = graticule.axes[anr].ypos
         if xpos is not None or ypos is not None:
            if ypos is None and xpos is not None:
               graticule.axes[anr].kwargs.update({'x':xpos})
            else:
               if xpos is None: xpos = 0.5
               # ypos is not None then use method set_label_coords
               # because setting y as attribute has no effect.
               if anr == bottom:
                  fr = frame
               else:
                  fr = frame2
               fr.xaxis.set_label_coords(xpos, ypos)

      frame.set_ylabel( graticule.axes[left].label,   **graticule.axes[left].kwargs)
      frame.set_xlabel( graticule.axes[bottom].label, **graticule.axes[bottom].kwargs)
      frame2.set_ylabel(graticule.axes[right].label,  **graticule.axes[right].kwargs)
      frame2.set_xlabel(graticule.axes[top].label,    **graticule.axes[top].kwargs)

      # Plot the line pieces
      for gridline in graticule.graticule:
         for line in gridline.linepieces:
            #ppx = numpy.ma.masked_where(numpy.isfinite(line[0]), line[0])
            #ppy = numpy.ma.masked_where(numpy.isfinite(line[1]), line[1])
            #ppx = numpy.ma.masked_where(numpy.isinf(line[0]), line[0])
            #ppy = numpy.ma.masked_where(numpy.isinf(line[1]), line[1])
            frame.plot(line[0], line[1], **gridline.kwargs)
            #frame.plot(ppx, ppy, **gridline.kwargs)
      # set the limits of the plot axes
      # this setting can be overwritten in the calling environment
      frame.set_xlim((xlo,xhi))
      frame.set_ylim((ylo,yhi))
      frame.set_aspect(aspect=aspect, adjustable=adjust)

      # Plot the objects derived from this graticule
      if len(graticule.objlist) > 0:
         for obj in graticule.objlist:
            try:
               pt = obj.ptype
            except:
               raise Exception("Unknown object. Cannot plot this!")
            if pt in ["Insidelabels"]:
               try:
                  visible = obj.visible
               except:
                  visible = True
               # Previously: obj.plot(framebase)
               # We want the labels on top of the graticule lines.
               # Then they must be plotted in the frame of the inside labels
               # and not in the base frame, because that frame is not on top.
               obj.plot(frame2)
      framebase.figure.sca(framebase)    # back to frame from calling environment


   def scanborder(self, xstart, ystart, deltax=None, deltay=None, nxy=1000, tol=None):
      #-----------------------------------------------------------------
      """
      For the slanted azimuthal projections, it is
      not trivial to draw a border because these
      borders are not graticule lines with a constant
      longitude or constant latitude. Nor it is
      easy or even possible to find mathematical
      expressions for this type of projection.
      Also, the mathematical expressions return
      world coordinates which can suffer from loss
      of precision.
      This method tracks the border from a starting
      point by scanning in x- and y direction and
      tries to find the position of a limb with a
      standard bisection technique. This method has been applied
      to a number of all-sky plots with slanted projections.

      :param xstart: X-coordinate in pixels of position where to
                     start the scan to find a border.
                     The parameter has no default.
      :type xstart:  Floating point 

      :param ystart: Y-coordinate in pixels of position where to
                     start the scan to find border.
                     The parameter has no default.
      :type ystart:  Floating point

      :param deltax: Set range in pixels to look for a border in
                     scan direction. The default value is 10 percent
                     of the total pixel range in x- or y-direction.
      :type deltax:  Floating point

      :param deltay: See *deltayx*.
      :type deltay:  Floating point

      :param nxy:    Number of scan lines in x and y direction.
                     Default is 1000.
      :type nxy:     Integer

      :param tol:    See note below.
      :type tol:     Floating point

      :returns:   Identifier to set attributes of this
                  graticule line with method :meth:`setp_linespecial()`.

      :note:      This method uses an algorithm to find
                  positions along the border of a projection.
                  It scans along both x- and y-axis for
                  a NaN (Not a Number number) transition as a result
                  of an invalid coordinate transformation,
                  and repeats this for a number of scan lines
                  along the x-axis and y-axis.

                  ::

                     A position on a border off an all-sky plot is the position at
                     which a transition occurs from a valid coordinate to a NaN.

                  Its accuracy depends on the the tolerance
                  given in argument *tol*.
                  The start coordinates to find the next border
                  position on the next scan line is the
                  position of the previous border point.
                  If you have missing line pieces, then add more
                  borders by calling this method with different
                  starting points.
      """
      #-----------------------------------------------------------------
      xp = []; yp = []
      if deltax is None:
         deltax = (self.pxlim[1] - self.pxlim[0])/ 10.0
      if deltay is None:
         deltay = (self.pylim[1] - self.pylim[0])/ 10.0
      d = (float(self.pxlim[1] - self.pxlim[0]), float(self.pylim[1] - self.pylim[0])) 
      delta = (deltax, deltay)
      limlo = (self.pxlim[0], self.pylim[0])
      limhi = (self.pxlim[1], self.pylim[1])
      start = (xstart, ystart) 
      for i in [0,1]:
         if tol is None:
            tol = delta[i] / 1000.0
         nx1 = (start[i] - limlo[i])/d[i]
         nx2 = 1.0 - nx1
         nx1 = int(nxy*nx1)
         nx2 = int(nxy*nx2)
         l1 = numpy.linspace(start[i], limlo[i], nx1)
         l2 = numpy.linspace(start[i], limhi[i], nx2)
         X = numpy.concatenate((l1,l2))
         Y0 = (ystart, xstart)
         for xb in X:
            if xb == start[i]:
               y0 = Y0[i]
            yb = self.__bisect(i, xb, y0-delta[i]/2.0, y0+delta[i]/2.0, self.gmap, tol)
            if yb is not None:
               if i == 0:
                  xp.append(xb)
                  yp.append(yb)
               else:
                  xp.append(yb)
                  yp.append(xb)
               y0 = yb

      return self.addgratline(xp, yp, pixels=True)


   def addgratline(self, x, y, pixels=False):
      #-----------------------------------------------------------------
      """
      For any path given by a set of world coordinates
      of which none is a constant value (e.g. borders
      in slanted projections where the positions are calculated by an
      external routine),
      one can create a line that is processed as a graticule
      line, i.e. intersections and jumps are addressed.
      Instead of world coordinates, this method
      can also process pixel positions. The type of input is set by the
      *pixels* parameter.
      
      
      :param x:      A sequence of world coordinates or pixels
                     that correspond to the horizontal axis in a graticule plot..
      :type x:       Floating point numbers
      
      :param y:      The same for the second axis
      :type x:       Floating point numbers
      
      :param pixels: False or True
                     If False the coordinates in x and y are world-
                     coordinates. Else they are pixel coordinates.
      :type pixels:  Boolean
      
      :Returns:      A Identification number *id* which can be used
                     to set properties for this special path with
                     method :meth:`setp_linespecial`.
                     Return *None* if no line piece could be found
                     inside the pixel limits of the graticule.
      
      :note:         This method can be used to plot a border
                     around an all-sky plot e.g. for slanted
                     projections. See code at :meth:`scanborder`.
      """
      #-----------------------------------------------------------------
      if len(x) > 0:
         samples = len(x)
         # Create an unique id for this line
         id = len(self.graticule) + 2; # Avoid problems if there are no gridlines yet, so add at least 2 to the id value
         gridl = Gratline(id, '', self.gmap,
                          self.pxlim, self.pylim, 
                          self.wxlim, self.wylim, 
                          samples, 
                          self.mixpix, self.__skysys, 
                          addx=x, addy=y, addpixels=pixels)
         self.graticule.append(gridl)
         gridl.kwargs = {'color': 'r', 'lw': 1}
      else:
         id = None;
      return id


   def __str__(self):
      """
      -----------------------------------------------------------
      Purpose:    Show textual contents of graticule:
                  lines, ticks and line pieces.
      Parameters: -
      Returns:    -
      Notes:      The information is unsorted w.r.t. the 
                  plot axis number.
      Example:    g = Graticule(header)
                  print g
      -----------------------------------------------------------
      """
      s = ''
      for gridline in self.graticule:
         s += "\nWCS graticule line number %s\n" % gridline.wcsaxis
         s += "Axis type: %s.  Sky system %s:\n" % (gridline.axtype, gridline.skysys)
         for t in gridline.ticks:
            s += "tick x,y:  %f %f\n" % (t.x, t.y)
            s += "tick label: %f\n" % t.labval
            s += "tick on axis: %d\n" % t.axisnr
         for line in gridline.linepieces:
            s += "line piece has %d elements\n" % len(line[0])
            # line is (line[0], line[1])
      return s




   def get_aspectratio(self, xcm=None, ycm=None):
      #-----------------------------------------------------------------
      """
      Calculate and set, the aspect ratio for the current
      pixels. Also set default values for figure
      size and axes lengths (i.e. size of canvas depends
      on the size of plot window with this aspect ratio).
      
      :param xcm: Given a value for xcm or ycm (or omit both),
                  suggest a suitable figure size in and a viewport in
                  normalized device coordinates of a plot which has
                  an axes rectangle that corrects the figure for an
                  aspect ratio (i.e. CDELTy/CDELTx) unequal to 1 while
                  the length of the x-axis is xcm OR the length of the
                  y-axis is ycm. See note for non-spatial maps.
                  
      :type xcm:  Floating point number
      
      :param ycm: See description at *xcm*.
      :type ycm:  Floating point number
      
      :Returns:   The aspect ratio defined as: ``AR = CDELTy/CDELTx``.
      
      :Note:      (i.e. AR > 10 or AR < 0.1), an aspect ratio of 1
                  is returned. This method sets the attributes:
                  'axesrect', 'figsize', 'aspectratio'.
                  The attribute 'figsize' is in inches which is compatible
                  to the methods of Matplotlib.
      """
      #-----------------------------------------------------------------
      cdeltx = self.gmap.cdelt[0]
      cdelty = self.gmap.cdelt[1]
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      if xcm is None and ycm is None:
         xcm = 20.0
      aspectratio = abs(cdelty/cdeltx)
      if aspectratio > 10.0 or aspectratio < 0.1:
         aspectratio = nx/ny
         if xcm is None:
            xcm = ycm
         else:
            ycm = xcm
      if ycm is None:
         ycm = xcm * (ny/nx) * aspectratio
      if xcm is None:
         xcm = ycm * (nx/ny) / aspectratio
      fh = 0.7; fw = 0.7
      self.axesrect = (0.15, 0.15, fw, fh)
      self.figsize = (xcm/2.54/fw, ycm/2.54/fh)
      self.aspectratio = aspectratio
      return aspectratio



   def setp_tick(self, wcsaxis=None, plotaxis=None, position=None,
                 tol=1e-12, fmt=None, fun=None, tex=None, texsexa=None,
                 markerdict={}, **kwargs):
      #-----------------------------------------------------------------
      """
      Set (plot) attributes for a wcs tick label.
      A tick is identified by the type of grid line
      it belongs to, and/or the plot axis for which
      it defines an intersection and/or a position which
      corresponds to the constant value of the graticule
      line.
      All these parameters are valid with none, one or
      a sequence of values.
      
      .. warning:: If no value for *wcsaxis*, *plotaxis* or *position* is entered
            then this method applies the parameter setting on all the wcs axes.
      
      :param wcsaxis:
            Values are 0 or 1, corresponding to the
            first and second world coordinate types.
            Note that *wcsaxis=0* corresponds to the
            first element in the axis permutation array given in
            parameter *axnum*.
      :type wcsaxis: *None*, 0, 1 or tuple with both
      
      :param plotaxis:
            Accepted values are 'None', 0, 1, 2, 3 or a
            sequence of these numbers, to represent the left, bottom, right
            and top axis of the enclosing rectangle that
            represents the limits in pixel coordinates.
      :type plotaxis: One or more integers between 0 and 3.
      
      :param position:
            Accepted are None, or one or more values representing
            the constant value of the graticule line in
            world coordinates. These positions are used to identify
            individual graticule lines so that each line can have its
            own properties.
            The input can also be a string that represents a sexagesimal number.
      :type position: *None* or one or a sequence of floating point numbers
      
      :param tol:
            If a value > 0 is given, the gridline with the
            constant value closest to a given position within
            distance 'tol' gets updated attributes.
      :type tol: Floating point number
      
      :param fmt:
            A string that formats the tick value
            e.g. ``fmt="%10.5f"`` in the Python way, or a string
            that contains no percentage character (%) but a format
            to set the output of sexagesimal numbers e.g.
            fmt='HMs'. The characters in the format either force
            (uppercase) a field to be printed, or it suppresses
            (lowercase) a field to be printed.
            See also the examples at :func:`wcsgrat.makelabel`.
            To create labels with an exponential, use a second format in
            the same format string. The syntax is %nne where nn is an
            integer. This integer, which can be negative, sets the number
            in the exponential. The number before the exponential is
            formatted in the usual way e.g. fmt='%.3f%-3e'.
      :type fmt: String
      
      :param fun:
            An external function which will be used to
            convert the tick value e.g. to convert
            velocities from m/s to km/s. See also
            example 2_ below.
      :type fun: Python function or Lambda expression

      :param tex:
            Interpret the format in *fmt* as a TeX label.
            The default is set to *None* to indicate it has
            not been set (to True or False) so that it is possible
            to distinguish between global and local settings of this
            property.
      :type tex:
            Boolean

      :param texsexa:
            If False and parameter *tex* is True, then format the
            tick label without superscripts for sexagesimal labels.
            This option can be used if superscripts result in 'jumpy' labels.
            The reason is that in Matplotlib the TeX labels at the bottom
            of a plot are aligned at a baseline at the top of the characters.
      :type tex: Boolean

      :param markerdict:
            Properties for the tick marker. Amongst others:
      
               * markersize:
                 Size of tick line. Use a negative number (e.g. -4) to
                 get tick lines that point outside the plot instead
                 of the default which is inside.
               * markeredgewidth:
                 The width of the marker
               * color: Color of the marker (not the label)
      
      :type markerdict:
            Python dictionary
            
      :param `**kwargs`:
            Keyword arguments for plot properties like *color*,
            *visible*, *rotation* etc. The plot attributes are standard
            Matplotlib attributes which can be found in the
            Matplotlib documentation.
      :type `**kwargs`: Matplotlib keyword arguments
      
      :note:
            Some projections generate labels that are very close
            to each other. If you want to skip labels then you can
            use keyword/value *visible=False*. There is not a documented
            keyword *visible* in this method because *visible* is a
            valid keyword argument in Matplotlib.
      
      
      :Examples:
         1. Set tick properties with :meth:`setp_tick`. The last line makes the
         label at a declination of -10 degrees (we assume a spatial map) invisible::
      
               grat.setp_tick(wcsaxis=0, color='g')
               grat.setp_tick(wcsaxis=1, color='m')
               grat.setp_tick(wcsaxis=1, plotaxis=wcsgrat.bottom,
                  color='c', rotation=-30, ha='left')
               grat.setp_tick(plotaxis=wcsgrat.right, backgroundcolor='yellow')
               grat.setp_tick(plotaxis=wcsgrat.left, position=-10, visible=False)
      
      
         .. _2:
      
         2. Example of an external function to change the values of the tick
         labels for the horizontal axis only::
      
               def fx(x):
                  return x/1000.0
      
               setp_tick(wcsaxis=0, fun=fx)
      
         Or use the lambda operator as in: ``fun=lambda x: x/1000``
      """
      #-----------------------------------------------------------------
      if wcsaxis is None and plotaxis is None and position is None:
         ## Nothing to do
         #return
         wcsaxis = [0,1]
      if wcsaxis is not None:
         if not issequence(wcsaxis):
            wcsa = [wcsaxis]
         else:
            wcsa = wcsaxis
      if plotaxis is not None:
         plta = parseplotaxes(plotaxis)
      if position is not None:
         if not issequence(position):
            posn = [position]
         else:
            posn = position

      for gridline in self.graticule:
        skip = False
        if (wcsaxis is not None):
           skip = not (gridline.wcsaxis in wcsa)
        if not skip:
           if position is None:
              for t in gridline.ticks:
                 skip = False
                 if plotaxis is not None:
                    skip = not (t.axisnr in plta)
                 if not skip:
                    t.kwargs.update(kwargs)
                    t.markkwargs.update(markerdict)
                    # The attributes fmt and fun to format tick labels could have
                    # been initialized when the object was created. If values are
                    # given then they override the defaults.
                    if fmt is not None: t.fmt = fmt
                    if fun is not None: t.fun = fun
                    if tex is not None: t.tex = tex
                    if not texsexa is None: t.texsexa = texsexa
                    # There are defaults for the precision in seconds for each axis
                    # If the user sets a format with precision in seconds (e.g. HMS.SS)
                    # then we need to tell this to different methods that deal with it.
                    if not fmt is None and fmt.find('%') == -1:   # Not a common format, must be a HMS/DMS format
                       s2 = fmt.split('.')
                       if len(s2) > 1:
                          self.prec[gridline.wcsaxis] = len(s2[1])
           else:      # One or more positions are given. Find the right index.
              for pos in posn:
                 if isinstance(pos, six.string_types):
                    C = pos.upper()
                    if 'H' in C or 'D' in C or 'M' in C or 'S' in C:
                       pos, err = parsehmsdms(pos)
                       if err != '':
                          raise Exception(err)
                       else:
                          pos = pos[0]
                    else:
                       raise ValueError("[%s] is entered as a string but does not represent valid HMS or DMS"%pos)
                 d0 = None
                 for i, t in enumerate(gridline.ticks):
                    skip = False
                    if plotaxis is not None:
                       skip = not (t.axisnr in plta)
                    if not skip:
                       d = abs(t.labval - pos)
                       if d <= tol:
                          if d0 is None:
                             d0 = d
                             indx = i
                          elif d < d0:
                             d0 = d
                             indx = i
                 if d0 is not None:
                    gridline.ticks[indx].kwargs.update(kwargs)
                    gridline.ticks[indx].markkwargs.update(markerdict)
                    if len(markerdict) == 0 or fmt is not None:
                        gridline.ticks[indx].fmt = fmt
                    if len(markerdict) == 0 or fun is not None:
                        gridline.ticks[indx].fun = fun
                    if tex is not None:
                        gridline.ticks[indx].tex = tex
                    if texsexa is not None:
                        gridline.ticks[indx].texsexa = texsexa


   def setp_tickmark(self, wcsaxis=None, plotaxis=None, position=None,
                     tol=1e-12, **mkwargs):
      #-----------------------------------------------------------------
      """
      Utility method for :meth:`setp_tick`. It handles the
      properties of the tick marks, which are :class:`Line2D` objects
      in Matplotlib. The most useful properties are *color*, *markeredgewidth*
      and *markersize*.
      """
      #-----------------------------------------------------------------
      self.setp_tick(wcsaxis=wcsaxis, plotaxis=plotaxis, position=position,
                     tol=tol, markerdict=mkwargs)


   def setp_ticklabel(self, wcsaxis=None, plotaxis=None, position=None,
                      tol=1e-12, fmt=None, fun=None, tex=None, texsexa=None,
                      **kwargs):
      #-----------------------------------------------------------------
      """
      Utility method for :meth:`setp_tick`. It handles the
      properties of the tick labels, which are :class:`Text` objects
      in Matplotlib. The most useful properties are *color*, *fontsize*
      and *fontstyle*.
      
      :param wcsaxis:
            Values are 0 or 1, corresponding to the
            first and second world coordinate types.
            Note that *wcsaxis=0* corresponds to the
            first element in the axis permutation array given in
            parameter *axnum*.
      :type wcsaxis: *None*, 0, 1 or tuple with both
      
      :param plotaxis:
            Accepted values are 'None', 0, 1, 2, 3 or a
            combination, to represent the left, bottom, right
            and top axis of the enclosing rectangle that
            represents the limits in pixel coordinates.
      :type plotaxis: One or more integers between 0 and 3.
      
      :param position:
            Accepted are None, or one or more values representing
            the constant value of the graticule line in
            world coordinates. These positions are used to identify
            individual graticule lines so that each line can have its
            own properties.
            The input can also be a string that represents a sexagesimal number.
      :type position: *None* or one or a sequence of floating point numbers
      
      :param tol:
            If a value > 0 is given, the gridline with the
            constant value closest to a given position within
            distance 'tol' gets updated attributes.
      :type tol: Floating point number
      
      :param fmt:
            A string that formats the tick value
            e.g. ``fmt="%10.5f"`` in the Python way, or a string
            that contains no percentage character (%) but a format
            to set the output of sexagesimal numbers e.g.
            fmt='HMs'. The characters in the format either force
            (uppercase) a field to be printed, or it suppresses
            (lowercase) a field to be printed.
            See also the examples at :func:`wcsgrat.makelabel`.
      :type fmt: String
      
      :param fun:
            An external function which will be used to
            convert the tick value e.g. to convert
            velocities from m/s to km/s. See also
            example 2_ below.
      :type fun: Python function or Lambda expression

      :param tex:
            If True then format the tick label in LaTeX. This is the
            default. If False then standard text will applies.
            Some text properties cannot be changed if LaTeX is
            in use.
      :type tex: Boolean

      :param texsexa:
            If False and parameter *tex* is True, then format the
            tick label without superscripts for sexagesimal labels.
            This option can be used if superscripts result in 'jumpy' labels.
            The reason is that in Matplotlib the TeX labels at the bottom
            of a plot are aligned at a baseline at the top of the characters.
      :type tex: Boolean


      :param `**kwargs`:
            Keyword arguments for plot properties like *color*,
            *visible*, *rotation* etc. The plot attributes are standard
            Matplotlib attributes which can be found in the
            Matplotlib documentation.
      :type `**kwargs`: Matplotlib keyword arguments


      :note:
            Some projections generate labels that are very close
            to each other. If you want to skip labels then you can
            use keyword/value *visible=False*. There is not a documented
            keyword *visible* in this method because *visible* is a
            valid keyword argument in Matplotlib.

      """
      #-----------------------------------------------------------------
      self.setp_tick(wcsaxis=wcsaxis, plotaxis=plotaxis, position=position,
                     tol=tol, fmt=fmt, fun=fun, tex=tex, **kwargs)




   def setp_gratline(self, wcsaxis=None, position=None, tol=1e-12, **kwargs):
      #-----------------------------------------------------------------
      """
      Set (plot) attributes for one or more graticule
      lines.
      These graticule lines are identified by the wcs axis
      number (*wcsaxis=0* or *wcsaxis=1*) and by their constant
      world coordinate in *position*.
      
      :param wcsaxis:    If omitted, then for both types of graticule lines
                         the attributes are set.
                         If one value is given then only for that axis
                         the attributes will be set.
      :type wcsaxis:     *None* , integer or tuple with integers from set 0, 1.
      
      :param position:   None, one value or a sequence of
                         values representing the constant value of a graticule
                         line in world coordinates. For the graticule line(s)
                         that match a position in this sequence, the attributes
                         are updated.
      :type position:    *None*, one or a sequence of floating point numbers
      
      :param tol:        If a value > 0 is given, the graticule line with the
                         constant value closest to a given position within
                         distance *tol* gets updated attributes.
      :type tol:         Floating point number
      
      :param `**kwargs`: Keyword arguments for plot properties like *color*,
                         *rotation* or *visible*, *linestyle* etc.
      :type  `**kwargs`: Matplotlib keyword argument(s)
      
      :Returns:          --
      
      :Notes:            For each value in *position* find the index of
                         the graticule line that belongs to *wcsaxis* so that
                         the distance between that value and the constant
                         value of the graticule line is the smallest of all
                         the graticule lines. If *position=None* then
                         apply change of properties to ALL graticule lines.
                         The (plot) properties are stored in `**kwargs`
                         Note that graticule lines are initialized with
                         default properties. These kwargs only update
                         the existing kwargs i.e. appending new keywords
                         and update existing keywords.
      """
      #-----------------------------------------------------------------
      # Upgrade 'wcsaxis' to a sequence
      if wcsaxis is None:
         wcsaxislist = [0,1]
      elif not issequence(wcsaxis):
         wcsaxislist = [wcsaxis]
      else:
         wcsaxislist = wcsaxis
      if position is None:
         for gridline in self.graticule:
            if gridline.wcsaxis in wcsaxislist:
               gridline.kwargs.update(kwargs)
      else: # Upgrade 'position' to a sequence
         if not issequence(position):
            S = [position]
         else:
            S = position
         for constval in S:
            if isinstance(constval, six.string_types):
               C = constval.upper()
               if 'H' in C or 'D' in C or 'M' in C or 'S' in C:
                  constval, err = parsehmsdms(constval)
                  if err != '':
                     raise Exception(err)
                  else:
                     constval = constval[0]
               else:
                  raise ValueError("[%s] is entered as a string but does not represent valid HMS or DMS"%constval)
            d0 = None
            for i, gridline in enumerate(self.graticule):
               if gridline.wcsaxis in wcsaxislist:
                  d = abs(gridline.constval - constval)
                  if d <= tol:
                     if d0 is None:
                        d0 = d
                        indx = i
                     else:
                        if d < d0:
                           d0 = d
                           indx = i
            if d0 is not None:     # i.e. we found a closest position
               self.graticule[indx].kwargs.update(kwargs)


   def setp_lineswcs0(self, position=None, tol=1e-12, **kwargs):
      #-----------------------------------------------------------------
      """
      Helper method for :meth:`setp_gratline`.
      It pre-selects the grid line that
      corresponds to the first world coordinate.
      
      :Parameters:  See description at :meth:`setp_gratline`
      
      :Examples:  Make lines of constant latitude magenta and
                  lines of constant longitude green. The line that
                  corresponds to a latitude of 30 degrees and
                  the line that corresponds to a longitude of 0
                  degrees are plotted in red with a line width of 2:: 
      
                     grat.setp_lineswcs1(color='m')
                     grat.setp_lineswcs0(color='g')
                     grat.setp_lineswcs1(30, color='r', lw=2)
                     grat.setp_lineswcs0(0, color='r', lw=2)

      """
      #-----------------------------------------------------------------
      axis = 0
      self.setp_gratline(axis, position, tol, **kwargs)


   def setp_lineswcs1(self, position=None, tol=1e-12, **kwargs):
      #-----------------------------------------------------------------
      """
      Equivalent to  method :meth:`setp_gratline`.
      It pre-selects the grid line that
      corresponds to the second world coordinate.
      
      :Parameters:  See description at :meth:`setp_gratline`
      
      :Examples:     See example at :meth:`setp_lineswcs0`.
      """
      #-----------------------------------------------------------------
      axis = 1
      self.setp_gratline(axis, position, tol, **kwargs)



   def setp_linespecial(self, id, **kwargs):
      #-----------------------------------------------------------------
      """
      Set (plot) attributes for a special type
      of graticule line made with method :meth:`addgratline`
      or method :meth:`scanborder`.
      This graticule line has no constant x- or y- value.
      It is identified by an id returned by method
      :meth:`addgratline`.
      
      :param id:         id from :meth:`addgratline`
      :type id:          Integer
      :param `**kwargs`: keywords for (plot) attributes
      :type `**kwargs`:  Matplotlib keyword argument(s)


      :Examples:         Create a special graticule line
                         which follows the positions in two
                         given sequences *x* and *y*. and set
                         the line width for this line to 2::
      
                              id = grat.addgratline(x, y)
                              grat.setp_linespecial(id, lw=2)
      """
      #-----------------------------------------------------------------
      # Set properties for an added gridline
      for gridline in self.graticule:
         if gridline.wcsaxis == id and id > 1:
            gridline.kwargs.update(kwargs)


   def switchdefaults(self):
      self.setp_axislabel(plotaxis=("right","top"), visible=True)
      self.setp_axislabel(plotaxis=("left","bottom"), visible=False)
      self.set_tickmode(plotaxis=("right","top"), mode="NATIVE")
      self.set_tickmode(plotaxis=("left","bottom"), mode="NO")


   def setp_plotaxis(self, plotaxis, mode=None, label=None, xpos=None,
                     ypos=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Set (plot) attributes for titles along a plot axis and set the ticks mode.
      The ticks mode sets the relation between the ticks and the plot axis.
      For example a rotated map will show a rotated graticule, so ticks for both
      axes can appear along a plot axis. With parameter *mode* one can influence this
      behaviour.
      
      .. Note::
         This method addresses the four axes of a plot separately. Therefore
         its functionality cannot be incorporated in :meth:`setp_tick`
      
      :param plotaxis:   The axis number of one of the axes of the
                         plot rectangle:
      
                           * wcsgrat.left
                           * wcsgrat.bottom
                           * wcsgrat.right
                           * wcsgrat.top
      
                         or (part of) a string which can be (case insensitive)
                         matched by one from 'left', 'bottom', 'right', 'top'.
                        
      :type plotaxis:    Integer or String
      
      :param mode:       What should this axis do with the tick
                         marks and labels?
      
                           * 0 = ticks native to axis type only
                           * 1 = only the tick that is not native to axis type
                           * 2 = both types of ticks (map could be rotated)
                           * 3 = no ticks
      
                         Or use a text that can (case insensitive) match one of:
      
                           * "NATIVE_TICKS"
                           * "SWITCHED_TICKS"
                           * "ALL_TICKS"
                           * "NO_TICKS"
            
      :type mode:        Integer or String
      
      :param label:      An annotation of the current axis
      :type label:       String
      
      :param `**kwargs`: Keywords for (plot) attributes
      :type  `**kwargs`: Matplotlib keyword argument(s)
      
      
      :Examples:         Change the font size of the tick labels along
                         the bottom axis in 11::
      
                           grat = Graticule(...)
                           grat.setp_plotaxis(wcsgrat.bottom, fontsize=11)
      """
      #-----------------------------------------------------------------
      plotaxis = parseplotaxes(plotaxis)
      for ax in plotaxis:
         # User wants to make something visible, but right and top
         # axis labels are by default invisible. Keyword 'visible' in the
         # kwargs list can overrule this default
         #self.axes[ax].kwargs.update({'visible':True})
         if len(kwargs):
            self.axes[ax].kwargs.update(kwargs)
         if mode is not None:
            mode = parsetickmode(mode)
            self.axes[ax].mode = mode
         if label is not None:
            self.axes[ax].label = label
         if xpos is not None:
            self.axes[ax].xpos = xpos
         if ypos is not None:
            self.axes[ax].ypos = ypos


   def setp_axislabel(self, plotaxis=None, label=None, xpos=None, ypos=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Utility method that calls method :meth:`setp_plotaxis` but
      the parameters are restricted to the axis labels.
      These labels belong to one of the 4 plot axes.
      See the documentation at setp_plotaxis for the input
      of the *plotaxis* parameter. The *kwargs* are Matplotlib
      attributes.
       

      Possible useful Matplotlib attributes:

      * backgroundcolor
      * color
      * rotation
      * style or fontstyle 	[ 'normal' | 'italic' | 'oblique']
      * weight or fontweight

      :param plotaxis: The axis number of one of the axes of the plot rectangle:

            * wcsgrat.left 
            * wcsgrat.bottom
            * wcsgrat.right
            * wcsgrat.top

         or (part of) a string which can be (case insensitive)
         matched by one from 'left', 'bottom', 'right', 'top'.
                   
      :type plotaxis:    Integer or String

      :param label:      The label text.
      :type label:       String

      :param xpos:       The x position of the label in normalized device coordinates
      :type xpos:        Floating point number
        
      :param `**kwargs`: Keywords for (plot) attributes
      :type  `**kwargs`: Matplotlib keyword argument(s)
      """
      #-----------------------------------------------------------------
      if plotaxis is None:
         plotaxis = [0,1,2,3]
      self.setp_plotaxis(plotaxis, mode=None, label=label, xpos=xpos, ypos=ypos, **kwargs)


   def set_tickmode(self, plotaxis=None, mode=None):
      #-----------------------------------------------------------------
      """
      Utility method that calls method :meth:`setp_plotaxis` but
      the parameters are restricted to the tick mode.

      Each plot axis has a tick mode.

      :param plotaxis:   The axis number of one of the axes of the
                         plot rectangle:

                           * wcsgrat.left
                           * wcsgrat.bottom
                           * wcsgrat.right
                           * wcsgrat.top

                         or (part of) a string which can be (minimal & case insensitive)
                         matched by one from 'left', 'bottom', 'right', 'top'.

      :type plotaxis:    Integer or String

      :param mode:       What should this axis do with the tick
                         marks and labels?

                           * 0 = ticks native to axis type only
                           * 1 = only the tick that is not native to axis type
                           * 2 = both types of ticks (map could be rotated)
                           * 3 = no ticks

                         Or use a text that can (minimal) match one of:

                           * "NATIVE_TICKS"
                           * "SWITCHED_TICKS"
                           * "ALL_TICKS"
                           * "NO_TICKS"
                           
      :type mode:        Integer or String
      
      """
      #-----------------------------------------------------------------
      if plotaxis is None:
         plotaxis = [0,1,2,3]
      self.setp_plotaxis(plotaxis, mode=mode)




   def Insidelabels(self, wcsaxis=0, world=None, constval=None,
                    deltapx=0.0, deltapy=0.0, angle=None, addangle=0.0,
                    fun=None, fmt=None, tex=True, aspect=1.0, **kwargs):
      #-----------------------------------------------------------------
      """
      Annotate positions in world coordinates
      within the boundaries of the plot.
      This method can be used to plot positions
      on all-sky maps where there are usually no
      intersections with the enclosing axes rectangle.
      
      
      :param wcsaxis:    Values are 0 or 1, corresponding to the
                         first and second world coordinate types.
                         The accepted values are 0 and 1. The default
                         is 0.
      :type wcsaxis:     Integer 
      
      :param world:      One or a sequence of world coordinates on the axis given
                         by *wcsaxis*. The positions are completed
                         with one value for *constval*.
                         If world=None (the default) then the world
                         coordinates are copied from graticule
                         world coordinates.
      :type world:       One or a sequence of floating point number(s) or None
      
      :param constval:   A constant world coordinate to complete the positions
                         at which a label is plotted. The value can also be a
                         string representing a sexagesimal number.
      :type constval:    Floating point number or String
      
      :param deltapx:    Small shift in pixels in x-direction of text. This enables
                         us to improve the layout of the plot by preventing that
                         labels are intersected by lines.
      :type deltapx:     Floating point number.
      
      :param deltapy:    See description at *deltapx*.
      :type deltapy:     Floating point number.
      
      :param angle:      Use this angle (in degrees) instead of
                         calculated defaults. It is the angle at which
                         then **all**
                         position labels are plotted.
      :type angle:       Floating point number
      
      :param addangle:   Add this angle (in degrees) to the calculated
                         default angles.
      :type addangle:    Floating point number

      :param fun:        Function or lambda expression to convert the
                         label value.
      :type func:        Python function or lambda expression
      
      :param fmt:        String to format the numbers. If omitted the
                         format '%g' is used.
      :type fmt:         String

      :param tex:        Format these 'inside' labels in LaTeX if this
                         parameter is set to True (which is the default).
      :type param:       Boolean
      
      :param aspect:     The aspect ratio of the frame. This number is needed to
                         plot labels at the right angle. It cannot be derived from
                         the aspect ratio of the frame, because at the moment of
                         creation, the frame is not known (only after a call to
                         the plot() method, a frame is known). If the aspect ratio
                         is known in the calling environment, we should use it
                         there to get the angles right.
 
      :type aspect:      Floating point number


      :param `**kwargs`: Keywords for (plot) attributes.
      :type  `**kwargs`: Matplotlib keyword argument(s)
      
      :returns:   An Insidelabel object with a series of derived label objects.
                  These label objects
                  have a number of attributes, see :class:`Insidelabels`
      
      :Notes:     For a map with only one spatial axis, the value of
                  'mixpix' is used as pixel value for the
                  matching spatial axis. The *mixed()* method
                  from module *wcs* is used to calculate the right
                  positions.
      
      :Examples:  Annotate a plot with labels at positions from a list
                  with longitudes at given fixed latitude:: 
      
                     grat = Graticule(...)
                     lon_world = [0,30,60,90,120,150,180]
                     lat_constval = 30
                     inlabs = grat.Insidelabels(wcsaxis=0,
                                                world=lon_world,
                                                constval=lat_constval,
                                                color='r')
      """
      #-----------------------------------------------------------------
      if world is None:
         if wcsaxis == 0:
            world = self.xstarts
         if wcsaxis == 1:
            world = self.ystarts
      if not issequence(world):
         world = [world,]

      if constval is None:
         if wcsaxis == 0:
            constval = self.ystarts[0]
         else:            
            constval = self.xstarts[0]

      if isinstance(constval, six.string_types):
         pos, err = parsehmsdms(constval)
         if err != '':
            raise Exception(err)
         else:
            constval = pos[0]

      unknown = numpy.nan
      wxlim0 = self.wxlim[0]
      wxlim1 = self.wxlim[1]
      #if self.wxlim[0] < 0.0:
      #   wxlim0 += 180.0
      #   wxlim1 += 180.0
      insidelabels = Insidelabels(wcsaxis)                       # Initialize the result
      if len(world) > 0 and wcsaxis in [0,1]:
         if wcsaxis == 0:
            defkwargs = {'ha':'center', 'va':'center', 'fontsize':10}
            phi = 0.0
            for xw in world:
               #if xw < 0.0:
               #   xw += 180.0
               if self.mixpix is None:     # Could be projection with matching axis
                  wt = (xw, constval)
                  xp, yp = self.gmap.topixel(wt)
               else:
                  wt = (xw, constval, unknown)
                  pixel = (unknown, unknown, self.mixpix)
                  (wt, pixel) = self.gmap.mixed(wt, pixel)
                  xp = pixel[0]; yp = pixel[1]
               labval = xw
               if xw < 0.0 and self.gmap.types[wcsaxis] == 'longitude':
                  labval += 360.0
               if not numpy.isnan(xp):
                  #if wxlim0 <= xw < wxlim1 and self.pxlim[0] < xp < self.pxlim[1]:
                  if self.pxlim[0]-0.5 < xp < self.pxlim[1]+0.5 and self.pylim[0]-0.5 < yp < self.pylim[1]+0.5:
                     if angle is None:
                        if self.mixpix is None:
                           d = (self.wylim[1] - self.wylim[0])/200.0 
                           xp1, yp1 = self.gmap.topixel((xw, constval-d))
                           xp2, yp2 = self.gmap.topixel((xw, constval+d))
                           if not (numpy.isnan(xp1) or numpy.isnan(xp2)):
                              yp1 *= aspect; yp2 *= aspect
                              phi = numpy.arctan2(yp2-yp1, xp2-xp1)*180.0/numpy.pi
                              if self.gmap.cdelt[1] < 0.0:
                                 phi -= 180.0
                              if phi < 0:
                                 phi += 360.0
                              if 90 <= phi < 270:
                                 phi += 180.0
                        else:
                           phi = 90.0                                                       
                     else:
                        phi = angle
                     defkwargs.update({'rotation':phi+addangle})
                     defkwargs.update(kwargs)
                     insidelabels.append(xp+deltapx, yp+deltapy,
                                         xw, constval,
                                         labval,
                                         phi+addangle,
                                         self.gmap.types[wcsaxis],
                                         skysys=self.__skysys,   #self.gmap.skyout,
                                         fun=fun,
                                         fmt=fmt,
                                         offset=False,
                                         prec=self.prec[wcsaxis],
                                         tex=tex,
                                         **defkwargs)

         if wcsaxis == 1:
            defkwargs = {'ha':'center', 'va':'center', 'fontsize':10}
            for yw in world:
               phi = 0.0
               if self.mixpix is None:
                  wt = (constval, yw)
                  xp, yp = self.gmap.topixel(wt)
               else:
                  wt = (constval, yw, unknown)
                  pixel = (unknown, unknown, self.mixpix)
                  (wt, pixel) = self.gmap.mixed(wt, pixel)
                  xp = pixel[0]; yp = pixel[1]
               labval = yw 
               if yw < 0.0 and self.gmap.types[wcsaxis] == 'longitude':
                  labval += 360.0
               if not numpy.isnan(xp):
                  if self.wylim[0] <= yw < self.wylim[1] and self.pylim[0] < yp < self.pylim[1] and self.pxlim[0] < xp < self.pxlim[1]:
                     # Delta's make minus sign more visible on graticule lines
                     if angle is None:
                        if self.mixpix is None:
                           d = (self.wxlim[1] - self.wxlim[0])/200.0 
                           xp1, yp1 = self.gmap.topixel((constval-d, yw))
                           xp2, yp2 = self.gmap.topixel((constval+d, yw))
                           if not (numpy.isnan(xp1) or numpy.isnan(xp2)):
                              yp1 *= aspect; yp2 *= aspect
                              phi = numpy.arctan2(yp2-yp1, xp2-xp1)*180.0/numpy.pi
                              if self.gmap.cdelt[0] < 0.0:
                                 phi -= 180.0
                              if phi < 0:
                                 phi += 360.0
                              if 90 <= phi < 270:
                                 phi += 180.0
                                 
                     else:
                        phi = angle
                     defkwargs.update({'rotation':phi+addangle})
                     defkwargs.update(kwargs)
                     insidelabels.append(xp+deltapx, yp+deltapy,
                                         constval, yw,
                                         labval,
                                         phi+addangle,
                                         self.gmap.types[wcsaxis],
                                         skysys=self.__skysys,  #self.gmap.skyout,
                                         fun=fun,
                                         fmt=fmt,
                                         offset=False,
                                         prec=self.prec[wcsaxis],
                                         tex=tex,
                                         **defkwargs)
         insidelabels.pxlim = self.pxlim
         insidelabels.pylim = self.pylim
      self.objlist.append(insidelabels)   # This graticule stores a list of derived objects
      return insidelabels


   def Ruler(self, pos1=None, pos2=None,
             x1=None, y1=None, x2=None, y2=None, lambda0=0.5, step=None,
             world=False, angle=None, addangle=0.0,
             fmt=None, fun=None, fliplabelside=False, mscale=None,
             labelsintex=True, **kwargs):
      #-----------------------------------------------------------------
      """
      Look at documentation of same method of class Annotatedimage.
      This is a version that is needed to calculate offsets along
      the plot axes.
      """
      #-----------------------------------------------------------------
      ruler = rulers.Ruler(self.gmap, self.mixpix, self.pxlim, self.pylim,
                           pos1=pos1, pos2=pos2,
                           x1=x1, y1=y1, x2=x2, y2=y2, lambda0=lambda0, step=step,
                           world=world, angle=angle, addangle=addangle,
                           fmt=fmt, fun=fun, fliplabelside=fliplabelside, mscale=mscale,
                           labelsintex=labelsintex, **kwargs)
      return ruler




# ==========================================================================
#                           function _update_metrics()
# --------------------------------------------------------------------------
# The function _update_metrics() is a replacement for the method
# matplotlib.mathtext.Char._update_metrics(). It tries to modify
# the metrics of characters 'm' and 's' so that hms tick labels
# are aligned correctly. To prevent unwanted side effects in other
# text labels, this is done only when the font is 'rm' and the current
# fontsize is less than variable 'tweakhms' in an attempt to
# limit the metrics modification to minute and second superscripts.
#
def _update_metrics(self):
   global tweakhms
   metrics = self._metrics = self.font_output.get_metrics(
      self.font, self.font_class, self.c, self.fontsize, self.dpi)
   if self.c in ['m', 's'] and \
      self.fontsize<=float(tweakhms) and \
      self.font=='rm':
      metrics_ms = self.font_output.get_metrics(
         self.font, self.font_class, 'h', self.fontsize, self.dpi)
      metrics.iceberg = self._metrics.iceberg = metrics_ms.iceberg
      metrics.height = self._metrics.height = metrics_ms.height
   if self.c == ' ':
      self.width = metrics.advance
   else:
      self.width = metrics.width
   self.height = metrics.iceberg
   self.depth = -(metrics.iceberg - metrics.height)

from  matplotlib.mathtext import Char
Char._update_metrics = _update_metrics