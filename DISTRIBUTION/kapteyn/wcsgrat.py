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
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#----------------------------------------------------------------------

"""
Module wcsgrat
==============
A graticule is a system of crossing lines on a map representing.
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
are based module *wcs* which is also part of the Kapteyn Package.
Module :mod:`wcs` is a Python binding for Mark R. Calabretta's library
`WCSLIB <http://www.atnf.csiro.au/people/mcalabre/WCS>`_.
>From *WCSLIB* we use only the core transformation routines.
Header parsing is done with module :mod:`wcs`.

Axes types that are not recognized by this software is treated as being linear.
The axes type correspond with keyword *CTYPEn* in a FITS file.
The information from a FITS file is retrieved by module
`PyFITS <http://www.stsci.edu/resources/software_hardware/pyfits>`_

Below you will find a reference to a tutorial with working examples. 

.. seealso:: Tutorial material:
   
     * Tutorial maputils module
       which contains many examples with source code,
       see :ref:`maputils_tutorial`.

     * Figure gallery 'all sky plots'
       with many examples of Graticule constructors,
       see :ref:`allsky_tutorial`.
   
.. moduleauthor:: Martin Vogelaar <gipsy@astro.rug.nl>

.. versionadded:: 1.0 Support for offset labels


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

Class Plotversion
-----------------

.. autoclass:: Plotversion


Class Graticule
---------------

.. autoclass:: Graticule
.. autoclass:: WCStick
"""

# TODO sequencetype uitbreiden met numpy
# Controle op input ruler coordinaten

from kapteyn import wcs        # The Kapteyn Python binding to WCSlib, including celestial transformations
from types import TupleType, ListType, StringType
import numpy


__version__ = '1.0'
(left,bottom,right,top) = range(4)                 # Names of the four plot axes
(native, notnative, bothticks, noticks) = range(4)
sequencelist = (TupleType, ListType)   # Tuple with sequence types


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
   if type(plotaxes) not in sequencelist:
      plotaxes = [plotaxes,]
   if type(plotaxes) == TupleType:
      plotaxes = list(plotaxes)
   for i,pa in enumerate(plotaxes):
      if type(pa) == StringType:
         if "LEFT".find(pa.upper()) == 0:
            plotaxes[i] = left
         elif "BOTTOM".find(pa.upper()) == 0:
            plotaxes[i] = bottom
         elif "RIGHT".find(pa.upper()) == 0:
            plotaxes[i] = right
         elif "TOP".find(pa.upper()) == 0:
            plotaxes[i] = top
         else:
            raise ValueError, "[%s] Cannot identify this plot axis!" % pa
   for pa in plotaxes:                           # Check validity
      if pa < 0 or pa > 3:
         raise ValueError, "Cannot identify this plot axis!"
   aset = {}
   map(aset.__setitem__, plotaxes, [])
   return aset.keys()



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
   if type(tickmode) == StringType:
      if "NATIVE_TICKS".find(tickmode.upper()) == 0:
         tickmode = native
      elif "SWITCHED_TICKS".find(tickmode.upper()) == 0:
         tickmode = notnative
      elif "ALL_TICKS".find(tickmode.upper()) == 0:
         tickmode = bothticks
      elif "NO_TICKS".find(tickmode.upper()) == 0:
         tickmode = noticks
      else:
         raise ValueError, "[%s] Cannot identify this tick mode!" % tickmode
   if tickmode < 0 or tickmode > 3:
      raise ValueError, "%d does not correspond to a supported tick mode!" % tickmode
   return tickmode
      


class Plotversion(object):
   """
Return an object which serves as a container for plot objects.
These plot objects are all related to the World Coordinate System (WCS)
provided by Mark R. Calabretta's library WCSLIB
(http://www.atnf.csiro.au/people/mcalabre/WCS).

The contents of the container is plotted with method plot().
The classes are designed with the idea that other libraries than Matplotlib
could be added in the future.

:param interface:
   A string that sets the plot package that will
   be used to plot the graticule.
   Currently only Matplotlib (i.e. string: 'matplotlib')
   is supported.
:type interface: String
:param fig:
   An object made with Matplotlib's *figure function*.
:type fig: Matplotlib *Figure* instance
:param frame:
   We prefere to call the Axes instance a *frame* to avoid confusion
   with the plural of axis.
:type frame: Matplotlib *Axes* instance
:param tex:
   If set to True, the labels are converted to TeX
   strings in the functions that convert degrees to hms/dms notation.
:type tex: Boolean
:raises:
   :exc:`ValueError`
      *Matplotlib expects an figure and axes instance!*
      Matplotlib expects an figure and axes instance!
      This exception is generated if the plot package
      Matplotlib was selected, but no figure instance
      or an Matplotlib Axes object was given.
   :exc:`NotImplementedError`
      *Cannot initialize. Currently only Maplotlib is supported!*

:Example:
   The next example demonstrates the outline of the procedure
   which plots graticule lines and wcs labels with Matplotlib.
   We assume that a header object (see documentation of the *wcs* module)
   is available::

         from matplotlib import pyplot

         # Assume you have a header object
         fig = pyplot.figure()
         frame = fig.add_subplot(111)
         gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
         grat = wcsgrat.Graticule(header)
         pixellabels = grat.Pixellabels()
         ruler = grat.Ruler()
         gratplot.add( [grat,pixellabels,ruler] )

   One can also put objects from other sources in the same container
   as long as the corresponding axis limits are the same.

:Notes:
   It should not be too difficult to support 
   other plot software (e.g. ppgplot) but it will be a lot of
   work to translate plot attributes to properties known
   by the other package/module. Matplotlib offers a
   nice plot canvas which allows for user interaction like panning and
   zooming. It rescales a plot when the size of the plot window is changed.
   This enables a user to modify the default layout which is
   not always optimal.

**Methods:**

.. automethod:: add

   """
   
   def __init__(self, interface='matplotlib', fig=None, frame=None):
      if interface == 'matplotlib':
         if frame == None or fig == None:
            raise ValueError, "Matplotlib expects an figure and axes instance!"
      else:
         raise NotImplementedError, "Cannot initialize. Currently only Maplotlib is supported!"

      self.plotrectangle = True
      if interface == 'matplotlib':
         self.fig = fig
         self.frame = frame
         self.frames = []
         self.graticules = []
         self.gridframes = []
         self.rulers = []
         self.images = []
         self.insidelabs = []
         from string import join, letters
         from random import choice
         self.baselabel = join([choice(letters) for i in range(8)], "")
         self.frameindx = 0
         self.firstframe = False  # was True
         self.frames.append(frame)
         frame.xaxis.set_visible(False)
         frame.yaxis.set_visible(False)


   def add(self, objlist):
      """
      Add object to plot container. These objects can be related to different
      world coordinate systems as long as the dimensions in pixels correspond.
      See the graticules tutorial for an example).

      :parameter obj:
         Object of class :class:`Graticule`, Ruler or Gridframe or Insidelabels.
         Objects from class Ruler are created with Graticule's method
         :meth:`Graticule.Ruler`. Objects from class Gridframe
         are created with Graticule's method
         :meth:`Graticule.Pixellabels` and Objects from class Insidelabels
         are created with Graticule's method :meth:`Graticule.Insidelabels`

      """
      if type(objlist) not in sequencelist:
         objlist = [objlist]
      for obj in objlist:
         # Do not use isinstance() here because we don't have the types available in other modules
         # Protect against unknown objects:
         try:
            t = obj.ptype
         except:
            raise Exception("Unknown object. Cannot plot this!")
         if obj.ptype == "Graticule":
            #self.graticules.append(obj)
            self.__plot1graticule(obj)
         elif obj.ptype == "Ruler":
            #self.rulers.append(obj)
            self.__plotruler(obj)
         elif obj.ptype == "Gridframe":
            #self.gridframes.append(obj)
            self.__plot1grid(obj)
         elif obj.ptype == "Pixellabels":
            self.__plot1grid(obj.gridlabs)
         elif obj.ptype == "Insidelabels":
            #self.insidelabs.append(obj)
            self.__plotinsidelabels(obj)

   def __plot1graticule(self, graticule):
      """
      -----------------------------------------------------------
      Purpose:      Plot the graticule lines and labels
                    Labels are either along the plot axes or 
                    inside the plot.
      Parameters:
        graticule - An object from class Graticule
      Returns:      --
      Notes:        --
      -----------------------------------------------------------
      """
      # We need to sort and format the ticks for the 4 plot axes
      tex = graticule.labelsintex
      pos, lab, kwargs, size = graticule.sortticks(tex=tex)
      aspect = self.frame.get_aspect()
      adjust = self.frame.get_adjustable()
      if self.firstframe:
         frame = self.frame
         self.firstframe = False
      else:
         framelabel = "F%s%d" % (self.baselabel, self.frameindx)
         self.frameindx += 1
         frame = self.fig.add_axes(self.frame.get_position(), 
                                    aspect=aspect, 
                                    adjustable=adjust, 
                                    autoscale_on=False, 
                                    frameon=False, 
                                    label=framelabel)
         self.frames.append(frame)
      graticule.frame = frame  # !!!! DOCUMENTEREN
      xlo = graticule.pxlim[0]-0.5; ylo = graticule.pylim[0]-0.5; 
      xhi = graticule.pxlim[1]+0.5; yhi = graticule.pylim[1]+0.5  
      frame.set_yticks(pos[left])
      frame.set_yticklabels(lab[left])
      for tick, kw, msize in zip(frame.yaxis.get_major_ticks(), kwargs[left], size[left]):
         tick.label1.set(**kw)
         if msize != None:
            tick.tick1line.set_markersize(msize)
         tick.tick2on = False
         tick.tick2line.set_visible(False)

      frame.set_xticks(pos[bottom])
      frame.set_xticklabels(lab[bottom])

      for tick, kw, msize in zip(frame.xaxis.get_major_ticks(), kwargs[bottom], size[bottom]):
         tick.label1.set(**kw)
         if msize != None:
            tick.tick1line.set_markersize(msize)
         tick.tick2on = False
         tick.tick2line.set_visible(False) 

      framelabel = "S%s%d" % (self.baselabel, self.frameindx)
      self.frameindx += 1
      frame2 = self.fig.add_axes(frame.get_position(), frameon=False, label=framelabel)
      self.frames.append(frame2)
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
      for tick, kw, msize in zip(frame2.xaxis.get_major_ticks(),kwargs[top], size[top]):
         tick.label2.set(**kw)
         if msize != None:
            tick.tick2line.set_markersize(msize)
         tick.tick1line.set_visible(False)
      
      frame2.set_yticks(pos[right])
      frame2.set_yticklabels(lab[right])
      for tick, kw, msize in zip(frame2.yaxis.get_major_ticks(),kwargs[right], size[right]):
         tick.label2.set(**kw)
         if msize != None:
            tick.tick2line.set_markersize(msize)
         tick.tick1line.set_visible(False)
      
      frame.set_ylabel( graticule.axes[left].label,   **graticule.axes[left].kwargs)
      frame.set_xlabel( graticule.axes[bottom].label, **graticule.axes[bottom].kwargs)
      frame2.set_ylabel(graticule.axes[right].label,  **graticule.axes[right].kwargs)
      frame2.set_xlabel(graticule.axes[top].label,    **graticule.axes[top].kwargs)

      # Plot the line pieces
      for gridline in graticule.graticule:
         for line in gridline.linepieces:
            frame.plot(line[0], line[1], **gridline.kwargs)
      # set the limits of the plot axes
      # this setting can be overwritten in the calling environment
      frame.set_xlim((xlo,xhi))
      frame.set_ylim((ylo,yhi))
      frame.set_aspect(aspect=aspect, adjustable=adjust)

      self.fig.sca(self.frame)    # back to frame from calling environment


   def __plotinsidelabels(self, insidelabels):
      """
      -----------------------------------------------------------
      Purpose:         Plot world coordinate labels inside a plot
      Parameters:
         insidelabels - Object from class Insidelabels created with 
                       method Graticule.Insidelabels
      Returns:         --
      Notes:
      -----------------------------------------------------------
      """
      for inlabel in insidelabels.labels:
         self.frame.text(inlabel.x, inlabel.y, inlabel.label, clip_on=True, **inlabel.kwargs)

      # Set limits
      xlo = insidelabels.pxlim[0]-0.5
      ylo = insidelabels.pylim[0]-0.5
      xhi = insidelabels.pxlim[1]+0.5
      yhi = insidelabels.pylim[1]+0.5
      self.frame.set_xlim((xlo,xhi))
      self.frame.set_ylim((ylo,yhi))

   def __plot1grid(self, pixellabels):
      """
      -----------------------------------------------------------
      Purpose:         Plot labels that annotate the pixels
                       (or another system if an offset is given)
      Parameters:
         pixellabels - Object from class Gridframe made with 
                       method Graticule.Pixellabels
      Returns:         --
      Notes:           This method can only plot the grid labels along
                       two axes. If you need to label the other axes 
                       too, then add another grid with method Pixellabels().

                       Only one frame is plotted. Needs maintenance
      -----------------------------------------------------------
      """
      plotaxes = parseplotaxes(pixellabels.plotaxis)  # Is always a list with integers now!
      # What are the combinations:
      # Only one axis 0, 1, 2, 3
      # Two axes 0,1 - 0,2 - 0,3
      #          1,2 - 1,3
      #          2,3
      #
      # The combinations 0,1 - 0,3 - 1,2 - 2,3 can be plotted in 1 frame
      # For 0,2 - 1,3 we need two frames. Then we raise an exception
      # and suggest a user to plot it in two steps
      # If there are more than two plot axes then raise also
      # an exception.

      if len(plotaxes) > 2:
         raise Exception, "Can plot labels for a maximum of 2 axes. Please split up!"

      if (0 in plotaxes and 2 in plotaxes) or (1 in plotaxes and 3 in plotaxes):
         raise Exception, "Cannot plot labels for this combination. Please split up!"
      
      aspect = self.frame.get_aspect()
      adjust = self.frame.get_adjustable()
      kwargs = pixellabels.kwargs
      xlo = pixellabels.pxlim[0]-0.5 
      ylo = pixellabels.pylim[0]-0.5
      xhi = pixellabels.pxlim[1]+0.5
      yhi = pixellabels.pylim[1]+0.5
      # Copy frame
      framelabel = "G%s%d" % (self.baselabel, self.frameindx)
      self.frameindx += 1
      gframe = self.fig.add_axes(self.frame.get_position(),
                                 aspect=aspect,
                                 adjustable=adjust,
                                 autoscale_on=False,
                                 frameon=False,
                                 label=framelabel)
      
      gframe.set_xlim((xlo,xhi))
      gframe.set_ylim((ylo,yhi))
      self.frames.append(gframe)

      if 3 in plotaxes:
         gframe.xaxis.set_label_position('top')
         gframe.xaxis.tick_top()
      elif 1 in plotaxes:
         gframe.xaxis.set_label_position('bottom')
         gframe.xaxis.tick_bottom()
      else:  # both not available -> make invisible
         for tick in gframe.xaxis.get_major_ticks():
            tick.label2.set(visible=False)
            tick.label1.set(visible=False)
           
      setmarker = pixellabels.markersize != None
      for tick in gframe.xaxis.get_major_ticks():
         if 3 in plotaxes:
            tick.label2.set(**kwargs)
            if setmarker:
               tick.tick2line.set_markersize(pixellabels.markersize)
            tick.tick1On = False
         elif 1 in plotaxes:
            tick.label1.set(**kwargs)
            if setmarker:
               tick.tick1line.set_markersize(pixellabels.markersize)
            tick.tick2On = False
               
      if 2 in plotaxes:
         gframe.yaxis.set_label_position('right')
         gframe.yaxis.tick_right()
      elif 0 in plotaxes:
         gframe.yaxis.set_label_position('left')
         gframe.yaxis.tick_left()
      else:
         for tick in gframe.yaxis.get_major_ticks():
            tick.label2.set(visible=False)
            tick.label1.set(visible=False)
         
      for tick in gframe.yaxis.get_major_ticks():
         if 2 in plotaxes:
            tick.label2.set(**kwargs)
            if setmarker:
               tick.tick2line.set_markersize(pixellabels.markersize)
            tick.tick1line.set_visible(False)
         elif 0 in plotaxes:
            tick.label1.set(**kwargs)
            if setmarker:
               tick.tick1line.set_markersize(pixellabels.markersize)
            tick.tick2line.set_visible(False)
               
      gframe.grid(pixellabels.gridlines)
      self.fig.sca(self.frame)    # back to frame from calling environment


   def __plotruler(self, ruler):
      """
      Plot one ruler object in the current frame
      """
      self.frame.plot((ruler.x1,ruler.x2), (ruler.y1,ruler.y2), '-', **ruler.linekwargs)
      dx = ruler.tickdx
      dy = ruler.tickdy
      #self.frame.plot( [ruler.x1, ruler.x1+dx], [ruler.y1, ruler.y1+dy], '-', **ruler.linekwargs)
      #self.frame.plot( [ruler.x2, ruler.x2+dx], [ruler.y2, ruler.y2+dy], '-', **ruler.linekwargs)
      for x, y, label in zip(ruler.x, ruler.y, ruler.label):
         self.frame.plot( [x, x+dx], [y, y+dy], '-', color='k')
         self.frame.text(x+ruler.mscale*dx, y+ruler.mscale*dy, label, **ruler.kwargs)

      # Set limits explicitly
      xlo = ruler.pxlim[0]-0.5
      ylo = ruler.pylim[0]-0.5
      xhi = ruler.pxlim[1]+0.5
      yhi = ruler.pylim[1]+0.5
      self.frame.set_xlim((xlo,xhi))
      self.frame.set_ylim((ylo,yhi))

   def moeteruit_plot(self):
      """
      Plot the container objects.
      
      :raises:
         :exc:`Exception` 
            *No graticules set yet!*
      """
      # plot all
      if len(self.graticules) == 0:
         raise Exception,"No graticules set yet!"
      for graticule in self.graticules:
         self.__plot1graticule(graticule)
      for pixellabels in self.gridframes:
         self.__plot1grid(pixellabels)
      if len(self.graticules) > 0:
         # Draw enclosing rectangle
         xlo = self.graticules[0].pxlim[0]-0.5; xhi = self.graticules[0].pxlim[1]+0.5
         ylo = self.graticules[0].pylim[0]-0.5; yhi = self.graticules[0].pylim[1]+0.5
         rectx = (xlo, xhi, xhi, xlo, xlo)
         recty = (ylo, ylo, yhi, yhi, ylo)
         self.frame.plot(rectx,recty, color='k')
      for ruler in self.rulers:
         self.__plotruler(ruler)
      # !!! Hier nog de insidelabels aan toevoegen


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
      self.markersize = None  # Length of the tick lines 
      self.kwargs = {}        # Keyword arguments to set plot attributes


class Gratline(object):
   """
   -------------------------------------------------------------------------------
   This class is used to find a set of coordinates that defines 
   (part of) a graticule line for which one of the world coordinates
   is a constant. It stores the coordinates of the intersections
   with the box and a corresponding label for annotating purposes.
   -------------------------------------------------------------------------------
   """
   def __init__(self, wcsaxis, constval, gmap, pxlims, pylims, wxlims, wylims, 
                linesamples, mixgrid, skysys, addx=None, addy=None, addpixels=False, offsetlabel=None,
                fun=None, fmt=None):

      """
      -----------------------------------------------------------
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
      -----------------------------------------------------------
      """

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
         if x >= box[2] or x <= box[0] or y >= box[3] or y <= box[1]:
            return False
         else: 
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
      elif wcsaxis == 1:             # constant value on Y axis
         dw = (wxlims[1] - wxlims[0])/1000.0
         xw = numpy.linspace(wxlims[0]-dw, wxlims[1]+dw, linesamples)
         yw = numpy.zeros(linesamples) + constval
      else:                    # A grid line without one of the coordinates being constant e.g. a border
         xw = addx
         yw = addy
      if (mixgrid == None):    # Probably matching axis pair of two independent axes
         world = (xw, yw)
         if wcsaxis in [0,1]:
            pixel = gmap.topixel(world)
         else:              # This should be an 'added' grid line
            if addpixels:   # A set with pixels for a border is already provided
               pixel = (addx, addy)
            else:           # A set with world coordinates for a border is provided.
               pixel = gmap.topixel(world)
      else:                    # wcs coordinate pair with one extra grid coordinate
         unknown = numpy.zeros(linesamples)
         unknown += numpy.nan
         zp = numpy.zeros(linesamples) + mixgrid
         world = (xw, yw, unknown)
         pixel = (unknown, unknown, zp)
         (world, pixel) = gmap.mixed(world, pixel)
      if wcsaxis == 0 or wcsaxis == 1:
         self.axtype = gmap.types[wcsaxis]
      else:
         # E.g. grid lines made without a constant value do not belong to an axis
         # so we use a dummy type 'border' 
         self.axtype = 'border'
         
      self.skysys = skysys
      self.wcsaxis = wcsaxis
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
         if not numpy.isnan(xp) and not numpy.isnan(yp):  # NaN's can occur with divergent projections
            currentinside = __inbox(xp, yp, box)
            if lastx != None  and (lastinside or currentinside):
               # These jump conditions are somewhat arbitrary.
               # If the projection diverges then these steps sizes may be
               # not enough to have a full coverage.
               jump = abs(lastx-xp) > 10.0*stepx or abs(lasty-yp) > 10.0*stepy
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
                  if offsetlabel != None:
                     labelvalue = offsetlabel
                     offs = True
                  else:
                     labelvalue = constval
                     offs = False
                  tick = WCStick(xlab, ylab, axisnr, labelvalue, wcsaxis, offs, fun=fun, fmt=fmt)
                  if wcsaxis == 0:
                     tick.kwargs.update({'fontsize':'11'})
                  else:
                     tick.kwargs.update({'fontsize':'11'})
                  self.ticks.append(tick)
               x.append(xlab); y.append(ylab)  # Add this intersection as element of the graticule line
            if crossing2out:
               self.linepieces.append( (x,y) )
            lastx = xp; lasty = yp
            lastinside = currentinside

      if countin > 0:                   # Store what is left over
         self.linepieces.append( (x,y) )


class WCSaxis(object):
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
      Notes:        Each plot axis is associated with a WCSaxis 
                    instance.
      -----------------------------------------------------------
      """
      self.axisnr = axisnr         # left=0, bottom=1, right=2, top=3
      self.mode = mode             # Set which (native/not native) labels should be stored for this axis 
      self.label = label           # Default title for this axis
      self.kwargs = kwargs         # Keyword aguments for the axis labels



class Insidelabels(object):
   """
   A small utility class for wcs labels inside a plot with a graticule
   """
   class Ilabel(object):
      def __init__(self, Xp, Yp, lab, rots, **kwargs):
         self.x = Xp
         self.y = Yp
         self.label = lab
         self.rotation = rots
         self.kwargs = kwargs

   def __init__(self):
      self.labels = []
      self.ptype = "Insidelabels"
   def append(self, Xp, Yp, lab, rots, **kwargs):
      ilab = self.Ilabel(Xp, Yp, lab, rots, **kwargs)
      self.labels.append(ilab)


class Ruler(object):
   """
   -------------------------------------------------------------------------------
   Attributes and methods for a ruler object. A ruler is a line piece with
   a start- end endpoint and along this line there are labels which 
   annotates values of constant offset w.r.t. a selected point on the line.
   -------------------------------------------------------------------------------
   """
   def __init__(self, x1, y1, x2, y2, angle, dx, dy, mscale, **kwargs):
      self.ptype = "Ruler"
      self.x1 = x1
      self.y1 = y1
      self.x2 = x2
      self.y2 = y2
      self.x = []
      self.y = []
      self.xw = []
      self.yw = []
      self.stepsizeW = None
      self.label = []
      self.offsets = []      # Store the offsets in degrees
      self.angle = angle
      self.kwargs = {}        # {'clip_on' : True}   # clip_on is buggy for plot() in MPL versions <= 0.98.3 change later
      self.tickdx = dx
      self.tickdy = dy
      self.mscale = mscale
      self.fun = None
      self.fmt = None
      self.linekwargs = {'color' : 'k'}
      self.kwargs.update(kwargs)    # These are the kwargs for the labels
      
   def setp_line(self, **kwargs):
      self.linekwargs.update(kwargs)
   
   def setp_labels(self, **kwargs):
      self.kwargs.update(kwargs)
      
   def append(self, x, y, offset, label):
      self.x.append(x)
      self.y.append(y)
      self.offsets.append(offset)
      self.label.append(label)

   def appendW(self, xw, yw):
      self.xw.append(xw)
      self.yw.append(yw)



class Gridframe(object):
   """
   -------------------------------------------------------------------------------
   Helper class which defines objects with properties which are read
   when pixel coordinates need to be plotted.
   -------------------------------------------------------------------------------
   """
   def __init__(self, pxlim, pylim, plotaxis, markersize, gridlines, **kwargs):
      self.ptype = "Gridframe"
      self.pxlim = pxlim
      self.pylim = pylim
      self.plotaxis = plotaxis
      self.markersize = markersize
      self.kwargs = kwargs
      self.gridlines = gridlines;


class Graticule(object):
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
                  have at least values for the properties listed above.
                  Then these objects could be used as a shortcut parameter.
:type graticuledata:  Object with some required attributes

:param axnum:     This parameter sets which FITS axis corresponds
                  to the x-axis of your graticule plot rectangle
                  and which one corresponds to the y-axis
                  (see also description at *pxlim and *pylim*).
                  The first axis in a FITS file is axis 1.
                  If *axnum* set to *None* then the default
                  FITS axes will be 1 and 2.
                  With a sequence you can set different FITS axes
                  like ``axnum=(1,3)`` Then the input is a tuple
                  or a list.
:type axnum:      None, Integer or sequence of Integers

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
                  See also article
                  `Representations of spectral coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/scs.pdf>`_
                  by Greisen, Calabretta, Valdes & Allen.
                  Module *maputils* from the Kapteyn Package provides
                  a method that creates a list with possible spectral
                  translations given a arbitrary header.
:type spectrans:  String

:param skyout:    A single number or a tuple which specifies
                  the celestial system.
                  The syntax for the tuple is:
                  ``(sky system, equinox, reference system,
                  epoch of observation)``.
                  Predefined are the systems:

                     * wcs.equatorial
                     * wcs.ecliptic,
                     * wcs.galactic
                     * wcs.supergalactic
                  
                  Predefined reference systems are:

                     * wcs.fk4,
                     * wcs.fk4_no_e,
                     * wcs.fk5,
                     * wcs.icrs,
                     * wcs.j2000

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
 
                
                  See the documentation of module *celestial* (part of the
                  Kapteyn Package) to read more details.
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
:type wxlim:      *None* or exactly two Floating point numbers

:param wylim:     See wxlim, but now applied for the y-axis
:type wylim:      *None* or exactly two Floating point numbers

:param boxsamples: Number of random pixel positions within a box
                  with limits *pxlim* and *pylim* for which world
                  coordinates are calculated to get an estimate of
                  the range in world coordinates (see description
                  at wxlim). The default is listed in the argument list
                  of this method. If speed is essential one can try smaller
                  numbers than the default.
:type boxsamples: Integer

:param startx:    If one value given then this is the
                  first graticule line that has a constant
                  x **world coordinate** equal to *startx*.
                  The other values will be
                  calculated, either with distance *deltax*
                  between them or
                  with a default distance calculated by this method.
                  If *None* is set, then a suitable value will be
                  calculated.
:type startx:     *None* or 1 Floating point number or a sequence
                  of Floating point numbers

:param starty:    [None, one value, sequence]
                  Same for the graticule line with constant
                  y world coordinate equal to starty.
:type startx:     *None* or 1 Floating point number or a sequence
                  of Floating point numbers
                  
:param deltax:    Step in **world coordinates** along the x-axis
                  between two subsequent graticule lines.
:type deltax:     *None* or a floating point number

:param deltay:    Same as deltax but now as step in y direction.
:type deltay:     *None* or a floating point number

:param skipx:     Do not calculate the graticule lines with the
                  constant world coordinate that is associated
                  with the x-axis.
:type skipx:      Boolean

:param skipy:     The same as skipx but now associated with
                  the y-axis.
:type skipy:      Boolean

:param gridsamples: Number of positions on a graticule line for which
                    a pixel position is calculated and stored as part
                    of the graticule line. If *None* is set than the
                    default is used (see the argument list of this method).
:type gridsamples:  Integer

:param labelsintex: The default is that all labels are formatted for LaTeX.
:type labelsintex:  Boolean

:param offsetx:     Change the default mode which sets either plotting
                    the labels for the given -or calculated world coordinates-
                    or plotting labels which represents constant offsets
                    with respect to a given starting point.
                    The offset mode is default for plots with mixed axes,
                    i.e. with only one spatial axis. In spatial maps
                    this offset mode it is not
                    very useful to plot the graticule lines because these lines
                    are plotted at a constant world coordinate and do not know
                    about offsets.
                    The offset axes correspond to the pixel positions of
                    start- and endpoint of
                    the left and bottom axes and the start point of
                    the offsets (value 0) is at the centre of the axis.
:type offsetx:      *None* or Boolean

:param offsety:     Same as *offsetx* but now for the left plot axis.
:type offsety:      *None* or Boolean

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
   
   :exc:`ValueError` *pxlim must have two elements*
   
   :exc:`ValueError` *pylim needs to be of type tuple or list*
   
   :exc:`ValueError` *pylim must have two elements*
   
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
   
   :exc:`ValueError` *wxlim must have two elements*
   
   :exc:`ValueError` *wylim needs to be of type tuple or list*
   
   :exc:`ValueError` *wylim must have two elements*
   
   :exc:`ValueError` *boxsamples < 2: Need at least two samples to find limits*
      There is a minimum number of random positions we have to
      calculate to get an impression of the axis limits in world coordinates.

   :exc:`ValueError` *Number of samples along graticule line must be >= 2 to avoid a step size of zero*
      The value of parameter *gridsamples* is too low. Low values give
      distorted graticule lines. Hogher values (like the default) give
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
                        
                  Read also docstring for WCSaxis class.
                  Four WCSaxis instances, one for each axis of the 
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
                        
                     WCSaxis modes are:
                         
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
                  The attribute is implemented as a read-only attribute.

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
                  as calculated by the constructor.
                  This attribute is meant as a read-only attribute.

.. attribute:: ystarts
   
                  Same for the y-axis


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
                  one can swap the graticule plat axes with input parameter
                  *axnum*::
                        
                     #3. Swap x and y- axis in a FITS file
                     grat = wcsgrat.Graticule(header, axnum= (2,1))

                  For data with more than two axes, one can select the axes
                  with input parameter *axnum*::
                        
                     #4. For a FITS file with axes (RA,DEC,FREQ) 
                     #  create a graticule for the FREQ,RA axes:
                     grat = wcsgrat.Graticule(header, axnum=(3,1))

                  See also example in description of class :class:`Plotversion`
                  
**Methods which set (plot) attributes:**

.. automethod::   setp_tick
.. automethod::   setp_plotaxis
.. automethod::   setp_lineswcs0
.. automethod::   setp_lineswcs1
.. automethod::   setp_gratline

**Methods that deal with special curves like borders:**

.. automethod::   scanborder
.. automethod::   addgratline
.. automethod::   setp_linespecial

**Methods related to storing and plotting elements:**

.. automethod::   Insidelabels
.. automethod::   Ruler
.. automethod::   Pixellabels
.. automethod::   plot

**Utility methods:**

.. automethod::   get_aspectratio

   """

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
      tickpos = [[],[],[],[]]
      ticklab = [[],[],[],[]]
      tickkwa = [[],[],[],[]]
      ticksize = [[],[],[],[]]
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
                # There are some conditions for plotting labels in hms/dms:
                if gridline.axtype in ['longitude', 'latitude'] and t.offset == False and t.fmt == None and t.fun == None:
                  if gridline.axtype == 'longitude':
                     if (gridline.skysys == wcs.equatorial):
                        lab = wcs.lon2hms(t.labval, prec=self.prec[wcsaxis], delta=self.delta[wcsaxis], tex=tex)
                     else:
                        lab = wcs.lon2dms(t.labval, prec=self.prec[wcsaxis], delta=self.delta[wcsaxis], tex=tex)
                  else:    # must be latitude
                     lab = wcs.lat2dms(t.labval, prec=self.prec[wcsaxis], delta=self.delta[wcsaxis], tex=tex)
                else:
                   if t.fun == None:
                      val = t.labval
                   else:
                      val = t.fun(t.labval)
                   if t.fmt == None:
                      lab = "%g" % val
                   else:
                      lab = t.fmt % val
                      if tex:
                         lab = r"%s" % lab
                      #if tex and not t.fmt.startswith('$'):
                      #   lab = r"$%s$"% lab
                ticklab[anr].append(lab)
                if anr in [left,right]:
                   tickpos[anr].append(t.y)
                else:
                   tickpos[anr].append(t.x)
                tickkwa[anr].append(t.kwargs)
                ticksize[anr].append(t.markersize)

      return tickpos, ticklab, tickkwa, ticksize


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

      if delta == None:
         if axtype in ['longitude', 'latitude']:
            # Nice numbers for dms should also be nice numbers for hms 
            sec = numpy.array([30, 20, 15, 10, 5, 2, 1])
            minut = sec
            deg = numpy.array([60, 30, 20, 15, 10, 5, 2, 1])
            nicenumber = numpy.concatenate((deg*3600.0, minut*60.0, sec))
            if skysys == wcs.equatorial and axtype == 'longitude':
               fact = 240.0
            else:
               fact = 3600.0

            x1 *= fact; x2 *= fact
            d = x2 - x1
            step2 = 0.9*d/2.0         # We want at least two graticule lines in this range
            for p in nicenumber:
               k = int(step2/p)
               if k >= 1.0:
                  step2 = k * p
                  step = step2
                  # nc = nicenumber
                  break           # Stop if we have a candidate

         if step == None:
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
         if skysys == wcs.equatorial and axtype == 'longitude':
            pstep *= 240.0
         else:
            pstep *= 3600.0
      # Set the precision for the seconds
      f0 = numpy.floor( numpy.log10(pstep) )
      if f0 < 0.0:
         prec = int(abs(f0))
      else:
         prec = 0
      
      startx = None
      if start != None:
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
      if self.mixpix == None:
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
         raise Exception, "Could not find enough (>1) valid world coordinates in this map!"
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
                pxlim=None, pylim=None, 
                mixpix=None, spectrans=None, skyout=None, alter='', 
                wxlim=None, wylim=None,
                boxsamples=5000, 
                startx=None, starty=None, deltax=None, deltay=None,
                skipx=False, skipy=False,
                gridsamples=1000,
                labelsintex=True,
                offsetx=None, offsety=None):
      """
     -----------------------------------------------------------
      Purpose:    Creates an object that defines a graticule
                  See class documentation above.
      -----------------------------------------------------------
      """
      self.ptype = 'Graticule'
      if graticuledata != None:
         header = graticuledata.hdr
         axnum  = graticuledata.axperm
         pxlim  = graticuledata.pxlim
         pylim  = graticuledata.pylim
         mixpix = graticuledata.mixpix
         # Allow these to be overwritten
         if spectrans == None:
            spectrans = graticuledata.spectrans
         if skyout == None:
            skyout = graticuledata.skyout
         
      # Try to get two axis numbers if none are given
      if axnum == None:
         naxis = header['NAXIS']
         if naxis < 2:
            raise Exception, "Need data with at least two axes"
         else:
            self.xaxnum = 1
            self.yaxnum = 2
      else:
         if len(axnum) != 2:
            raise Exception, "Need two axis numbers to create a graticule"
         else:
           self.xaxnum = axnum[0]
           self.yaxnum = axnum[1]
      if (self.xaxnum == self.yaxnum):
         raise Exception, "Need two different axis numbers"
      # Get the axes limits in pixels. The default is copied from
      # the values of NAXISn in the header. Note that there are no alternative
      # keywords for NAXIS.
      if pxlim == None:
         self.pxlim = (1, header['NAXIS' +str(self.xaxnum)])
      else:
         if type(pxlim) not in sequencelist:
            raise Exception, "pxlim needs to be of type tuple or list"
         elif len(pxlim) != 2:
            raise Exception,"pxlim must have two elements"
         else: 
            self.pxlim = pxlim
      if pylim == None:
         self.pylim = (1, header['NAXIS' +str(self.yaxnum)])
      else:
         if type(pylim) not in sequencelist:
            raise Exception, "pylim needs to be of type tuple or list"
         elif len(pylim) != 2:
            raise Exception, "pylim must have two elements"
         else:
            self.pylim = pylim

      # wcs.debug=True
      # Create the wcs projection object
      if spectrans == None:
         proj = wcs.Projection(header, skyout=skyout, alter=alter)
      else:
         proj = wcs.Projection(header, skyout=skyout, alter=alter)   #.spectra(spectrans)
      
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
         if mixpix == None:
            mixpix = proj.source['CRPIX'+str(self.matchingaxnum)+proj.alter]
         if mixpix == None:
            raise Exception, "Could not find a grid for the missing spatial axis"
         ok = proj.lonaxnum != None and proj.lataxnum != None 
         if not ok:
            raise Exception, "Could not find a matching spatial axis pair"
         gmap = proj.sub([self.xaxnum, self.yaxnum, self.matchingaxnum])
         self.mixpix = mixpix
      else:
         gmap = proj.sub([self.xaxnum, self.yaxnum])
         self.mixpix = None

      # Set the spectral translation and make Projection object an attribute
      if spectrans != None:
         self.gmap = gmap.spectra(spectrans)
      else:
         self.gmap = gmap

      self.gmap.allow_invalid = True
      
      # For the labeling format (hms/dms) we need to know the sky system.
      # If the input was a tuple, then it must be equatorial or ecliptical
      if type(proj.skyout) == TupleType:
         s = proj.skyout[0]
         try:
            if s < 4:              # Vervangen door len(xxx)
               # This is a sky system
               self.__skysys = s
            else:
               self.__skysys = 0
         except:
               self.__skysys = 0
      else:
         self.__skysys = proj.skyout

      # Now we have a projection object available and we want the limits of the axes in 
      # world coordinates. If nothing is specified for the constructor, we have
      # to calculate estimates of these ranges.
      if wxlim != None:
         if type(wxlim) not in sequencelist:
            raise Exception, "wxlim needs to be of type tuple or list"
         elif len(wxlim) != 2:
            raise Exception, "wxlim must have two elements"
         else: 
            self.wxlim = wxlim
      if wylim != None:
         if type(wylim) not in sequencelist:
            raise Exception, "wylim needs to be of type tuple or list"
         elif len(wylim) != 2:
            raise Exception, "wylim must have two elements"
         else: 
            self.wylim = wylim
      if wxlim == None or wylim == None:
         if boxsamples < 2:
            raise Exception, "boxsamples < 2: Need at least two samples to find limits"
         minmax = self.__estimateLonLatRanges(boxsamples)
         if wxlim == None:
            self.wxlim = (minmax[0], minmax[1])
         if wylim == None:
            self.wylim = (minmax[2], minmax[3])

      self.labelsintex = labelsintex
      
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
      if self.offsetx == None:
         self.offsetx = spatialx and not spatialy
      if self.offsety == None:
         self.offsety = spatialy and not spatialx

      self.prec  = [0, 0]
      self.delta = [None, None]

      self.offsetvaluesx = None
      self.offsetvaluesy = None
      self.funx = self.funy = None
      self.fmtx = self.fmty = None
      self.radoffsetx = self.radoffsety = False
      if spatialx and not spatialy and self.offsetx:
         # Then the offsets are distances on a sphere
         xmax = self.pxlim[1] + 0.5
         xmin = self.pxlim[0] - 0.5
         ymin = self.pylim[0] - 0.5
         x1 = xmin; x2 = xmax; y1 = y2 = ymin
         ruler = self.Ruler(x1, y1, x2, y2, lambda0=0.5, step=deltax)
         self.xstarts = ruler.xw
         self.prec[0] = 0
         self.delta[0] = ruler.stepsizeW
         self.offsetvaluesx = ruler.offsets
         self.funx = ruler.fun
         self.fmtx = ruler.fmt
         self.radoffsetx = True
      elif type(startx) in sequencelist or type(startx) == numpy.ndarray:
         self.xstarts = startx
         if len(startx) >= 2:
            self.delta[0] = startx[1] - startx[0]  # Assume this delta also for not equidistant values in startx
      else:
         # startx is a scalar
         if startx == None and self.offsetx:
            startx = (self.wxlim[1] + self.wxlim[0]) / 2.0
         self.xstarts, self.offsetvaluesx, self.prec[0], self.delta[0] = self.__nicenumbers(self.wxlim[0],self.wxlim[1],
                                                        start=startx, 
                                                        delta=deltax, 
                                                        axtype=self.gmap.types[0], 
                                                        skysys=self.__skysys)

      if spatialy and not spatialx and self.offsety:
         # Then the offsets are distances on a sphere
         ymax = self.pylim[1] + 0.5
         xmin = self.pxlim[0] - 0.5
         ymin = self.pylim[0] - 0.5
         x1 = xmin; x2 = xmin; y1 = ymin; y2 = ymax
         ruler = self.Ruler(x1, y1, x2, y2, lambda0=0.5, step=deltay)
         self.ystarts = ruler.xw
         self.prec[1] = 0
         self.delta[1] = ruler.stepsizeW
         self.offsetvaluesy = ruler.offsets
         self.funy = ruler.fun
         self.fmty = ruler.fmt
         self.radoffsety = True
      elif type(starty) in sequencelist or type(starty) == numpy.ndarray:
         self.ystarts = starty
         if len(starty) >= 2:
            self.delta[1] = starty[1] - starty[0]  # Assume this delta also for not equidistant values in startx
      else:
         # starty is a scalar
         if starty == None and self.offsety:
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
         raise Exception, "Number of samples along graticule line must be >= 2 to avoid a step size of zero"

      # Create the plot axes. The defaults are: plot native ticks to axis
      # for the left and bottom axis and omit ticks along right and top axis.
      epoch = str(self.gmap.equinox)
      annot = ['']*2
      for aa in [0,1]:
         if (aa == 0 and self.offsetx) or (aa == 1 and self.offsety):
            annot[aa] = "Offset " 
         if self.gmap.types[aa] in [None, 'spectral']:
            annot[aa] += self.gmap.ctype[aa].split('-')[0] + ' (' + self.gmap.units[aa] + ')'
         else:        # Must be spatial
            if (aa == 0 and self.radoffsetx):
               annot[aa] = 'Radial offset lon.'
            elif (aa == 1 and self.radoffsety):
               annot[aa] = 'Radial offset lat.'
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
               
               # annot[aa] = self.gmap.ctype[aa].split('-')[0] 


      self.graticule = []   # Initialize the list with graticule lines
      if not skipx:
         for i, x in enumerate(self.xstarts):
            offsetlabel = None
            fie = fmt = None
            if self.radoffsetx:
               offsetlabel = self.offsetvaluesx[i]
               fie = self.funx
               fmt = self.fmtx
            elif self.offsetx:
               offsetlabel = self.offsetvaluesx[i]
               fmt = "%g"
            gridl = Gratline(0, x, self.gmap,
                             self.pxlim, self.pylim, 
                             self.wxlim, self.wylim, 
                             gridsamples,self.mixpix, self.__skysys,
                             offsetlabel=offsetlabel,
                             fun=fie, fmt=fmt)
            self.graticule.append(gridl)
            gridl.kwargs = {'color': 'k', 'lw': 1}
      if not skipy:
         for i, y in enumerate(self.ystarts):
            offsetlabel = None
            fie = fmt = None
            if self.radoffsety:
               offsetlabel = self.offsetvaluesy[i]
               fie = self.funy
               fmt = self.fmty
            elif self.offsety:
               offsetlabel = self.offsetvaluesy[i]
               fmt = "%g"
            gridl = Gratline(1, y, self.gmap,
                             self.pxlim, self.pylim, 
                             self.wxlim, self.wylim, 
                             gridsamples,self.mixpix, self.__skysys,
                             offsetlabel=offsetlabel,
                             fun=fie, fmt=fmt)
            gridl.kwargs = {'color': 'k', 'lw': 1}
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
      axes.append( WCSaxis(left,   y1mode, annot[1], **kwargs1) )
      axes.append( WCSaxis(bottom, x1mode, annot[0], **kwargs2) )
      axes.append( WCSaxis(right,  y2mode, annot[1], **kwargs3) )
      axes.append( WCSaxis(top,    x2mode, annot[0], **kwargs4) )
      self.axes = axes

      # Finally set default values for aspect ratio
      dummy = self.get_aspectratio()
      self.objlist = []


   def scanborder(self, xstart, ystart, deltax=None, deltay=None, nxy=1000, tol=None):
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
point by scanning in x- and -direction and
tries to find the position of a limb with a
standard bisection technique. This method has been applied
to a number of all-sky plots with slanted projections.

:param xstart: X-coordinate in pixels of position where to
               start the scan to find border.
               The parameter has no default.
:type xstart:  Floating point 

:param ystart: Y-coordinate in pixels of position where to
               start the scan to find border.
               The parameter has no default.
:type ystart:  Floating point

:param deltax: Set range in pixels to look for border in
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
      xp = []; yp = []
      if deltax == None:
         deltax = (self.pxlim[1] - self.pxlim[0])/ 10.0
      if deltay == None:
         deltay = (self.pylim[1] - self.pylim[0])/ 10.0
      d = (float(self.pxlim[1] - self.pxlim[0]), float(self.pylim[1] - self.pylim[0])) 
      delta = (deltax, deltay)
      limlo = (self.pxlim[0], self.pylim[0])
      limhi = (self.pxlim[1], self.pylim[1])
      start = (xstart, ystart) 
      for i in [0,1]:
         if tol == None:
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
            if yb != None:
               if i == 0:
                  xp.append(xb)
                  yp.append(yb)
               else:
                  xp.append(yb)
                  yp.append(xb)
               y0 = yb

      return self.addgratline(xp, yp, pixels=True)


   def addgratline(self, x, y, pixels=False):
      """
For any path given by a set world coordinates
of which none is a constant value (e.g. borders
in slanted projections where the positions are calculated by an
external routine),
one can create a line that is processed as a graticule
line, i.e. intersections and jumps are addressed.
Instead of world coordinates, this method
can also process pixel positions. The type of input is set by the
*pixels* parameter.


:param x:      A sequence of world coordinates or pixels
               that correspond to horizontal axis in a graticule plot..
:type x:       Floating point numbers

:param y:      The same for the second axis
:type x:       Floating point numbers

:param pixels: False or True
               If False the coordinates in x and y are world-
               coordinates. Else they are pixel coordinates.
:type pixels:  Boolean

:Returns:      A **Identification number** *id* which can be used
               to set properties for this special path with
               method :meth:`setp_linespecial`.
               Return *None* if no line piece could be found
               inside the pixel limits of the graticule.

:note:         This method can be used to plot a border
               around an all-sky plot e.g. for slanted
               projections. See code at :meth:`scanborder`.
      """
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
      """
Calculate and set, the aspect ratio for the current
pixels. Also set default values for figure
size and axes lengths (i.e. size of canvas depends
on the size of plot window with this aspect ratio).

:param xcm: Given a value for xcm or ycm (or omit both),
            then suggest a figure size in inches and a viewport in
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
            'axesrect', 'figsize', 'aspectratio'
      """
      cdeltx = self.gmap.cdelt[0]
      cdelty = self.gmap.cdelt[1]
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      if xcm == None and ycm == None:
         xcm = 20.0
      aspectratio = abs(cdelty/cdeltx)
      if aspectratio > 10.0 or aspectratio < 0.1:
         aspectratio = nx/ny
         if xcm == None:
            xcm = ycm
         else:
            ycm = xcm
      if ycm == None:
         ycm = xcm * (ny/nx) * aspectratio
      if xcm == None:
         xcm = ycm * (nx/ny) / aspectratio
      fh = 0.7; fw = 0.7
      self.axesrect = (0.15, 0.15, fw, fh)
      self.figsize = (xcm/2.54/fw, ycm/2.54/fh)
      self.aspectratio = aspectratio
      return aspectratio


   def setp_tick(self, wcsaxis=None, plotaxis=None, position=None, tol=0.0, fmt=None, fun=None, markersize=None, **kwargs):
      """
Set (plot) attributes for a wcs tick label.
A tick is identified by the type of grid line
it belongs to, and/or the plot axis for which
it defined an intersection and/or a position which
corresponds to the constant value of the graticule
line.
All these parameters are valid with none, one or
a sequence of values.

.. warning:: If no value for *wcsaxis*, *plotaxis* or *position* is entered
      then this method does nothing (and without a warning or raising an exception).

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
:type position: *None* or one or a sequence of floating point numbers

:param tol:
      If a value > 0 is given, the gridline with the
      constant value closest to a given position within
      distance 'tol' gets updated attributes.
:type tol: Floating point number

:param fmt:
      A string that formats the tick value
      e.g. ``fmt="%10.5f"``
:type fmt: String

:param fun:
      An external function which will be used to
      convert the tick value e.g. to convert
      velocities from m/s to Km/s. See also
      example 2_ below.
:type fun: Python function or Lambda expression
      
:param markersize:
      Size of tick line. Use negative number (e.g. -4) to
      get tick lines that point outside the plot instead
      of the default which is inside.
:type markersize: Integer

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
      if wcsaxis == None and plotaxis == None and position == None:
         # Nothing to do
         return
      if wcsaxis != None:
         if type(wcsaxis) not in sequencelist:
            wcsa = [wcsaxis]
         else:
            wcsa = wcsaxis
      if plotaxis != None:
         plta = parseplotaxes(plotaxis)
      if position != None:
         if type(position) not in sequencelist:
            posn = [position]
         else:
            posn = position

      for gridline in self.graticule:
        skip = False
        if (wcsaxis != None):
           skip = not (gridline.wcsaxis in wcsa)
        if not skip:
           if position == None:
              for t in gridline.ticks:
                 skip = False
                 if plotaxis != None:
                    skip = not (t.axisnr in plta)
                 if not skip:
                    t.kwargs.update(kwargs)
                    # The attributes fmt and fun to format tick labels could have
                    # been initialized when the object was created. If values are
                    # given then they override the defaults.
                    if fmt != None: t.fmt = fmt
                    if fun != None: t.fun = fun
                    t.markersize = markersize
           else:      # One or more positions are given. Find the right index.
              for pos in posn:
                 d0 = None
                 for i, t in enumerate(gridline.ticks):
                    skip = False
                    if plotaxis != None:
                       skip = not (t.axisnr in plta)
                    if not skip:
                       d = abs(t.labval - pos)
                       if d <= tol:
                          if d0 == None:
                             d0 = d
                             indx = i
                          elif d < d0:
                             d0 = d
                             indx = i
                 if d0 != None:
                    gridline.ticks[indx].kwargs.update(kwargs)
                    gridline.ticks[indx].fmt = fmt
                    gridline.ticks[indx].fun = fun
                    gridline.ticks[indx].markersize = markersize


   def setp_gratline(self, wcsaxis=None, position=None, tol=0.0, **kwargs):
      """
Set (plot) attributes for one or more graticule
lines.
These graticule lines are identified by the wcs
number (*wcsaxis=0* or *wcsaxis=1*) and by their constant
world coordinate in *position*.

:param wcsaxis:    If omitted, then for both types of graticule lines
                   the attributes are set.
                   If one value is given then only for that axis
                   the attributes will be set.
:type wcsaxis:     *None* or Integer(s) from set 0, 1. 

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
                   *rotation* or *visible* etc.
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
      # Upgrade 'wcsaxis' to a sequence
      if wcsaxis == None:
         wcsaxislist = [0,1]
      elif type(wcsaxis) not in sequencelist:
         wcsaxislist = [wcsaxis]
      else:
         wcsaxislist = wcsaxis
      if position == None:
         for gridline in self.graticule:
            if gridline.wcsaxis in wcsaxislist:
               gridline.kwargs.update(kwargs)
      else: # Upgrade 'position' to a sequence
         if type(position) not in sequencelist:
            S = [position]
         else:
            S = position
         for constval in S:
            d0 = None
            for i, gridline in enumerate(self.graticule):
               if gridline.wcsaxis in wcsaxislist:
                  d = abs(gridline.constval - constval)
                  if d <= tol:
                     if d0 == None:
                        d0 = d
                        indx = i
                     else:
                        if d < d0:
                           d0 = d
                           indx = i
            if d0 != None:     # i.e. we found a closest position
               self.graticule[indx].kwargs.update(kwargs)


   def setp_lineswcs0(self, position=None, tol=0.0, **kwargs):
      """
Helper method for :meth:`setp_gratline`.
It pre-selects the grid line that
corresponds to the first world coordinate.

:Parameters:  See description at :meth:`setp_gratline`

:Returns:     --

:Notes:       --

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
      axis = 0
      self.setp_gratline(axis, position, tol, **kwargs)


   def setp_lineswcs1(self, position=None, tol=0.0, **kwargs):
      """
Helper method for :meth:`setp_gratline`.
It pre-selects the grid line that
corresponds to the second world coordinate.

:Parameters:  See description at :meth:`setp_gratline`

:Returns:       --

:Notes:         --
:Examples:     See example at :meth:`setp_lineswcs0`.

      """
      axis = 1
      self.setp_gratline(axis, position, tol, **kwargs)



   def setp_linespecial(self, id, **kwargs):
      """
Set (plot) attributes for a special type
of graticule line made with method :meth:`addgratline`
or method :meth:`scanborder`.
This graticule line has no constant x- or y- value.
It is identified by an id returned by method
:meth:`addgratline`.

:param id:          id from :meth:`addgratline`
:type id:           Integer
:param `**kwargs`:  keywords for (plot) attributes
:type `**kwargs`:   Matplotlib keyword argument(s)

:Returns:           --

:Notes:             --

:Examples:          Create a special graticule line
                    which follows the positions in two
                    given arrays *x* and *y*. and set
                    the line width for this line to 2::

                        id = grat.addgratline(x, y)
                        grat.setp_linespecial(id, lw=2)
      """
      # Set properties for an added gridline
      for gridline in self.graticule:
         if gridline.wcsaxis == id and id > 1:
            gridline.kwargs.update(kwargs)


   def setp_plotaxis(self, plotaxis, mode=None, label=None, **kwargs):
      """
Set (plot) attributes for titles along a plot axis and set the ticks mode.
The ticks mode sets the relation between the ticks and the plot axis.
For example a rotated map will show a rotated graticule, so ticks for both
axes can appear along a plot axis. With parameter *mode* one can influence this
behaviour.

.. Note::
   This method addresses the four axes of a plot seperately. Therefore
   its functionality cannot be incorporated in :meth:`setp_tick`!

:param plotaxis:   The axis number of one of the axes of the
                   plot rectangle:

                     * wcsgrat.left (== 0)
                     * wcsgrat.bottom (==1)
                     * wcsgrat.right (==2)
                     * wcsgrat.top (==3)

:type plotaxis:    Integer

:param mode:       What should this axis do with the tick
                   marks and labels?

                     * 0 = ticks native to axis type only
                     * 1 = only the tick that is not native to axis type
                     * 2 = both types of ticks (map could be rotated)
                     * 3 = no ticks
:type mode:        Integer

:param label:      An annotation of the current axis
:type label:       String

:param `**kwargs`: Keywords for (plot) attributes
:type  `**kwargs`: Matplotlib keyword argument(s)

:Returns:          --

:Note:             --

:Examples:         Change the font size of the tick labels along
                   the bottom axis in 11::

                     grat = Graticule(...)
                     grat.setp_plotaxis(wcsgrat.bottom, fontsize=11)
      """
      plotaxis = parseplotaxes(plotaxis)
      for ax in plotaxis:
         # User wants to make something visible, but right and top
         # axis labels are default invisible. Keyword 'visible' in the
         # kwargs list can overrule this default
         self.axes[ax].kwargs.update({'visible':True})
         if len(kwargs):
            self.axes[ax].kwargs.update(kwargs)
         if mode != None:
            mode = parsetickmode(mode)
            self.axes[ax].mode = mode
         if label != None:
            self.axes[ax].label = label


   def Insidelabels(self, wcsaxis=0, world=None, constval=None, deltapx=0.0, deltapy=0.0, angle=None, addangle=0.0, fmt=None, **kwargs):
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

:param world:      One or more world coordinates on the axis given
                   by *wcsaxis*. The positions are completed
                   with one value for *constval*.
                   If world=None (the default) then the world
                   coordinates are copied from graticule
                   world coordinates.
:type world:       Floating point number(s) or None

:param constval:   A constant world coordinate to complete the positions
                   at which a label is plotted.
:type constval:    Floating point number

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

:param fmt:        String to format the numbers. If omitted the
                   format '%g' is used.
:type fmt:         String

:param `**kwargs`: Keywords for (plot) attributes.
:type  `**kwargs`: Matplotlib keyword argument(s)

:returns:   A list with *insidelabel* objects. These objects
            have attributes: X-position, Y-position, a label that represents
            a world coordinate, a
            rotation angle and a dictionary with keyword
            arguments for plot attributes.
            The attribute names of an *insidelabel* object are:

            * *Xp*   - The X-positions in pixels, corrected for *deltapx*
            * *Yp*   - The Y-positions in pixels, corrected for *deltapy*
            * *lab*  - List with labels for each position (Xp, Yp)
            * *rots* - List with angles, one for each label and all in degrees
            * *kwargs* - Matplotlib keyword argument(s)

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
         
      if world == None:
         if wcsaxis == 0:
            world = self.xstarts
         if wcsaxis == 1:
            world = self.ystarts
      if type(world) not in sequencelist and not isinstance(world, numpy.ndarray):
         world = [world,]

      if constval == None:
         if wcsaxis == 0:
            if self.wylim[0] <= 0.0 <= self.wylim[1]:
               constval = 0.0
            else:
               constval = (self.wylim[1] + self.wylim[0]) / 2.0
         if wcsaxis == 1:
            if self.wxlim[0] <= 0.0 <= self.wxlim[1]:
               constval = 0.0
            else:
               constval = (self.wxlim[1] + self.wxlim[0]) / 2.0

      if fmt == None:
         fmt = '%g'
      unknown = numpy.nan
      wxlim0 = self.wxlim[0]
      wxlim1 = self.wxlim[1]
      #if self.wxlim[0] < 0.0:
      #   wxlim0 += 180.0
      #   wxlim1 += 180.0
      insidelabels = Insidelabels()                       # Initialize the result
      if len(world) > 0 and wcsaxis in [0,1]:
         if wcsaxis == 0:
            defkwargs = {'ha':'center', 'va':'center', 'fontsize':10}
            phi = 0.0
            for xw in world:
               #if xw < 0.0:
               #   xw += 180.0
               if self.mixpix == None:     # Could be projection with matching axis
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
               s =  fmt%labval
               if not numpy.isnan(xp):
                  #if wxlim0 <= xw < wxlim1 and self.pxlim[0] < xp < self.pxlim[1]:
                  if self.pxlim[0]-0.5 < xp < self.pxlim[1]+0.5 and self.pylim[0]-0.5 < yp < self.pylim[1]+0.5:
                     if angle == None:
                        if self.mixpix == None:
                           d = (self.wylim[1] - self.wylim[0])/200.0 
                           xp1, yp1 = self.gmap.topixel((xw, constval-d))
                           xp2, yp2 = self.gmap.topixel((xw, constval+d))
                           if not (numpy.isnan(xp1) or numpy.isnan(xp2)):
                              phi = numpy.arctan2(yp2-yp1, xp2-xp1)*180.0/numpy.pi
                              if self.gmap.cdelt[1] < 0.0:
                                 phi -= 180.0
                     else:
                        phi = angle
                     defkwargs.update({'rotation':phi})
                     defkwargs.update(kwargs)
                     insidelabels.append(xp+deltapx, yp+deltapy, s, phi+addangle, **defkwargs)

         if wcsaxis == 1:
            defkwargs = {'ha':'right', 'va':'bottom', 'fontsize':10}
            ha = 'right'
            va = 'bottom'
            for yw in world:
               phi = 0.0
               if self.mixpix == None:
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
               s =  fmt%labval
               if not numpy.isnan(xp):
                  if self.wylim[0] <= yw < self.wylim[1] and self.pylim[0] < yp < self.pylim[1] and self.pxlim[0] < xp < self.pxlim[1]:
                     # Delta's make minus sign more visible on graticule lines
                     if angle == None:
                        if self.mixpix == None:
                           d = (self.wxlim[1] - self.wxlim[0])/200.0 
                           xp1, yp1 = self.gmap.topixel((constval-d, yw))
                           xp2, yp2 = self.gmap.topixel((constval+d, yw))
                           if not (numpy.isnan(xp1) or numpy.isnan(xp2)):
                              phi = numpy.arctan2(yp2-yp1, xp2-xp1)*180.0/numpy.pi
                              if self.gmap.cdelt[0] < 0.0:
                                 phi -= 180.0
                     else:
                        phi = angle
                     defkwargs.update({'rotation':phi})
                     defkwargs.update(kwargs)
                     insidelabels.append(xp+deltapx, yp+deltapy, s, phi+addangle, **defkwargs)
         insidelabels.pxlim = self.pxlim
         insidelabels.pylim = self.pylim
         self.objlist.append(insidelabels)
         return insidelabels


   def Ruler(self, x1=None, y1=None, x2=None, y2=None, lambda0=0.5, step=None,
             world=False, angle=None, addangle=0.0, 
             fmt=None, fun=None, fliplabelside=False, mscale=None, **kwargs):
      """
Draw a line between two spatial positions
from a start point (x1,y1) to an end point (x2,y2)
with labels indicating a constant offset in world
coordinates. The positions are either in pixels
or in world coordinates. The ruler is a straight
line but the ticks are usually not equidistant
because projection effects make the offsets non linear.
Default, the zero point is exactly in the middle of
the ruler but this can be changed by setting a
value for *lambda0*.  The step size
for the ruler ticks in units of the spatial
axes is entered in parameter *step*.
At least one of the axes in the plot needs to be
a spatial axis.

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

:param step:          Step size in units of world coordinates that corresponds to
                      the spatial axis (i.e. degrees).
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
      a sphere for the current projection system. So we nead at least
      one spatial axis and if there is only one spatial axis in the plot,
      then we need a matching spatial axis.
   :exc:`Exception`
      *Cannot make ruler with step size equal to zero!*
      Either the input of the step size...
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

:Examples:     Create a ruler object and add it to plot container::

                  ruler = grat.Ruler(x1,y1,x2,y2)
                  gratplot.add(grat)
                  gratplot.add(pixellabels)
                  gratplot.add(ruler)
                  gratplot.plot()

               Practical example of a vertical ruler positioned
               at the right side of a plot. Note that position 0.5
               is a plot boundary. It corresponds with the
               lower or left side of the first pixel::

                  xmax = grat.pxlim[1]+0.5; ymax = grat.pylim[1]+0.5
                  ruler = grat.Ruler(xmax,0.5, xmax, ymax, lambda0=0.5, step=5.0/60.0,
                                     fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$",
                                     fliplabelside=True, color='r')
                  ruler.setp_line(lw='2', color='g')
                  ruler.setp_labels(color='y')
                  gratplot.add(ruler)

      """
      # Recipe
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
               break                         # Succes..., leave the while loop
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
         if self.mixpix == None:
            xw, yw = self.gmap.toworld((x,y))
            xwo = xw     # Store originals
            ywo = yw
         else:
            xw1, xw2, yw1 = self.gmap.toworld((x, y, self.mixpix))
            if self.gmap.types[0] == 'longitude':
               xw = xw1
               yw = yw1
               xwo = xw1; ywo = yw1
            elif self.gmap.types[0] == 'latitude':  # First axis must be latitude
               xw = yw1
               yw = xw1
               xwo = xw1; ywo = yw1
            elif self.gmap.types[1] == 'longitude':
               xw = xw2
               yw = yw1
               xwo = xw2; ywo = yw1
            elif self.gmap.types[1] == 'latitude':
               xw = yw1
               yw = xw2
               xwo = xw2; ywo = yw1
            else:
               xw = yw = numpy.nan

         return xw, yw, xwo, ywo


      def topixel2(xw, yw):
         # Note that this conversion is only used to convert 
         # start and end position, given in world coordinates,
         # to pixels.
         if self.mixpix == None:
            x, y = self.gmap.topixel((xw,yw))
         else:
            unknown = numpy.nan
            wt = (xw, yw, unknown)
            pixel = (unknown, unknown, self.mixpix)
            (wt, pixel) = self.gmap.mixed(wt, pixel)
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

      # Set defaults for missing pixel coordinates
      if x1 == None: x1 = self.pxlim[0]
      if x2 == None: x2 = self.pxlim[1]
      if y1 == None: y1 = self.pylim[0]
      if y2 == None: y2 = self.pylim[1]

      spatial = self.gmap.types[0] in ['longitude', 'latitude'] or self.gmap.types[1] in ['longitude', 'latitude']
      if not spatial:
         raise Exception, "Rulers only suitable for maps with at least one spatial axis!"

      if world:
         x1, y1 = topixel2(x1, y1)
         x2, y2 = topixel2(x2, y2)
      # Check whether the start- and end point of the ruler are inside the frame

      # Get a step size for nice offsets
      if step == None:
         stepsizeW = nicestep(x1, y1, x2, y2)
      else:
         stepsizeW = step
      if step == 0.0:
         raise Exception, "Cannot make ruler with step size equal to zero!"


      # Look for suitable units (degrees, arcmin, arcsec) if nothing is
      # specified in the call. Note that 'stepsizeW' is in degrees.
      if fun == None and fmt == None:
         if self.labelsintex:
            fmt = r"$%4.0f^{\circ}$"
         else:
            fmt = u"%4.0f\u00B0"
         if abs(stepsizeW) < 1.0:
            # Write labels in arcmin
            fun = lambda x: x*60.0
            if self.labelsintex:
               fmt = r"$%4.0f^{\prime}$"
            else:
               fmt = r"$%4.0f'"
         if abs(stepsizeW) < 1.0/60.0:
            # Write labels in arcmin
            fun = lambda x: x*3600.0
            if self.labelsintex:
               fmt = r"$%4.0f^{\prime\prime}$"
            else:
               fmt = r"$%4.0f''"
      elif fmt == None:          # Then a default format
         fmt = '%g'
      
      start_in = (self.pxlim[0]-0.5 <= x1 <= self.pxlim[1]+0.5) and (self.pylim[0]-0.5 <= y1 <= self.pylim[1]+0.5)
      if not start_in:
         raise Exception, "Start point of ruler not in pixel limits!"
      end_in = (self.pxlim[0]-0.5 <= x2 <= self.pxlim[1]+0.5) and (self.pylim[0]-0.5 <= y2 <= self.pylim[1]+0.5)
      if not end_in:
         raise Exception, "End point of ruler not in pixel limits!"

      # Ticks perpendicular to ruler line.
      defangle = 180.0 * numpy.arctan2(y2-y1, x2-x1) / numpy.pi - 90.0

      l1 = self.pxlim[1] - self.pxlim[0] + 1.0; l1 /= 50.0
      l2 = self.pylim[1] - self.pylim[0] + 1.0; l2 /= 50.0
      ll = max(l1,l2)
      dx = ll*numpy.cos(defangle*numpy.pi/180.0)
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
      ruler = Ruler(x1, y1, x2, y2, defangle, dx, dy, mscale, **defkwargs)
      ruler.fmt = fmt
      ruler.fun = fun
      
      lambda_s = lambda0
      x0 = x1 + lambda_s*(x2-x1)
      y0 = y1 + lambda_s*(y2-y1)
      Xw, Yw, xw1, yw1 = tolonlat(x0, y0)
      ruler.append(x0, y0, 0.0, fmt%0.0)
      ruler.appendW(xw1, yw1)         # Store in original order i.e. not sorted
      ruler.stepsizeW = stepsizeW     # Needed elsewhere so store as an attribute


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
                  ruler.append(x, y, offset, fmt%off)
                  xw, yw, xw1, yw1 = tolonlat(x, y)
                  ruler.appendW(xw1, yw1)
            elif sign == -1.0:
               break
               # raise Exception, mes
      ruler.pxlim = self.pxlim
      ruler.pylim = self.pylim
      try:
         self.objlist.append(ruler)
      except:
         pass
      return ruler


   def Pixellabels(self, plotaxis=None, markersize=None, gridlines=False, offset=None, **kwargs):
      """
      Plot pixel coordinates
      """
      pixobj = Pixellabels(self.pxlim, self.pylim,
                           plotaxis, markersize, gridlines,
                           offset, **kwargs)
      self.objlist.append(pixobj)
      return pixobj
   
      
   def plot(self, frame):
      """
      Plot all objects stored in attribute *objlist*.
      """
      container = Plotversion("matplotlib", frame.figure, frame)
      #try:
      visible = self.visible
      #except:
         #visible = True
      if visible:
         container.add(self)
      if len(self.objlist) > 0:
         container.add(self.objlist)


class Pixellabels(object):
   """
   Draw positions in pixels along one or more
   plot axes. Nice numbers and step size are
   calculated by Matplotlib's own plot methods.
   
   
   :param plotaxis:     The axis number of one or two of the axes of the
                        plot rectangle:
   
                        * wcsgrat.left
                        * wcsgrat.bottom
                        * wcsgrat.right
                        * wcsgrat.top

                        or 'left', 'bottom', 'right', 'top'
                        
   :type  plotaxis:     Integer
   
   :param markersize:   Set size of ticks at pixel positions.
                        The size can be negative to get tick
                        marks that point outwards.
   :type  markersize:   Integer
   
   :param gridlines:    Set plotting of grid lines (connected tick marks)
                        on or off (True/False). The default is off.
   :type gridlines:     Boolean
   
   :param offset:       The pixels can have an integer offset.
                        If you want the reference pixel to be pixel
                        0 then supply offset=(crpixX, crpixY).
                        These crpix values are usually read from then
                        header. In this routine the nearest integer of
                        the input is calculated to ensure that the
                        offset is an integer value.
   :type offset:        *None* or a floating point number
   
   :param `**kwargs`:   Keyword arguments to set attributes for
                        the labels.
   :type `**kwargs`:    Matplotlib keyword argument(s)
   
   :Returns:            An object from class *Gridframe* which
                        is added to the plot container with Plotversion's
                        method :meth:`Plotversion.add`.
   
   :Notes:              --
   
   :Examples:           Annotate the pixels in a plot along the right and top axis
                        of a plot. Change the color of the labels to red::
   
                           mplim = f.Annotatedimage(frame)
                           ima = mplim.Image(visible=False)
                           mplim.Pixellabels(plotaxis=("bottom", "right"), color="r")

                           or with separate axes:
   
                           mplim.Pixellabels(plotaxis="bottom", color="r")
                           mplim.Pixellabels(plotaxis="right", color="b", markersize=10)
                           mplim.Pixellabels(plotaxis="top", color="g", markersize=-10)

   """
   def __init__(self, pxlim, pylim, plotaxis=None, markersize=None,
                gridlines=False, offset=None, **kwargs):

      def nint(x):
         return numpy.floor(x+0.5)
   
      self.ptype = "Pixellabels"       # not a gridframe object
      defkwargs = {'fontsize':7}
      defkwargs.update(kwargs)
      if plotaxis == None:
         plotaxis = [2,3]

      px = [0,0]; py = [0,0]
      px[0] = pxlim[0]; py[0] = pylim[0]    # Do not copy directly because new values must be temporary
      px[1] = pxlim[1]; py[1] = pylim[1]
      if offset != None:
         offX = nint(offset[0])
         offY = nint(offset[1])
         px[0] -= offX; px[1] -= offX;
         py[0] -= offY; py[1] -= offY;
   
      gridlabs = Gridframe(px, py, plotaxis, markersize, gridlines, **defkwargs)
      self.gridlabs = gridlabs

   def plot(self, frame):
      container = Plotversion("matplotlib", frame.figure, frame)
      container.add(self.gridlabs)
   

      
#--------End of file--------
