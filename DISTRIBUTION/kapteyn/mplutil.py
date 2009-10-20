"""
Module mplutil
==============

.. moduleauthor:: Hans Terlouw <J.P.Terlouw@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5

Utilities for use with matplotlib.
Classes :class:`AxesCallback`, :class:`TimeCallback`
and :class:`VariableColormap`
and module-internal function :func:`KeyPressFilter`.

Class AxesCallback
------------------

.. autoclass:: AxesCallback

Class TimeCallback
------------------

.. autoclass:: TimeCallback

Class VariableColormap
----------------------

.. autoclass:: VariableColormap

Key press filter
----------------

Via its internal function :func:`KeyPressFilter` the module filters key_press
events for the backend in which the application displays its contents.
By default all key_press events are discarded by the filter and do not reach
the backend. This behaviour can be changed by assigning a list of acceptable
keys to KeyPressFilter's attribute *allowed*. E.g.,
``KeyPressFilter.allowed = ['g', 'f']`` will allow characters ``g`` and ``f``
to reach the backend so that the backend's grid- and full-screen toggles
will be available again.

"""
# ==========================================================================
#                             class AxesCallback
# --------------------------------------------------------------------------
class AxesCallback(object):
   """
:class:`AxesCallback` has been built on top of matplotlib's event
handling mechanism. Objects of this class provide a more powerful
mechanism for handling events from :class:`LocationEvent` and derived classes
than matplotlib provides itself.
This class allows the programmer to register a callback function with
an event type combined with an Axes object. Whenever the event occurs
within the specified Axes object, the callback function is called
with the AxesCallback object as its single argument. Different from
matplotlib-style event handlers, it is possible to handle overlapping
Axes objects. An AxesCallback object will not be deleted as long as it
is scheduled ("active"), so it is not always necessary to keep a reference
to it.

:param proc:
   the function to be called upon receiving an event of the specified
   type and occurring in the specified Axes object. It is called with one
   argument: the current AxesCallback object. If it returns a value which
   evaluates to True, processing of the current event stops, i.e., no
   further callback functions will be called for this event.
:param axes:
   the matplotlib Axes object.
:param eventtype:
   the matplotlib event type such as 'motion_notify_event' or 'key_press_event'.
:param schedule:
   indicates whether the object should start handling events immediately.
   Default True.
:param attr:
   keyword arguments each resulting in an attribute with the same name.
   
**Attributes:**

.. attribute:: axes

   The specified axes object.
   
.. attribute:: canvas

   The FigureCanvas object to which `axes` belongs.

.. attribute:: eventtype

   The specified event type.
   
.. attribute:: active

   True if callback is scheduled, False otherwise.
   
.. attribute:: xdata, ydata

   The cursor position in data coordinates within the specified Axes object.
   These values may be different from the attributes with the same name
   of the event object.
   
.. attribute:: event

   The Event object delivered by matplotlib.

**Methods:**

.. automethod:: schedule
.. automethod:: deschedule
   
**Example:**

::

   #!/usr/bin/env python
   
   from matplotlib.pyplot import figure, show
   from kapteyn.mplutil import AxesCallback
   
   def draw_cb(cb):
      if cb.event.button:
         if cb.pos is not None:
            cb.axes.plot((cb.pos[0], cb.xdata), (cb.pos[1], cb.ydata), cb.c)
            cb.canvas.draw()
         cb.pos = (cb.xdata, cb.ydata)
      else:
         cb.pos = None
   
   def colour_cb(cb):
      cb.drawer.c = cb.event.key
   
   fig = figure()
   
   frame = fig.add_axes((0.1, 0.1, 0.8, 0.8))
   frame.set_autoscale_on(False)
   
   draw = AxesCallback(draw_cb, frame, 'motion_notify_event', pos=None, c='r')
   setc = AxesCallback(colour_cb, frame, 'key_press_event', drawer=draw)
   
   show()

The above code implements a complete, though very simple, drawing program. It
first creates a drawing frame and then connects two :class:`AxesCallback`
objects to it.
The first object, `draw`, connects to the callback function :func:`draw_cb`,
which will draw line segments as long as the mouse is moved with a button down.
The previous position is "remembered" by `draw` via its attribute :attr:`pos`.
The drawing colour is determined by `draw`'s attribute :attr:`c` which
can be modified by the callback function :func:`colour_cb` by typing
one of the letters 'r', 'g', 'b', 'y', 'm', 'c', 'w' or 'k'. This callback
function is called via the second AxesCallback object `setc` which has the
first :class:`AxesCallback` object `draw` as an attribute.

"""

   __scheduled = []                           # currently scheduled callbacks
   __handlers  = {}                           # currently active event handlers

   def __init__(self, proc, axes, eventtype, schedule=True, **attr):
      self.proc      = proc
      self.axes      = axes
      self.eventtype = eventtype
      self.canvas    = axes.get_figure().canvas
      for name in attr.keys():
         self.__dict__[name] = attr[name]
      self.active    = False
      if schedule:
         self.schedule()
      
   def schedule(self):
      """
      Activate the object so that it will start receiving matplotlib events
      and calling the callback function. If the object is already
      active, it will be put in front of the list of active
      objects so that its callback function will be called before others.
      """

      if self.active:
         self.__scheduled.remove(self)        # remove from current position ..
         self.__scheduled.insert(0, self)     # .. and move to front of list
         return                               # no further action
      # Try to find a handler and increment the number of
      # registrations for this canvas-eventtype combination.
      # If no handler can be found, connect this event type
      # to __handler() and register this combination.
      try:
         id, numreg = self.__handlers[self.canvas, self.eventtype]
         self.__handlers[self.canvas, self.eventtype] = id, numreg+1
      except KeyError:
         id = self.canvas.mpl_connect(self.eventtype, self.__handler)
         self.__handlers[self.canvas,self.eventtype] = id, 1
      self.active = True                      # mark active
      self.__scheduled.insert(0, self)        # insert in active list
      
   def deschedule(self):
      """
      Deactivate the object so that it does not receive matplotlib events
      anymore and will not call its callback function. If the object is
      already inactive, nothing will be done.
      """
      
      if not self.active:
         return                               # no action, stays inactive
      id, numreg = self.__handlers[self.canvas, self.eventtype]
      numreg -= 1                             # decrement number of callbacks
      if numreg==0:                           # was this the last one?
         del self.__handlers[self.canvas, self.eventtype]  # remove registration
         self.canvas.mpl_disconnect(id)       # disconnect handler
      else:
         self.__handlers[self.canvas, self.eventtype] = id,  numreg
      self.active = False                     # mark inactive
      self.__scheduled.remove(self)           # remove from active list

   def __handler(event):
      for callback in AxesCallback.__scheduled:
         if event.canvas is callback.canvas   and \
               event.name==callback.eventtype and \
               callback.axes.contains(event)[0]:
            callback.event = event
            callback.xdata, callback.ydata = \
               callback.axes.transData.inverted().transform((event.x, event.y))
            if callback.proc(callback):
               break
   __handler = staticmethod(__handler)


# ==========================================================================
#                          class VariableColormap
# --------------------------------------------------------------------------
import numpy, math
from numpy import ma
from matplotlib.colors import Colormap
from matplotlib import cm
from tabarray import tabarray

class VariableColormap(Colormap):
   """
:class:`VariableColormap` is a subclass of
:class:`matplotlib.colors.Colormap` with special methods that allow the
colormap to be modified. A VariableColormap can be constructed from
any other matplotlib colormap object
or from a textfile with one RGB triplet per
line. Values should be between 0.0 and 1.0.

:param source:
   the object from which the VariableColormap is created. Either an other
   colormap object or its registered name,
   or the name of a text file containing RGB triplets.
:param name:
   the name of the color map.
   
  
**Attributes:**

.. attribute:: auto 

   Indicates whether Axes objects registered with method :meth:`add_frame`
   will be automatically updated when the colormap changes. Default True.

.. attribute:: slope

   The colormap slope as specified with method :meth:`modify`.

.. attribute:: shift

   The colormap shift as specified with method :meth:`modify`.

.. attribute:: scale

   The colormap's current scale as specified with methon :meth:`set_scale`.

.. attribute:: source

   The object (string or colormap) from which the colormap is currently
   derived.
 
**Methods**

.. automethod:: modify
.. automethod:: set_scale
.. automethod:: set_source
.. automethod:: add_frame
.. automethod:: remove_frame
.. automethod:: update
""" 

   def __init__(self, source, name='Variable'):
      self.name = None                      # new VariableColormap object
      self.set_source(source)
      self.monochrome = False
      Colormap.__init__(self, name, self.worklut.shape[0]-3)
      self.canvases = {}
      self.frames = set()
      self.slope = 1.0
      self.shift = 0.0
      self.invrt = 1.0
      self.scale = 'LINEAR'
      self.auto  = True
      self.bad_set = False

   def __call__(self, X, alpha=1.0, bytes=False):
      if self.bad_set:
         if not isinstance(X, numpy.ma.masked_array):
            X = numpy.ma.asarray(X)
         X.mask = ma.make_mask(-numpy.isfinite(X))
      return Colormap.__call__(self, X, alpha, bytes)

   def set_bad(self, color='k', alpha = 1.0):
      self.bad_set = True
      Colormap.set_bad(self, color, alpha)
      if self.auto:
         self.update()

   def set_source(self, source):
      """
      Define an alternative source for the colormap.
      *source* can be any other matplotlib colormap object or its registered
      name, or the name of a textfile
      with one RGB triplet per line. Values should be between 0.0 and 1.0.
      """
      self.source = source
      try:
         source = cm.get_cmap(source)
      except:
         pass
      if isinstance(source, Colormap):
         if not source._isinit:
            source._init()
         self.baselut = source._lut
      else:
         colors = tabarray(source)
         ncolors = colors.shape[0]
         self.baselut = numpy.ones((ncolors+3,4), numpy.float)
         self.baselut[:ncolors,:3] = colors
      self.worklut = self.baselut.copy()
      if self.name is not None:             # existing VariableColormap object?
         self.set_scale(self.scale)


   def _init(self):
      self._lut = self.worklut.copy()
      self._isinit = True
      self._set_extremes()
      
   def modify(self, slope, shift):
      """
      Apply a slope and a shift to the colormap. Defaults are 1.0 and 0.0.
      If one or more Axes objects have been registered with method
      :meth:`add_frame`, the images in them will be updated and
      the corresponding canvases will be redrawn.
      """
      if not self._isinit:
         self._init()
      self.slope = slope
      self.shift = shift
      ncolors = self.N
      lut     = self._lut
      worklut = self.worklut
      slope   = slope*self.invrt
      for i in xrange(ncolors):
         x = (float(i)/float(ncolors-1))-0.5
         y = slope*(x-shift)+0.5
         if y>1.0:
            y = 1.0
         elif y<0.0:
            y = 0.0
         m = float(ncolors-1)*y+0.5
         lut[i] = worklut[m]
      
      if self.auto:
         self.update()

   def set_inverse(self, inverse=False):
      if inverse:
         self.invrt = -1.0
      else:
         self.invrt =  1.0
      self.modify(self.slope, self.shift)

   def set_scale(self, scale='LINEAR'):
      """
      Apply a scale to this colormap. *scale* can be one of:
      'LINEAR', 'LOG', 'EXP', 'SQRT' and 'SQUARE'.
      """
      scale = scale.upper()
      ncolors = self.N
      baselut = self.baselut
      worklut = self.worklut

      if scale=='LOG':
         fac = float(ncolors-1)/math.log(ncolors)
         for i in xrange(ncolors):
            worklut[i] = baselut[fac*math.log(i+1)]

      elif scale=='EXP':
         fac = float(ncolors-1)/math.pow(10.0, (ncolors-1)/100.0 -1.0)
         for i in xrange(ncolors):
            worklut[i] = baselut[fac*math.pow(10.0, i/100.0-1.0)]

      elif scale=='SQRT':
         fac = float(ncolors-1)/math.sqrt(ncolors)
         for i in xrange(ncolors):
            worklut[i] = baselut[fac*math.sqrt(i)]
            
      elif scale=='SQUARE':
         fac = float(ncolors-1)/(ncolors*ncolors)
         for i in xrange(ncolors):
            worklut[i] = baselut[fac*i*i]

      elif scale=='LINEAR':
         worklut[:] = baselut[:]

      else:
         raise Exception, 'invalid colormap scale'
      
      self.scale = scale
      self.modify(self.slope, self.shift)

   def add_frame(self, frame):
      """
      Associate matplotlib Axes object *frame* with this colormap.
      If the colormap is subsequently modified, images in this frame will
      be updated and *frame*'s canvas will be redrawn.
      """
      self.frames.add(frame)
      canvas = frame.figure.canvas
      if canvas not in self.canvases:
         self.canvases[canvas] = 1
      else:
         self.canvases[canvas] += 1

   def remove_frame(self, frame):
      """
      Disassociate matplotlib Axes object *frame* from this colormap.
      """
      self.frames.remove(frame)
      canvas = frame.figure.canvas
      self.canvases[canvas] -= 1
      if self.canvases[canvas]==0:
         del self.canvases[canvas]

   def update(self):
      """
      Redraw all images in the Axes objects registered with method
      :meth:`add_frame`. update() is called automatically when the colormap
      changes while :attr:`auto` is True.
      """
      for frame in self.frames:
         for image in frame.get_images():
            image.changed()
      for canvas in self.canvases:
         canvas.draw()


# ==========================================================================
#                        function KeyPressFilter
# --------------------------------------------------------------------------
from matplotlib.backend_bases import FigureManagerBase

__key_press = FigureManagerBase.key_press

def KeyPressFilter(fmb_obj, event):
   if event.key in KeyPressFilter.allowed:
      __key_press(fmb_obj, event)
      
KeyPressFilter.allowed = []

FigureManagerBase.key_press = KeyPressFilter

# ==========================================================================
#                        class TimeCallback
# --------------------------------------------------------------------------
from matplotlib import rcParams

class TimeCallback(object):
   """
Objects of this class are responsible for handling timer events.  Timer
events occur periodically whenever a predefined period of time expires. 
A TimeCallback object will not be deleted as long as it
is scheduled ("active"), so it is not always necessary to keep a reference
to it.
This class is backend-dependent. Currently supported backends are GTKAgg,
GTK and Qt4Agg.

:param proc:
   the function to be called upon receiving an event of the specified
   type and occurring in the specified Axes object. It is called with one
   argument: the current TimeCallback object.
:param interval:
   the time interval in seconds.
:param schedule:
   indicates whether the object should start handling events immediately.
   Default True.
:param attr:
   keyword arguments each resulting in an attribute with the same name.  

**Attribute:**

.. attribute:: active
   
   True if callback is scheduled, False otherwise.

**Methods:**

.. automethod:: schedule
.. automethod:: deschedule
.. automethod:: set_interval

**Example:**

::

   #/usr/bin/env python

   from matplotlib import pyplot
   from kapteyn.mplutil import VariableColormap, TimeCallback
   import numpy
   from matplotlib import mlab

   def colour_cb(cb):
      slope = cb.cmap.slope
      shift = cb.cmap.shift
      if shift>0.5:
         shift = -0.5
      cb.cmap.modify(slope, shift+0.01)                   # change colormap

   figure = pyplot.figure(figsize=(8,8))
   frame = figure.add_axes([0.05, 0.05, 0.85, 0.85])

   colormap = VariableColormap('jet')
   colormap.add_frame(frame)
   TimeCallback(colour_cb, 0.1, cmap=colormap)             # change every 0.1 s

   x = y = numpy.arange(-3.0, 3.0, 0.025)
   X, Y  = numpy.meshgrid(x, y)
   Z1    = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0) # Gaussian 1
   Z2    = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)     # Gaussian 2
   Z     = Z2-Z1                                           # difference

   img = frame.imshow(Z, origin="lower", cmap=colormap)

   pyplot.show()

This code displays an image composed of 2 Gaussians and continuously modifies
its colormap's shift value between -0.5 and 0.5 in steps of 0.01.
These steps take place at 0.1 second intervals.
"""
   supported = {}
   scheduled = []

   def __new__(cls, *args, **kwds):
      backend = rcParams['backend'].upper()
      if backend in TimeCallback.supported:
         return object.__new__(TimeCallback.supported[backend], *args, **kwds)
      else:
         raise Exception, 'TimeCallback not supported for backend %s' % backend

   def __init__(self, proc, interval, schedule=True, **attr):
      self.proc     = proc
      self.interval = interval
      for name in attr.keys():
         self.__dict__[name] = attr[name]
      self.id     = 0
      self.active   = False
      if schedule: self.schedule()

   def schedule(self):
      """
      Activate the object so that it will start calling the callback function
      periodically. If the object is already active, nothing will be done.
      """
      pass

   def deschedule(self):
      """
      Deactivate the object so that it stops calling its callback function.
      If the object is already inactive, nothing will be done.
      """
      pass

   def set_interval(self, interval):
      """
      Changes the object's time interval in seconds.
      """
      self.interval = interval
      if self.active:
         self.deschedule()
         self.schedule()

# --------------------------------------------------------------------------
#                     backend-dependent implementations
# --------------------------------------------------------------------------

# GTK, GTKAgg:
#
try:
   import gobject
   
   class TimeCallback_GTK(TimeCallback):
       
      def schedule(self):
         if self.id:
            return
         milliseconds = max(1,int(round(self.interval*1000.0)))
         self.id    = gobject.timeout_add(milliseconds, self.reached)
         self.active = True
         self.scheduled.append(self)
      
      def deschedule(self):
         if not self.id:
            return
         gobject.source_remove(self.id)
         self.id = 0
         self.active = False
         self.scheduled.remove(self)

      def reached(self):
         self.proc(self)
         return True

   TimeCallback.supported['GTKAGG'] = TimeCallback_GTK
   TimeCallback.supported['GTK'] = TimeCallback_GTK
except:
   pass


# Qt4AGG:
#
try:
   from PyQt4 import QtCore
   
   class TimeCallback_QT4(TimeCallback):
       
      def schedule(self):
         if self.id:
            return
         milliseconds = max(1,int(round(self.interval*1000.0)))
         self.id   = QtCore.QTimer()
         QtCore.QObject.connect(self.id, QtCore.SIGNAL("timeout()"), self.reached)
         self.id.start(milliseconds)
         self.active = True
         self.scheduled.append(self)
      
      def deschedule(self):
         if not self.id:
            return
         self.id.deleteLater()
         self.id = 0
         self.active = False
         self.scheduled.remove(self)
         
      def reached(self):
         self.proc(self)

   TimeCallback.supported['QT4AGG'] = TimeCallback_QT4
except:
   pass
