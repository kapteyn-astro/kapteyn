#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    maputils.py
# PURPOSE: Provide methods to extract 2-dim image data from FITS file
#          if dimension of data set is 2 or higher.
#          A WCS map class allows a user to add wcs labels and 
#          wcs graticules.
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    April 17, 2008
# UPDATE:  May 17, 2008
#          June 30, 2009; Docstrings converted to Sphinx format
# VERSION: 1.11
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#
# TODO:
# -Positionmessage offsets laten laten tonen.
# -Betere melding als header niet valide is
# -Rulers met 1 (start)punt
# -Blur factors aanpassen aan pixelaspectratio
# -Insertspatial kan teveel invoegen als bv een classic header wordt
#  nagestreefd en de oude header nog een CD element heeft voor niet
#  spatiele assen.
# -Classicheader taak moet ook kleinere box in 3e richting kunnen maken
# -Bij FREQ RA subset en invoer van {} 1.4... {} 178... geen foutmelding
#  en geen grids.
# -Colorbar met optie bar tegen plot aan
# -WCSflux geschikt maken voor subplots
# -Truc voor alignment. va=bottom en een delta introduceren
#----------------------------------------------------------------------

"""
.. highlight:: python
   :linenothreshold: 10

Module maputils 
===============

In the maputils tutorial we show many examples with Python code and 
figures to illustrate the functionality and flexibility of this module.
The documentation below is restricted to the module's classes and methods.

Introduction
------------

One of the goals of the Kapteyn Package is to provide a user/programmer basic
tools to make plots (with WCS annotation) of image data from FITS files.
These tools are based on the functionality of PyFITS and Matplotlib.
The methods from these packages are modified in *maputils* for an optimal
support of inspection and presentation of astronomical image data with
easy to write and usually very short Python scripts. To illustrate
what can be done with this module, we list some steps you need
in the process to create a hard copy of an image from a FITS file:

* Open FITS file on disk or from a remote location (URL)
* Specify in which header data unit the image data is stored
* Specify the data slice for data sets with dimensions > 2
* Specify the order of the image axes
* Set the limits in pixels of both image axes
* Set the sky system in which you want to plot wcs information.

Then for the display:

* Plot the image or a mosaic of images in the correct aspect ratio
* Plot (labeled) contours
* Plot world coordinate labels along the image axes  (basic routines in :mod:`wcsgrat`)
* Plot coordinate graticules (basic routines in :mod:`wcsgrat`)
* Interactively change color map and color limits
* Read the position of features in a map and write these positions in your terminal.
* Resize your plot canvas to get an optimal layout while preserving the aspect ratio.
* Write the result to *png* or *pdf* (or another format from a list)

Of course there are many programs that can do this job some way or the other.
But most probably no program does it exactly the way you want or the program
does too much. Also many applications cannot be extended, at least not as simple
as with the building blocks in :mod:`maputils`.

Module :mod:`maputils` is also very useful as a tool to extract and plot
data slices from data sets with more than two axes. For example it can plot
so called *Position-Velocity* maps from a radio interferometer data cube
with channel maps. It can annotate these plots with the correct WCS annotation using
information about the 'missing' spatial axis.

To facilitate the input of the correct data to open a FITS image,
to specify the right data slice or to set the pixel limits for the
image axes, we implemented also some helper functions.
These functions are primitive (terminal based) but effective. You can
replace them by enhanced versions, perhaps with a graphical user interface.

Here is an example of what you can expect. We have a three dimensional data set
on disk called *ngc6946.fits* with axes RA, DEC and VELO.
The program prompts the user to enter
image properties like data limits, axes and axes order.
The image below is a data slice in RA, DEC at VELO=50.
We changed interactively the color map (keys *page-up/page-down*)
and the color limits (pressing right mouse button while moving the mouse) and saved
a hard copy on disk.

In the next code we use keyword parameter *promptfie* a number of times.
Abbreviation 'fie' stands for *Function Interactive Environment*.

.. literalinclude:: EXAMPLES/mu_introduction.py
   
.. image:: EXAMPLES/mu_introduction.png
   :width: 700
   :align: center
   
   
.. centered:: Image from FITS file with graticules and WCS labels

Module level data
-----------------

:data:`cmlist`
   Object from class Colmaplist which has attribute *colormaps* which
   is a sorted list with names of colormaps.

:Example:

   >>> from kapteyn import maputils
   >>> print maputils.cmlist.colormaps
   >>> cmap = raw_input("Enter name of a colormap: ")
   ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'Dark2',
    'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn',
    'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn',
    'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu',
    'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3',
    'Spectral', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd',
    'autumn', 'binary', 'bone', 'cool', 'copper', 'flag',
    'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar',
    'gist_rainbow', 'gist_stern', 'gist_yarg', 'gray',
    'hot', 'hsv', 'jet', 'pink', 'prism', 'spectral',
    'spring', 'summer', 'winter']

Prompt functions
----------------

.. index:: Open a FITS file
.. autofunction:: prompt_fitsfile
.. index:: Set axis numbers of FITS image
.. autofunction:: prompt_imageaxes
.. index:: Set pixel limits in FITS image
.. autofunction:: prompt_box
.. index:: Set spectral translation from list
.. autofunction:: prompt_spectrans
.. index:: Set sky system for output
.. autofunction:: prompt_skyout
.. index:: Get clip values for image data
.. autofunction:: prompt_dataminmax

Utility functions
-----------------

.. index:: Convert PyFITS header to Python dictionary
.. autofunction:: fitsheader2dict

.. index:: Calculate distance on sphere
.. autofunction:: dist_on_sphere

.. autofunction:: showall

Class FITSimage
---------------

.. index:: Extract image data from FITS file
.. autoclass:: FITSimage


Class Annotatedimage
---------------------

.. index:: Plot FITS image data with Matplotlib
.. autoclass:: Annotatedimage


Class Image
-----------

.. autoclass:: Image

Class Contours
--------------

.. autoclass:: Contours

Class Colorbar
--------------

.. autoclass:: Colorbar

Class Beam
----------

.. autoclass:: Beam

Class Skypolygon
-----------------

.. autoclass:: Skypolygon


Class Marker
------------

.. autoclass:: Marker


Class Pixellabels
-----------------

.. autoclass:: Pixellabels

Class Colmaplist
----------------

.. autoclass:: Colmaplist


Class FITSaxis
--------------

.. autoclass:: FITSaxis

Class Positionmessage
---------------------

.. autoclass:: Positionmessage

Class MovieContainer
--------------------

.. autoclass:: MovieContainer

"""
# ----------------- Use for experiments -----------------
"""
# Use this to change the default backend
#from matplotlib import use
use('qt4agg')

from matplotlib import __version__ as mplversion
print "Matplotlib version:", mplversion
# Experiment with a local LaTeX e.g. to improve horizontal label alignment
#from matplotlib import rc
#rc('text', usetex=True)
"""
# Use this to find the current backend. We need this parameter to find
# out whether we work with a QT canvas or not. For a QT canvas we deal with
# toolbar messages in a different way.
from matplotlib import rcParams
backend = rcParams['backend'].upper()

# !!!!!!!!!!!!
# Uitproberen met nieuwe MPL versies. Het wijzigt de default
# file format in savefig:
#rcParams['savefig.extension'] = 'pdf'
#rcParams['savefig.format'] = 'pdf'


from matplotlib.pyplot import setp as plt_setp,  get_current_fig_manager as plt_get_current_fig_manager
from matplotlib.pyplot import figure, show
from matplotlib import cm
from matplotlib.colors import Colormap, Normalize          #, LogNorm, NoNorm
from matplotlib.colorbar import make_axes, Colorbar, ColorbarBase
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator, ScalarFormatter, FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.image import AxesImage
from matplotlib.cbook import report_memory
import matplotlib.axes as axesclass  
from mpl_toolkits.axes_grid1 import make_axes_locatable # VOGNEW
import matplotlib.nxutils as nxutils
import pyfits
import numpy
from kapteyn import wcs, wcsgrat
from kapteyn.celestial import skyrefsystems, epochs, skyparser, lon2hms, lat2dms, lon2dms
from kapteyn.tabarray import tabarray, readColumns
#from kapteyn.mplutil import AxesCallback, CanvasCallback, VariableColormap, TimeCallback, KeyPressFilter
from mplutil import AxesCallback, CanvasCallback, VariableColormap, TimeCallback, KeyPressFilter
from kapteyn.positions import str2pos, mysplit, unitfactor
from kapteyn.interpolation import map_coordinates  # original from scipy.ndimage.interpolation
from kapteyn.filters import gaussian_filter
# Original location was: from scipy.ndimage.filters import gaussian_filter
from kapteyn import rulers
import readline
from gc import collect as garbagecollect
from types import TupleType as types_TupleType
from types import ListType as types_ListType
from types import StringType as types_StringType 
from string import upper as string_upper
from string import letters, join
from random import choice
from re import split as re_split
from datetime import datetime
import warnings
import time
from sys import stdout
from os import getpid as os_getpid
from os.path import basename as os_basename
from subprocess import Popen, PIPE
from platform import system as os_system

try:
   from gipsy import anyout, typecli
   gipsymod = True
except:
   gipsymod = False


KeyPressFilter.allowed = ['f', 'g']

__version__ = '1.11'

(left,bottom,right,top) = (wcsgrat.left, wcsgrat.bottom, wcsgrat.right, wcsgrat.top)                 # Names of the four plot axes
(native, notnative, bothticks, noticks) = (wcsgrat.native, wcsgrat.notnative, wcsgrat.bothticks, wcsgrat.noticks) 


# Each object of class Annotatedimage is stored in a list.
# This list is used by function 'showall()' to plot
# all the objects in each Annotatedimage object.
annotatedimage_list = []

# For the administration of toolbars
globalfigmanager = None
globalmessenger = None



# Redefine some methods to intercept actions which causes a canvas.draw() call
# and a change in the Axes bbox.

from matplotlib.backend_bases import NavigationToolbar2
NavigationToolbar2.ext_callback = None
NavigationToolbar2.ext_callback2 = None
def _update_view(self):
    '''update the viewlim and position from the view and
    position stack for each axes
    '''
    lims = self._views()
    if lims is None:  return
    pos = self._positions()
    if pos is None: return
    for i, a in enumerate(self.canvas.figure.get_axes()):
        xmin, xmax, ymin, ymax = lims[i]
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        # Restore both the original and modified positions
        a.set_position( pos[i][0], 'original' )
        a.set_position( pos[i][1], 'active' )
    
    self.draw()
    if self.ext_callback2:
       self.ext_callback2()
    if self.ext_callback:
       self.ext_callback()

def release(self, event):
    if self.ext_callback2:
       self.ext_callback2()
    if self.ext_callback:
       self.ext_callback()


def drag_pan(self, event):
    'the drag callback in pan/zoom mode'

    for a, ind in self._xypress:
        #safer to use the recorded button at the press than current button:
        #multiple button can get pressed during motion...
        a.drag_pan(self._button_pressed, event.key, event.x, event.y)
    self.dynamic_update()

    if self.ext_callback:
       self.ext_callback()


NavigationToolbar2._update_view = _update_view
NavigationToolbar2.release = release
NavigationToolbar2.drag_pan = drag_pan


from matplotlib.figure import Figure
Figure.ext_callback = None
def subplots_adjust(self, *args, **kwargs):
    """
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
        wspace=None, hspace=None)

    Update the :class:`SubplotParams` with *kwargs* (defaulting to rc where
    None) and update the subplot locations

    """
    self.subplotpars.update(*args, **kwargs)
    import matplotlib.axes
    for ax in self.axes:
        if not isinstance(ax, matplotlib.axes.SubplotBase):
            # Check if sharing a subplots axis
            if ax._sharex is not None and isinstance(ax._sharex, matplotlib.axes.SubplotBase):
                ax._sharex.update_params()
                ax.set_position(ax._sharex.figbox)
            elif ax._sharey is not None and isinstance(ax._sharey, matplotlib.axes.SubplotBase):
                ax._sharey.update_params()
                ax.set_position(ax._sharey.figbox)
        else:
            ax.update_params()
            ax.set_position(ax.figbox)
    if self.ext_callback:
       self.ext_callback()
       
Figure.subplots_adjust = subplots_adjust


def getmemory():
#-------------------------------------------------------------------------------
# Use 'ps' on Linux to get the resident memory in use by the program.
# For other os, use cbook.report_memory() to get the memory pages.
# The Linux version is based on the report_memory() but with the # rss,sz
# options switched, because we want the residential memory here.
#-------------------------------------------------------------------------------
   pid = os_getpid()
   system = os_system()
   garbagecollect()
   if system.lower().startswith('linux'):
      a = Popen('ps -p %d -o sz,rss' % pid, shell=True,
                 stdout=PIPE).stdout.readlines()
      mem = int(a[1].split()[1])                     # Res. memory in kB
      mem /= 1024                                    # in mB
      mem = "Resident mem.: <b>" + str(mem) + "</b> mB "
   else:
      mem = report_memory()
      mem = "Mem. pages: " + str(mem)
   return mem
   

def nint(x):
#-------------------------------------------------------------------------------
# GIPSY compatible nearest integer
#-------------------------------------------------------------------------------
   return numpy.floor(x+0.5)

         
def flushprint(s):
#-------------------------------------------------------------------------------
# Helper function to write debug information to the terminal, and because
# we flush stdout, it will be printed immediately.
#-------------------------------------------------------------------------------
   return  # TODO if no debugging is required
   #print s
   #stdout.flush()


def showall():
   #--------------------------------------------------------------------
   """
   Usually in a script with only one object of class
   :class:`maputils.Annotatedimage` one plots this object,
   and its derived objects, with method :meth:`maputils.Annotatedimage.plot`.
   Matplotlib must be instructed to do the real plotting with
   pyplot's function *show()*. This function does this all.

   :Examples:

   
      >>> im1 = f1.Annotatedimage(frame1)
      >>> im1.Image()
      >>> im1.Graticule()
      >>> im2 = f2.Annotatedimage(frame2)
      >>> im2.Image()
      >>> maputils.showall()
      
   """
   #--------------------------------------------------------------------
   for ai in annotatedimage_list:
      ai.plot()
   show()     # For Matplotlib


def issequence(obj):
   return isinstance(obj, (list, tuple, numpy.ndarray))


def getfilename(pre='mu', post='fits'):
   # Create filename unique to the microsecond (%f), using
   # 'pre' as prefix and 'post' as filename extension.
   # Parameter 'post' should not contain a dot.
   d = datetime.now()
   stamp = d.strftime("%Y%m%d_%Hh%Mm%Ss")
   filename = "%s%s%d.%s" %(pre, stamp, d.microsecond, post)
   return filename


def randomlabel(base=''):
   # Generate random label (e.q. to distinguish frames with labels)
   chars = letters
   label = base
   for i in range(8):
        label = label + choice(chars)
   return label


def get_splitlon(projection):
   #--------------------------------------------------------------------
   """
   Assumed is a two dimensional map with two spatial axes.
   The projection is extracted from the ctype of the first axis.
   Then a decision is made whether we have a border in an all sky plot in
   longitude or not.
   """
   #--------------------------------------------------------------------
   if projection.category not in ['undefined', 'zenithal']:
      # then possible border problem for polygons
      splitlon = projection.crval[0] + 180.0
      if splitlon >= 360.0:
         splitlon -= 360.0
      if splitlon <= -360.0:
         splitlon += 360.0
   else:
      splitlon = None
   return splitlon


def getnumbers(prompt):
   #--------------------------------------------------------------------
   """
   Given a series of expressions all representing numbers, return a list
   with the evaluated numbers. An expression that could not be evaluated
   is skipped without a warning. Mathematical functions can be used
   from NumPy and should be entered e.g. as *numpy.sin(numpy.pi)*
   """
   #--------------------------------------------------------------------
   xstr = raw_input(prompt)
   xsplit = mysplit(xstr)
   X = []
   for x in xsplit:
      try:
         xlist = eval(x)
         if issequence(xlist):
            for xx in xlist:
               X.append(xx)
         else:
            X.append(eval(x))
      except:
         pass
   return X


def getscale(hdr):
   # Get relevant scaling keywords from this header
   # Make sure you use this function before assigning
   # the data from the header to a variable.
   bscale = None
   bzero = None
   blank = None
   bitpix = None
   if hdr.has_key('BITPIX'):
      bitpix = hdr['BITPIX']   # Exist always for PyFITS header, but not for dict.
   if hdr.has_key('BSCALE'):
      bscale = hdr['BSCALE'] 
   if hdr.has_key('BZERO'):
      bzero = hdr['BZERO']
   if hdr.has_key('BLANK'):
      blank = hdr['BLANK']
   return bitpix, bzero, bscale, blank


def dist_on_sphere(l1, b1, l2, b2):
#-----------------------------------------------------------
   """
Formula for distance on sphere accurate over entire sphere
(Vincenty, Thaddeus, 1975). Input and output are in degrees.

:param l1:
   Longitude of first location on sphere
:type l1:
   float
:param b1:
   Latitude of first location on sphere
:type b1:
   float
:param l2:
   Longitude of second location on sphere
:type l2:
   float
:param b2:
   Latitude of second location on sphere
:type b2:
   float

:Examples:

   >>> from kapteyn.maputils import dist_on_sphere
   >>> print dist_on_sphere(0,0, 20,0)
       20.0

   >>> print dist_on_sphere(0,30, 20,30)
       17.2983302106

   """
#-----------------------------------------------------------
   fac = numpy.pi / 180.0
   l1 *= fac; b1 *= fac; l2 *= fac; b2 *= fac
   dlon = l2 - l1
   aa1 = numpy.cos(b2)*numpy.sin(dlon)
   aa2 = numpy.cos(b1)*numpy.sin(b2) - numpy.sin(b1)*numpy.cos(b2)*numpy.cos(dlon)
   a = numpy.sqrt(aa1*aa1+aa2*aa2)
   b = numpy.sin(b1)*numpy.sin(b2) + numpy.cos(b1)*numpy.cos(b2)*numpy.cos(dlon)
   d = numpy.arctan2(a,b)
   return d*180.0/numpy.pi


def dispcoord(longitude, latitude, disp, direction, angle):
   #--------------------------------------------------------------------
   """
   INPUT:   longitude: numpy array, enter in degrees.
            latitude:  numpy array, enter in degrees.
            disp:      the displacement in the sky entered
                       in degrees. The value can also be
                       negative to indicate the opposite
                       direction
            angle:     the angle wrt. a great circle of
                       constant declination entered in
                       degrees.
            direction: If the longitude increases in the -X
                       direction (e.q. RA-DEC) then direction
                       is -1. else direction = +1

   Assume a triangle on a sphere with side b(=disp) connec-
   ting two positions along a great circle and sides 90-d1,
   and 90-d2 (d1, d2 are the declinations of the input and
   output positions) that connect the input and output
   position to the pole P of the sphere. Then the distance
   between the two points Q1=(a1,d1) and Q2=(a2,d2) is:
   cos(b)=cos(90-d1)cos(90-d2)+sin(90-d1)sin(90-d2)cos(a2-a1)
   Q2 is situated to the left of Q1.
   If the angle PQ1Q2 is alpha then we have another cosine
   rule:
   cos(90-d2) = cos(b)cos(90-d1)+sin(b)sin(90-d1)cos(alpha)
   or:
   sin(d2) = cos(b)sin(d1)+sin(b)cos(d1)cos(alpha)
   which gives d2. Angle Q1PQ2 is equal to a2-a1. For this
   angle we have the sine formula:
   sin(b)/sin(a2-a1) = sin(90-d2)/sin(alpha) so that:
   sin(a2-a1) = sin(b)sin(alpha)/cos(d2).
   b,alpha and d2 are known -> a2.
   """
   #--------------------------------------------------------------------
   Pi = numpy.pi
   b = abs(disp*Pi/180.0)
   a1 = longitude * Pi/180.0
   d1 = latitude * Pi/180.0
   alpha = angle * Pi/180.0
   d2 = numpy.arcsin( numpy.cos(b)*numpy.sin(d1)+numpy.cos(d1)*numpy.sin(b)*numpy.cos(alpha) )
   cosa2a1 = (numpy.cos(b) - numpy.sin(d1)*numpy.sin(d2))/(numpy.cos(d1)*numpy.cos(d2))
   sina2a1 = numpy.sin(b)*numpy.sin(alpha)/numpy.cos(d2)
   dH =  numpy.arctan2(direction*sina2a1, cosa2a1)

   a2 = a1 - dH
   lonout = a2*180.0/Pi
   latout = d2*180.0/Pi

   return lonout, latout


def split_polygons(plons, plats, splitlon):
#-----------------------------------------------------------
   """
   Given a set coordinates which form a polygon,
   we need to return a list of polygons. The polygons
   that should be distinghuised are those on one side
   of a split meridian (splitlon) and those on the other side.

   We need this if polygons are plotted near the edges
   (in longitudes) of allsky plots with cylindrical
   (at least: non-zenithal) projections.
   Then we need to split up the polygon. But this polygon
   can be complicated. For example it is possible that
   the polygon must be splitted in more than two parts.

   This algorith works only for those polygons that, when
   sliced by 'splitlon' fall apart in two pieces.

   Returns:

   lon1, lat1, lon2, lat2
   The longitudes and latitudes of first and second object.
   The lists could be empty if the polygon is contained in the
   range splitlon+epsilon to splitlon-epsilon

   Intersections:
   
   Line through: (a1,d1), (a2,d2). At which d3 is a3?
   lambda = (a3-a1)/(a2-a1). Then when a = a3 -->
   d3 = d1 + lambda *(d2-d1) 
   """
#-----------------------------------------------------------
   epsilon = 0.00000001
   # Find intersections with 'splitlon' meridian
   ilons = []; ilats = []
   prevlon = None; prevlat = None
   lons = list(plons); lons.append(lons[0])
   lats = list(plats); lats.append(lats[0])
   lon1 = []; lat1 = []
   lon2 = []; lat2 = []
   right2left = False
   left2right = False
   lastlat = None
   changedborder = 0
   intersectlon = [None, None]
   intersectlat = [None, None]
   for lon, lat in zip(lons, lats):
      if not prevlon is None:
         crossed = False
         if prevlon < splitlon:
            lon1.append(prevlon); lat1.append(prevlat)
            eps = -epsilon
            if lon > splitlon:
               crossed = True;
         else:
            lon2.append(prevlon); lat2.append(prevlat)
            eps = epsilon
            if lon <= splitlon:
               # Intersection from left to right
               crossed = True
         if crossed:
            # Intersection from right to left
            if prevlon == lon:
               d3 = prevlat
            else:
               Lambda = (splitlon-prevlon)/(lon-prevlon)
               d3 = prevlat + Lambda*(lat-prevlat)
               intersectlon[changedborder] = splitlon + eps  # i.e. +/-epsilon
               intersectlat[changedborder] = d3
               changedborder += 1
         # Now we have to close the separate parts of the polygon
         if changedborder == 2:
            dlats = numpy.linspace(0,1,20)
            for dlat in dlats:
                if eps > 0.0:
                   ds = intersectlat[0]; de = intersectlat[1]
                else:
                   ds = intersectlat[1]; de = intersectlat[0]
                d = ds + dlat*(de-ds)
                lon1.append(splitlon-epsilon)
                lat1.append(d)
                d = de + dlat*(ds-de)
                lon2.append(splitlon+epsilon)
                lat2.append(d)
            changedborder = 0    # Reset
      prevlon = lon; prevlat = lat
   return lon1, lat1, lon2, lat2



def change_header(hdr, **kwargs):
#-----------------------------------------------------------
   """
   Given header 'hdr', change/add or delete all keys
   given in **kwargs. Input 'hdr' is either a Python
   dictionary or a PyFITS header object.
   """
#-----------------------------------------------------------
   if type(hdr) == 'dict':
      dicttype = True
   else:
      dicttype = False

   # We need to loop over all keys because we distinguish
   # deleting/adding and changing of entries which cannot
   # be combined into 1 action.
   for name in kwargs.keys():
     value = kwargs[name]
     fitsname = name.upper()
     if value is None:
        try:
           del hdr[fitsname]
        except KeyError:
           pass
     else:
        if dicttype:
           hdr[fitsname] = value
        else:
           hdr.update(fitsname, value)
    # Nothing to return. Only contents of 'hdr' is changed


def colornavigation_info():
   #-----------------------------------------------------------------
   """
   This function compiles and returns a help text for
   color map interaction.
   """
   #-----------------------------------------------------------------
   helptext  = "pgUp and pgDown: browse through colour maps -- MB right: Change slope and offset\n"
   helptext += "Colour scales: 0=reset 1=linear 2=logarithmic"
   helptext += "3=exponential 4=square-root 5=square 9=inverse\n"
   helptext += "h: Toggle histogram equalization & raw image -- "
   helptext += "z: Toggle smooth & raw image -- x: Increase smooth factor\n"
   helptext += "m: Save current colour map to disk -- "
   helptext += "b: Change colour of bad pixels -- "
   helptext += "Shift MB left: Write pos. to term." # Last line has no line feed (bottom aligned)
   return helptext


class Positionmessage(object):
#-----------------------------------------------------------
   """
   This class creates an object with attributes that are needed to
   set a proper message with information about a position
   in a map and its corresponding image value.
   The world coordinates are calculated in the sky system
   of the image. This system could have been changed by the user.

   The input parameters are usually set after initialization of
   an object from class :class:`Annotatedimage`.
   For users/programmers the atributes are more important.
   With the attributes of objects of this class we can change
   the format of the numbers in the informative message.

   Note that the methods of this class return separate strings
   for the pixel coordinates, the world coordinates and the image
   values. The final string is composed in the calling environment.
   
   :param skysys:
      The sky definition of the current image
   :type skysys:
      A single parameter or tuple with integers or string
   :param skyout:
      The sky definition of the current image as defined by
      a user/programmer
   :type skyout:
      A single parameter or tuple with integers or string
   :param skysys:
      The sky definition of the current image
   :type axtype:
      tuple with strings

   :Attributes:

    .. attribute:: pixfmt

          Python number format to set formatting of pixel
          coordinates in position message in toolbar.

    .. attribute:: wcsfmt

          Python number format to set formatting of world
          coordinates in position message in toolbar.
          If the map has a valid sky system then the
          values will be formatted in hms/dms, unless
          attribute *hmsdms* is set to *False*.

    .. attribute:: zfmt

          Python number format to set formatting of image
          value(s) in position message in toolbar.

    .. attribute:: hmsdms

          If True, spatial coordinates are formatted in hms/dms.

    .. attribute:: dmsprec

          Precision in (dms) seconds if coordinate is
          formatted in dms. The precision in seconds of a
          longitude axis in an equatorial system is
          automatically copied from this number and increased
          with 1.
   """
#-----------------------------------------------------------
   def __init__(self, skysys, skyout, axtype):
      self.pixfmt = "%.1f"
      self.wcsfmt = "%.3g"     # If any is given, this will overrule hms/dms formatting
      self.wcsuffmt = "%.7f"
      self.zfmt = "%.3e"
      self.hmsdms = True
      self.dmsprec = 1
      sys = None
      if skyout is None:
         if not skysys is None:
            sys, ref, equinox, epoch = skyparser(skysys)
      else:
         sys, ref, equinox, epoch = skyparser(skyout)
      self.sys = sys
      self.axtype = axtype

   def z2str(self, z):
      # Image value(s) to string
      if self.zfmt is None:
         return None
      if issequence(z):
         if len(z) != 3:
            raise Exception, "z2str: Toolbar Message expects 1 or 3 image values"
         else:
            s = ''
            for number in z:
               if numpy.isnan(number):
                  s += 'NaN'
               else:
                  s += self.zfmt % number + ' '
            return s.rstrip()
      else:
         if numpy.isnan(z):
            s = 'NaN'
         else:
            s = self.zfmt % z
         return s

   def pix2str(self, x, y):
      # Pixel coordinates to string
      if self.pixfmt is None:
         return None
      s = self.pixfmt%x + ' ' + self.pixfmt%y
      return s

   def wcs2str(self, xw, yw, missingspatial, returnlist=False, unformatted=False):
      # World coordinates to string
      if not self.wcsfmt and not unformatted:
         return None
      if unformatted and self.wcsuffmt is None:
         return None
      vals = (xw, yw)
      if not missingspatial is None:
         vals += (missingspatial,)
      if returnlist:
         s = []
      else:
         s = ''
      if unformatted:
         wcsfmt = self.wcsuffmt
      else:
         wcsfmt = self.wcsfmt
      for atyp, val in zip(self.axtype, vals):
         if val is None:
            coord = "NaN"
         else:
            if self.hmsdms and not unformatted:
               if atyp == 'longitude':
                  if self.sys == wcs.equatorial:
                     coord = lon2hms(val, prec=self.dmsprec+1) # One extra digit for hms
                  else:
                     coord = lon2dms(val, prec=self.dmsprec)
               elif atyp == 'latitude':
                     coord = lat2dms(val, prec=self.dmsprec)
               else:                  
                  coord = wcsfmt%val
            else:
               coord = wcsfmt%val
         if returnlist:
            s.append(coord)
         else:
            s += coord + ' '
      return s


class Colmaplist(object):
#-----------------------------------------------------------
   """
   This class provides an object which stores
   the names of all available colormaps.
   The method *add()* adds external colormaps to
   this list. The class is used in the context of
   other classes but its attribute *colormaps*
   can be useful.

   .. attribute:: colormaps

      List with names of colormaps as used in combination
      with keyword parameter *cmap* in the constructor
      of :class:`Annotatedimage`.
      
   """
#-----------------------------------------------------------
   def compare(self, a, b): return cmp(a.lower(), b.lower())  # For case insensitive sorting
   def __init__(self):
      # A list with available Matplotlib color maps
      # The '_r' entries are reversed versions. We omit these versions
      # because maputils has an 'inverse' option to generate an inverted version
      # of each color lut.
      self.colormaps = [m for m in cm.datad.keys() if not m.endswith("_r")]
      # Sort this list in a case insensitive way
      self.colormaps.sort(self.compare)
      colormaps = self.colormaps
   def add(self, clist):
      if not issequence(clist):
         clist = [clist]
      # Just ad and sort afterwards:
      self.colormaps += clist
      #for c in clist[::-1]:
      #   self.colormaps.insert(0,c)
      self.colormaps.sort(self.compare)
      colormaps = self.colormaps
   def addfavorites(self, flist):
      # Prepend a list with (unsorted) favorites
      if not issequence(flist):
         flist = [flist]
      newlist = flist + self.colormaps
      self.colormaps = flist + self.colormaps
      colormaps = self.colormaps

cmlist = Colmaplist()
cmlist.add(VariableColormap.luts())      # Add luts from lut directory of the Kapteyn Package
#colormaps = cmlist.colormaps
# Add other attributes like a list with scales and the defaults
# For the default color map, we cannot set an index because we
# never know what color maps are added and how they are sorted.
# So we store only the name, and when necessary we retrieve an index
# with list method index()
cmlist.cmap_default = 'jet'

# TODO: Maak ook bovenstaande default een class var.



def fitsheader2dict(header, comment=True, history=True):
#-----------------------------------------------------------
   """
Transform a FITS header, read with PyFITS into a Python
dictionary.
This is useful if one wants to iterate over all keys in the
header. The PyFITS header is not iterable.
   """
#-----------------------------------------------------------
   class fitsdict(dict):
      pass

   result = fitsdict()
   result.comment = {}

   if isinstance(header, dict):
      for key in header:
         result[key] = header[key]
         try:
            result.comment[key] = header.comment[key]
         except:
            result.comment[key] = ''
   else:
      for card in header.ascard:
         try:
            key = card.key
            if (history and key=='HISTORY') or (comment and key=='COMMENT'):
               try:
                  result[key].append(card.value)
               except KeyError:
                  result[key] = [card.value]
            else:
               result[key] = header[key]
               result.comment[key] = card.comment
         except:
            card.verify()
   
   return result


def prompt_box(pxlim, pylim, axnameX, axnameY):
#-----------------------------------------------------------
   """
External helper function which returns the
limits in pixels of the x- and y-axis.
The input syntax is: xlo,xhi, ylo,yhi. For *x* and *y*
the names of the image axes are substituted.
Numbers can be separated by comma's and or
spaces. A number can also be specified
with an expression e.g. ``0, 10,  10/3, 100*numpy.pi``.
All these numbers are converted to integers.

:param pxlim:
   Sequence of two numbers representing limits
   in pixels along the x axis as defined in the FITS file.
:type pxlim:
   tuple with two integers
:param pylim:
   Sequence of two numbers representing limits
   in pixels along the y axis as defined in the FITS file.
:type pylim:
   tuple with two integers
:param axnameX:
   Name of image X-axis
:type axnameX:
   String
:param axnameY:
   Name of image Y-axis
:type axnameY:
   String

:Prompts:
   *Enter pixel limits in Xlo,Xhi,  Ylo,Yhi ..... [xlo,xhi, ylo,yhi]:*

   The default should be the axis limits as defined in the FITS header
   in keywords *NAXISn*.
   In a real case this could look like:
   
   *Enter pixel limits in RAlo,RAhi,  DEClo,DEChi ..... [1, 100, 1, 100]:*


:Returns:
   Tuple with two elements
   pxlim, pylim (see parameter description)

:Notes:
   This function does not check if the limits are
   within the index range of the (FITS)image.
   This check is done in the :meth:`FITSimage.set_limits` method
   of the :class:`FITSimage` class.

:Examples:
   Use of this function as prompt function in the
   :meth:`FITSimage.set_limits` method::
   
      fitsobject = maputils.FITSimage('rense.fits')
      fitsobject.set_imageaxes(1,2, slicepos=30) # Define image in cube
      fitsobject.set_limits(promptfie=maputils.prompt_box)
   
   This 'box' prompt needs four numbers. The first is the range in 
   x and the second is the range in y. The input are pixel coordinates,
   e.g.::
   
       >>>  0, 10   10/3, 100*numpy.pi

   Note the mixed use of spaces and comma's to
   separate the numbers. Note also the use of
   NumPy for mathematical functions. The numbers are truncated to integers.
   """
#-----------------------------------------------------------
   boxtxt = "%slo,%shi,  %slo,%shi" %(axnameX, axnameX, axnameY, axnameY)
   while True:
      try:
         s = "Enter pixel limits in %s ..... [%d, %d, %d, %d]: " % (boxtxt, pxlim[0], pxlim[1], pylim[0], pylim[1])
         box = raw_input(s)
         if box != "":
            lims = re_split('[, ]+', box.strip())
            xlo = int(eval(lims[0]))
            xhi = int(eval(lims[1]))
            ylo = int(eval(lims[2]))
            yhi = int(eval(lims[3]))
            readline.add_history(box)    # Facilitate input from terminal
            ok = (pxlim[0] <= xlo <= pxlim[1]) and \
                 (pxlim[0] <= xhi <= pxlim[1]) and \
                 (pylim[0] <= ylo <= pylim[1]) and \
                 (pylim[0] <= ylo <= pylim[1])
            if not ok:
               print "One of the values is outside the pixel limits"
            else:
               break
         else:
            return pxlim, pylim
      except KeyboardInterrupt:                  # Allow user to press ctrl-c to abort program
        raise
      except:
         print "Wrong box"

   return (xlo, xhi), (ylo, yhi)



def prompt_fitsfile(defaultfile=None, prompt=True, hnr=None, alter=None, memmap=None):
#-----------------------------------------------------------------
   """
An external helper function for the FITSimage class to
prompt a user to open the right Header and Data Unit (hdu)
of a FITS file.
A programmer can supply his/her own function
of which the return value is a sequence containing 
the hdu list, the header unit number, the filename and a character for 
the alternate header.
   
:param defaultfile:
   Name of FITS file on disk or url of FITS file on the internet.
   The syntax follows the standard described in the PyFITS documentation.
   See also the examples.
:type defaultfile:
   String
:param prompt:
   If False and a default file exists, then do not prompt for a file name.
   Open file and start checking HDU's
:type prompt:
   Boolean
:param hnr:
   The number of the FITS header that you want to use.
   This function lists the hdu information and when
   hnr is not given, you will be prompted.
:type hnr:
   Integer
:param alter:
   Selects an alternate header. Default is the standard header.
   Keywords in alternate headers end on a character A..Z
:type alter:
   Empty or a single character. Input is case insensitive.
:param memmap:
   Set PyFITS memory mapping on/off. Let PyFITS set the default.
:type memmap:
   Boolean


:Prompts:
   1. *Enter name of fits file ...... [a default]:*

      Enter name of file on disk of valid url.
   2. *Enter number of Header Data Unit ...... [0]:*

      If a FITS file has more than one HDU, one must decide
      which HDU contains the required image data.
      
:Returns:

   * *hdulist* - The HDU list and the user selected index of the wanted 
     hdu from that list. The HDU list is returned so that it
     can be closed in the calling environment.
   * *hnr* - FITS header number. Usually the first header, i.e. *hnr=0*
   * *fitsname* - Name of the FITS file.
   * *alter* - A character that corresponds to an alternate header
     (with alternate WCS information e.g. a spectral translation).
     
:Notes:
   --
   
:Examples:  
   Besides file names of files on disk, PyFITS allows url's and gzipped 
   files to retrieve FITS files e.g.::
   
      http://www.atnf.csiro.au/people/mcalabre/data/WCS/1904-66_ZPN.fits.gz
      
   """
#--------------------------------------------------------------------
   while True:
      try:
         if defaultfile is None:
            filename = ''
            s = "Enter name of FITS file: "
         else:
            filename = defaultfile
            s = "Enter name of FITS file ...... [%s]: " % filename   # PyFits syntax
         if defaultfile is None or prompt:
            fn = raw_input(s)
         else:
            fn = filename
         if fn != '':
            filename = fn
         # print "Filename memmap", filename, memmap
         hdulist = pyfits.open(filename, memmap=memmap)
         break
      except IOError, (errno, strerror):
         print "I/O error(%s): %s opening [%s]" % (errno, strerror, filename)
         prompt = True    # Also prompt when a default file name was entered
      except KeyboardInterrupt:
         raise
      except:
         defaultfile = None
         print "Cannot open file, unknown error."
         con = raw_input("Abort? ........... [Y]/N:")
         if con == '' or con.upper() == 'Y':
            raise Exception, "Loop aborted by user"

   hdulist.info()
   # Note that an element of this list can be accessed either
   # by integer number or by name of the extension.
   if hnr is None:
      n = len(hdulist)
      if  n > 1:
         while True:
            p = raw_input("Enter number of Header Data Unit ... [0]:")
            if p == '':
               hnr = 0
               break
            else:
               try:
                  p = int(p)
               except:
                  pass
               try:
                  k = hdulist[p]
                  hnr = p
                  break
               except:
                  pass
      else:
         hnr = 0
   # If there is no character given for an alternate header
   # but an alternate header is detected, then the user is
   # prompted to enter a character from a list with allowed
   # characters. Currently an alternate header is found if
   # there is a CRPIX1 followed by a character A..Z
   if alter == '':
      alternates = []
      hdr = hdulist[hnr].header
      for a in letters[:26]:
         k = "CRPIX1%c" % a.upper()  # To be sure that it is uppercase
         if hdr.has_key(k):
            print "Found alternate header:", a.upper()
            alternates.append(a)
   
      #alter = ''
      if len(alternates):
         while True:
            p = raw_input("Enter char. for alternate header ... [No alt. header]:")
            if p == '':
               alter = ''
               break
            else:
               if p.upper() in alternates:
                  alter = p.upper()
                  break
               else:
                  print "Character not in list with allowed alternates!"

   return hdulist, hnr, filename, alter



def prompt_imageaxes(fitsobj, axnum1=None, axnum2=None, slicepos=None):
#-----------------------------------------------------------------------
   """
Helper function for FITSimage class. It is a function that requires
interaction with a user. Therefore we left it out of any class
definition. so that it can be replaced by any other function that
returns the position of the data slice in a FITS file.

It prompts the user
for the names of the axes of the wanted image. For a
2D FITS data set there is nothing to ask, but for
dimensions > 2, we should prompt the user to enter two
image axes. Then also a list with pixel positions should
be returned. These positions set the position of the data
slice on the axes that do not belong to the image.
Only with this information the right slice can be extracted.

The user is prompted in a loop until a correct input is given.
If a spectral axis is part of the selected image then
a second prompt is prepared for the input of the required spectral
translation.

:param fitsobj:
   An object from class FITSimage. This prompt function derives useful
   attributes from this object such as the allowed spectral
   translations.
:type fitsobj:
   Instance of class FITSimage
:param axnum1:
   The axis number of the first (horizontal in terms of plot software)
   axis of the selected image which should be used as the default
   in the prompt. If *None* then the default is set to 1.
:type axnum1:
   Integer [1, NAXIS]
:param axnum2:
   The axis number of the first (horizontal in terms of plot software)
   axis of the selected image which should be used as the default
   in the prompt. If *None* then the default is set to 1.
   If both *axnum1* and *axnum2* are specified then the image
   axis input prompt is skipped.
:type axnum2:
   Integer [1, NAXIS]

:Prompts:
      Name of the image axes:
        *Enter 2 axes from (list with allowed axis names) .... [default]:*

        e.g.: ``Enter 2 axes from (RA,DEC,VELO) .... [RA,DEC]:``

        The axis names can be abbreviated. A minimal match is applied.


   
:Returns:
   Tuple with three elements:
   
   * *axnum1*:
     Axis number of first image axis. Default or entered by a user.
   * *axnum2*:
     Axis number of second image axis. Default or entered by a user.
   * *slicepos*:
     A list with pixel positions. One pixel for each axis
     outside the image in the same order as the axes in the FITS
     header. These pixel positions are necessary
     to extract the right 2D data from FITS data with dimensions > 2.

:Example:
   Interactively set the axes of an image using a prompt function::
   
      # Create a maputils FITSimage object from a FITS file on disk
      fitsobject = maputils.FITSimage('rense.fits')
      fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
   
   """
#-----------------------------------------------------------------------
   n = fitsobj.naxis
   if (axnum1 is None or axnum2 is None):
      a1 = axnum1; a2 = axnum2
      ax1 = a1; ax2 = a2
      unvalid = True
      axnamelist = '('
      for i in range(1,n+1):
         axnamelist += fitsobj.axisinfo[i].axname
         if i < n:
            axnamelist += ','
      axnamelist += ')'
      while unvalid:
         if axnum1 is None:
            a1 = 1
            ax1 = -1
         if axnum2 is None:
            a2 = 2
            ax2 = -1
         deflt = "%s,%s" % (fitsobj.axisinfo[a1].axname, fitsobj.axisinfo[a2].axname)
         mes = "Enter 2 axes from %s .... [%s]: " % (axnamelist, deflt)
         str1 = raw_input(mes)
         if str1 == '':
            str1 = deflt
         axes = re_split('[, ]+', str1.strip())
         if len(axes) == 2:
            for i in range(n):
               ax = i + 1
               str2 = fitsobj.axisinfo[ax].axname
               if str2.find(string_upper(axes[0]), 0, len(axes[0])) > -1:
                  ax1 = ax
               if str2.find(string_upper(axes[1]), 0, len(axes[1])) > -1:
                  ax2 = ax
            unvalid = not (ax1 >= 1 and ax1 <= n and ax2 >= 1 and ax2 <= n and ax1 != ax2)
            if unvalid:
               # No exceptions because we are in a loop
               print "Incorrect input of image axes!"
            if (ax1 == ax2 and ax1 != -1):
               print "axis 1 == axis 2"
         else:
            print "Number of images axes must be 2. You entered %d" % (len(axes),)
      print  "You selected: ", fitsobj.axisinfo[ax1].axname, fitsobj.axisinfo[ax2].axname
      axnum1 = ax1; axnum2 = ax2
   axperm = [axnum1, axnum2]


   # Retrieve pixel positions on axes outside image
   # To facilitate the parsing of defaults one needs to allow 
   # pre-defined values for slicepos.
   if slicepos is None:
      slicepos = []
      if n > 2:
         for i in range(n):
            axnr = i + 1
            maxn = fitsobj.axisinfo[axnr].axlen
            if (axnr not in [axnum1, axnum2]):
               unvalid = True
               while unvalid:
                  crpix = fitsobj.axisinfo[axnr].crpix
                  if crpix < 1 or crpix > fitsobj.axisinfo[axnr].axlen:
                     crpix = 1
                  prompt = "Enter pixel position between 1 and %d on %s ..... [%d]: " % (maxn, fitsobj.axisinfo[axnr].axname,crpix)
                  x = raw_input(prompt)
                  if x == "":
                     x = int(crpix)
                  else:
                     x = int(eval(x))
                     #x = eval(x); print "X=", x, type(x)
                  unvalid = not (x >= 1 and x <= maxn)
                  if unvalid:
                     print "Pixel position not in range 1 to %d! Try again." % maxn
               slicepos.append(x)

   return axnum1, axnum2, slicepos



def prompt_spectrans(fitsobj):
#-----------------------------------------------------------------------
   """
Ask user to enter spectral translation if one of the axes is spectral.

:param fitsobj:
   An object from class FITSimage. From this object we derive the allowed spectral
   translations.
:type fitsobj:
   Instance of class FITSimage

:Prompts:

      The spectral translation if one of the image axes is a spectral axis.
   
         *Enter number between 0 and N of spectral translation .... [native]:*

         *N* is the number of allowed translations  minus 1.
         The default *Native* in this context implies that no translation is applied.
         All calculations are done in the spectral type given by FITS header
         item *CTYPEn* where *n* is the number of the spectral axis.

:Returns: 
   * *spectrans* - The selected spectral translation from a list with spectral
     translations that are allowed for the input object of class FITSimage.
     A spectral translation translates for example frequencies to velocities.

:Example:

   >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
   >>> print fitsobject.str_spectrans()    # Print a list with options first
   >>> fitsobject.set_spectrans(promptfie=maputils.prompt_spectrans)
   """
#-----------------------------------------------------------------------
   asktrans = False
   for ax in fitsobj.axperm:
      if fitsobj.axisinfo[ax].wcstype == 'spectral':
         asktrans = True

   spectrans = None
   nt = len(fitsobj.allowedtrans)
   if (nt > 0 and asktrans):
      s = ''
      for i, tr in enumerate(fitsobj.allowedtrans):
         s += "%d:%s (%s) " % (i, tr[0], tr[1])
      print s
      unvalid = True
      while unvalid:
         try:
            prompt = "Enter number between 0 and %d of spectral translation .... [native]: " % (nt - 1)
            st = raw_input(prompt)
            if st != '':
               st = int(st)
               unvalid = (st < 0 or st > nt-1)
               if unvalid:
                  print "Not a valid number!"
               else:
                  spectrans = fitsobj.allowedtrans[st][0]
            else:
               unvalid = False
         except KeyboardInterrupt:
            raise
         except:
            unvalid = True

   return spectrans



def prompt_skyout(fitsobj):
#-----------------------------------------------------------------------
   """
Ask user to enter the output sky system if the data is a spatial map.

:param fitsobj:
   An object from class FITSimage. This prompt function uses this object to
   get information about the axis numbers of the spatial axes in a
   data structure.
:type fitsobj:
   Instance of class FITSimage

:Returns:
   * *skyout* - The sky definition to which positions in the native system
      will be transformed.

:Example:

   >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
   >>> fitsobject.set_skyout(promptfie=maputils.prompt_skyout)
   """
#-----------------------------------------------------------------------
   # The check below is also done in the calling environment (method set_skyout)
   # but we need this test also here when this function is used in a different context.
   spatials = [fitsobj.proj.lonaxnum, fitsobj.proj.lataxnum]
   spatialmap = fitsobj.axperm[0] in spatials and fitsobj.axperm[1] in spatials
   if not spatialmap:
      return         # Silently

   skyout = skysys = refsys = equinox = epoch = None
   unvalid = True
   maxskysys = 4
   while unvalid:
      try:
         prompt = "Sky system 0=eq, 1=ecl, 2=gal, 3=sup.gal .... [native]: "
         st = raw_input(prompt)
         if st != '':
            st = int(st)
            unvalid = (st < 0 or st >= maxskysys)
            if unvalid:
               print "Not a valid number!"
            else:
               skysys = st
         else:
            unvalid = False
      except KeyboardInterrupt:
         raise
      except:
         unvalid = True

   if skysys in [wcs.equatorial, wcs.ecliptic]:    # Equatorial or ecliptic, so ask reference system
      unvalid = True
      maxrefsys = 5
      prompt = "Ref.sys 0=fk4, 1=fk4_no_e, 2=fk5, 3=icrs, 4=dynj2000 ... [native]: "
      while unvalid:
         try:
            st = raw_input(prompt)
            if st != '':
               st = int(st)
               unvalid = (st < 0 or st >= maxrefsys)
               if unvalid:
                  print "Not a valid number!"
               else:
                  refsys = st + maxskysys  # The ref. systems start at maxskysys
            else:
               unvalid = False
         except KeyboardInterrupt:
            raise
         except:
            unvalid = True

      if refsys in [wcs.fk4, wcs.fk4_no_e, wcs.fk5]:
         prompt = "Enter equinox (e.g. J2000 or B1983.5) .... [native]: "
         unvalid = True
         while unvalid:
            try:
               st = raw_input(prompt)
               if st != '':
                  B, J, JD = epochs(st)
                  equinox = st
               unvalid = False
            except KeyboardInterrupt:
               raise
            except:
               unvalid = True

      if refsys in [wcs.fk4, wcs.fk4_no_e]:
         prompt = "Enter date of observation (e.g. MJD24034) .... [native]: "
         unvalid = True
         while unvalid:
            try:
               st = raw_input(prompt)
               if st != '':
                  B, J, JD = epochs(st)
                  epoch = st
               unvalid = False
            except KeyboardInterrupt:
               raise
            except:
               unvalid = True

   if skysys is None:
      skyout = None
   else:
      skyout = []
      skyout.append(skysys)
      if equinox != None:
         skyout.append(equinox)
      if refsys != None:
         skyout.append(refsys)
      if epoch != None:
         skyout.append(epoch)
      skyout = tuple(skyout)

   return skyout


def prompt_dataminmax(fitsobj):
   #-----------------------------------------------------------------------
   """
    Ask user to enter one or two clip values.
    If one clip level is entered then in display routines
    the data below this value will be clipped. If a second level is
    entered, then all data values above this level
    will also be filtered.

   :param fitsobj:
      An object from class FITSimage.
   :type fitsobj:
      Instance of class FITSimage

   :Returns:
      * *clipmin*, *clipmax* - Two values to set limits on the image value
        e.g. for color editing.

   :Example:

      >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
      >>> clipmin, clipmax = maputils.prompt_dataminmax(fitsobject)
      >>> annim = fitsobject.Annotatedimage(frame, clipmin=clipmin, clipmax=clipmax)
   """
   #-----------------------------------------------------------------------
   mi, ma = fitsobj.get_dataminmax(box=True)
   mes = "Enter clip levels      [%g %g]: " % (mi, ma)
   clips = raw_input(mes)
   if clips:
      clips = re_split('[, ]+', clips.strip())
      if len(clips) == 1:
         clipmin = eval(clips[0])
         clipmax = ma
      else:
         clipmin = eval(clips[0])
         clipmax = eval(clips[1])
   else:
      clipmin = mi
      clipmax = ma
   return clipmin, clipmax


class Image(object):
   #--------------------------------------------------------------------
   """
   Prepare the FITS- or external image data to be plotted in Matplotlib.
   All parameters are set by method :meth:`Annotatedimage.Image`.
   The keyword arguments are those for Matplotlib's method *imshow()*.
   Two of them are useful in the context of this class. These parameters
   are *visible*, a boolean to set the visibility of the image to on or off,
   and *alpha*, a number between 0 and 1 which sets the transparency of
   the image.

   See also: :meth:`Annotatedimage.Image`

   Methods:

   .. automethod:: plot
   """
   #--------------------------------------------------------------------
   def __init__(self, imdata, box, cmap, norm, **kwargs):
      #--------------------------------------------------------------------
      # Prepare the FITS- or external image data to be plotted in Matplotlib
      #--------------------------------------------------------------------
      self.ptype = "Image"
      self.box = box
      newkwargs = ({'origin':'lower', 'extent':self.box,
                    'interpolation':'nearest'})
      # Our first choice was interpolation='none' which is an option from
      # Matplotlib version 1.1.0 and newer. However without interpolation it
      # is impossible to use the set_alpha method to make an image transparent
      
      newkwargs.update(kwargs)
      self.kwargs = newkwargs
      self.data = imdata
      self.frame = None                  # MPL Axes object is where the image is displayed
      self.im = None                     # The MPL image as a result of imshow()
      self.xyn_mouse = [0.5,0.5]         # Mouse position for color editing
      self.cmap = cmap
      self.norm = norm


   def plot(self, frame):
      #--------------------------------------------------------------------
      """
      Plot image object. Usually this is done by method
      :meth:`Annotatedimage.plot` but it can also be used separately.

      """
      #--------------------------------------------------------------------
      if self.data is None:
         raise Exception, "Cannot plot image because image data is not available!"
      self.frame = frame
      # In the following call to imshow() it seems to be necessary to set the
      # aspect ratio explicitly. We copy its value from the current frame.
      self.im = self.frame.imshow(self.data, cmap=self.cmap, norm=self.norm,
                                  aspect=frame.get_aspect(), **self.kwargs)
      self.frame.set_xlim((self.box[0], self.box[1]))
      self.frame.set_ylim((self.box[2], self.box[3]))
      


class Contours(object):
   #--------------------------------------------------------------------
   """
   Objects from this class calculate and plot contour lines.
   Most of the parameters are set by method
   :meth:`Annotatedimage.Contours`. The others are:

   
   :param filled:
      If True, then first create filled contours and draw
      the contour lines upon these filled contours
   :type filled:
      Boolean
   :param negative:
      Set the line style of the contours that represent negative
      image numbers. The line styles are Matplotlib line styles e.g.:
      [None | 'solid' | 'dashed' | 'dashdot' | 'dotted']
   :type negative:
      String
   :param kwargs:
      Parameters for properties of all contours (e.g. *linewidths*).
      

   :Notes:
      If the line widths of contours are given in the constructor
      (parameter *linewidths*) then these linewidths are copied to the
      line widths in the colorbar (if requested).


   Methods:

   .. automethod:: plot
   .. automethod:: setp_contour
   .. automethod:: setp_label
   """
   #--------------------------------------------------------------------
   def __init__(self, imdata, box, levels=None, cmap=None, norm=None,
                filled=False, negative="dashed", **kwargs):
      #--------------------------------------------------------------------
      # See class description
      #--------------------------------------------------------------------
      self.ptype = "Contour"                     # Set type of this object
      self.box = box
      self.cmap = cmap                           # If not None, select contour colours from cmap
      self.norm = norm                           # Scale data according to this normalization
      newkwargs = ({'origin':'lower', 'extent':box})   # Necessary to get origin right
      newkwargs.update(kwargs)                   # Input kwargs can overrule this.
      self.kwargs = newkwargs
      self.data = imdata                         # Necessary for methods contour/contourf in plot()
      self.clevels = levels
      self.commoncontourkwargs = None            # Is set for all contours by method setp_contour()
      self.ckwargslist = None                    # Properties in setp_contour for individual contours
      if self.clevels != None:
         self.ckwargslist = [None]*len(self.clevels)
      self.commonlabelkwargs = None              # Is set for all contour labels in setp_labels()
      self.lkwargslist = None                    # Properties in setp_labels for individual labels
      if self.clevels != None:
         self.lkwargslist = [None]*len(self.clevels)
      self.labs = None                           # Label objects from method clabel() 
      self.filled = filled                       # Do we require filled contours?
      # Prevent exception for the contour colors
      if self.kwargs.has_key('colors'):          # One of them (colors or cmap) must be None!
         self.cmap = None                        # Given colors overrule the colormap
      self.negative = negative


   def plot(self, frame):
      #--------------------------------------------------------------------
      """
      Plot contours object. Usually this is done by method
      :meth:`Annotatedimage.plot` but it can also be used separately.
      """
      #--------------------------------------------------------------------
      if self.data is None:
         raise Exception, "Cannot plot image because image data is not available!"
      self.frame = frame
      if self.clevels is None:
         if self.filled:
            self.frame.contourf(self.data, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.CS = self.frame.contour(self.data, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.clevels = self.CS.levels
      else:
         if not issequence(self.clevels):
            self.clevels = [self.clevels]
         if self.filled:
            self.frame.contourf(self.data, self.clevels, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.CS = self.frame.contour(self.data, self.clevels, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.clevels = self.CS.levels
      # Restore the frame that includes entire pixels
      self.frame.set_xlim((self.box[0], self.box[1]))
      self.frame.set_ylim((self.box[2], self.box[3]))
      # Properties
      if self.commoncontourkwargs != None:
         for c in self.CS.collections:
            plt_setp(c, **self.commoncontourkwargs)
      if self.ckwargslist != None:
         for i, kws in enumerate(self.ckwargslist):
            if kws != None:
               plt_setp(self.CS.collections[i], **kws)

      if self.negative != None:
         for i, lev in enumerate(self.CS.levels):
            if lev < 0:
               plt_setp(self.CS.collections[i], linestyle=self.negative)
               
      if self.commonlabelkwargs != None:
         self.labs = self.frame.clabel(self.CS, **self.commonlabelkwargs)
         #for c in self.labs:
         #   plt_setp(c, **self.commonlabelkwargs)
            
      if self.lkwargslist != None:
         for i, kws in enumerate(self.lkwargslist):
            if kws != None:
               lab = self.frame.clabel(self.CS, [self.clevels[i]], **kws)
               #plt_setp(lab, **kws)


   def setp_contour(self, levels=None, **kwargs):
      #--------------------------------------------------------------------
      """
      Set properties for contours either for all contours if *levels*
      is omitted or for specific levels if keyword *levels*
      is set to one or more levels.

      :Examples:
      
      >>> cont = annim.Contours(levels=range(10000,16000,1000))
      >>> cont.setp_contour(linewidth=1)
      >>> cont.setp_contour(levels=11000, color='g', linewidth=3)
      """
      #--------------------------------------------------------------------
      if levels is None:
         self.commoncontourkwargs = kwargs
      else:
         if self.clevels is None:
            raise Exception, "Contour levels not set so I cannot identify contours"
         # Change only contour properties for levels in parameter 'levels'.
         if not issequence(levels):
            levels = [levels]
         for lev in levels:
            try:
               i = list(self.clevels).index(lev)
            except ValueError:
               i = -1 # no match
            if i != -1:
               self.ckwargslist[i] = kwargs


   def setp_label(self, levels=None, tex=True, **kwargs):
      #--------------------------------------------------------------------
      """
      Set properties for the labels along the contours.
      The properties are Matplotlib properties (fontsize, colors,
      inline, fmt).

      :param levels:
          None or one or more levels from the set of given contour levels
      :type levels:
          None or one or a sequence of numbers

      :param tex:
         Print the labels in TeX if a format is entered.
         If set to True, add '$' characters
         so that Matplotlib knows that it has to format the label
         in TeX. The default is *True*. 
      :type tex:
         Boolean

      Other parameters are Matplotlib parameters for method
      :meth:`clabel` in Matplotlib :class:`ContourLabeler`
      (fontsize, colors, inline, fmt).
      
      :Examples:

      >>> cont2 = annim.Contours(levels=(8000,9000,10000,11000))
      >>> cont2.setp_label(11000, colors='b', fontsize=14, fmt="%.3f")
      >>> cont2.setp_label(fontsize=10, fmt="%g \lambda")
      """
      #--------------------------------------------------------------------
      if tex and kwargs.has_key('fmt'):
         kwargs['fmt'] = r'$'+kwargs['fmt']+'$'
      if levels is None:
         self.commonlabelkwargs = kwargs
      else:
         if self.clevels is None:
            raise Exception, "Contour levels not set so I cannot identify contours"
         # Change only contour properties for levels in parameter 'levels'.
         if not issequence(levels):
            levels = [levels]
         for lev in levels:
            try:
               i = list(self.clevels).index(lev)
            except ValueError:
               i = -1 # no match
            if i != -1:
               self.lkwargslist[i] = kwargs


class Colorbar(object):
   #--------------------------------------------------------------------
   """
   Colorbar class. Usually the parameters will be provided by method
   :meth:`Annotatedimage.Colorbar`

   Useful keyword parameters:

   :param frame:
      If a frame is given then this frame will be the colorbar frame.
      If None, the frame is calculated by taking space from its parent
      frame.
   :type frame:
      Matplotlib Axes instance

      
   .. automethod:: plot
   .. automethod:: set_label
   """
   #--------------------------------------------------------------------
   def __init__(self, cmap, frame=None, norm=None, contourset=None, clines=False, fontsize=9,
                label=None, linewidths=None, visible=True, **kwargs):
      #--------------------------------------------------------------------
      """
      cmap: Was usuable in versions <= 0.99.8. Now we cannot use
            colorbaseBase anymore (changing color map did not update
            the colorbar) and have to use the colorbar method instead.
            Then in fact, either an image or a contourset should be
            supplied and the cmap is not necessary. For now we leave the code
            unaltered. TODO: Address the use of cmap.
      frame: If a frame is entered it will be stored here and used as
            Axes object. However when None, the calling environment
            must call plot() with a valid frame (usually this frame
            will be 'stolen' from a mother frame.
            clines: Draw contour lines in colorbar also. Does not matter
            if this comes from an image or contour set.
      linewidhts: One number to set the line width of all the contour
            lines in the colorbar. Note that linewidths set for a
            contour level will be copied for the line width of the
            line in the colorbar. However if parameter *linewidths*
            is set, it will be applied to all lines in the contour colorbar.
      """
      #--------------------------------------------------------------------
      self.ptype = "Colorbar"
      self.cmap = cmap
      self.norm = norm
      self.contourset = contourset
      self.plotcontourlines = clines
      self.cbfontsize = fontsize
      self.linewidths = linewidths
      self.frame = frame
      self.label = label
      self.labelkwargs = None
      newkwargs = ({'orientation':'vertical'})
      if not visible:
         alpha = {'alpha':0.0}
         newkwargs.update(alpha)
      newkwargs.update(kwargs)
      self.kwargs = newkwargs


   def colorbarticks(self):
      #--------------------------------------------------------------------
      """
      Usually used within the context of this class, but can also be used
      if one needs to resize the colorbar tick labels in the calling
      environment.
      """
      #--------------------------------------------------------------------
      for t in self.cb.ax.get_xticklabels():  # Smaller font for color bar
         t.set_fontsize(self.cbfontsize)
      for t in self.cb.ax.get_yticklabels():  # Smaller font for color bar along y
         t.set_fontsize(self.cbfontsize)


   def plot(self, cbframe, mappable=None):
      #--------------------------------------------------------------------
      """
      Plot image object. Usually this is done by method
      :meth:`Annotatedimage.plot` but it can also be used separately.

      Note:
      We changed the default formatter for the colorbar. This can be done with
      the 'format' parameter. We changed the formatter to a fixed format string.
      This prevents that MPL uses an offset and a scaling for the labels.
      If MPL does this, it adds an extra label, showing the offset and scale
      in scientific notation. We do not want this extra label because we don't
      have enough control over it (e.g. it can appear outside your viewport
      or in a black background with a black font). With the new formatter
      we are sure to get the real value in our labels.
      """
      #--------------------------------------------------------------------
      self.frame = cbframe                 # self.frame could have been None
      fig = cbframe.figure                 # We need the figure to use the colorbar method


      majorFormatter = FormatStrFormatter("%g")
      self.cb = fig.colorbar(mappable, cax=self.frame, norm=self.norm,
                             format=majorFormatter, **self.kwargs)

      if self.plotcontourlines and self.contourset != None:
         CS = self.contourset.CS
         if not self.kwargs.has_key("ticks"):
            self.kwargs["ticks"] = CS.levels
         # Copy the line widhts. Note that this is a piece of code
         # that assumes certain behaviour from the attributes in a
         # contour set. It can fail with newer Matplotlib versions.
         lws = []
         if not self.linewidths is None:
            for lw in CS.tlinewidths:
               lws.append((self.linewidths,))
            CS.tlinewidths = lws
         self.cb.add_lines(CS)
      else:
          CS = None

      self.colorbarticks()    # Set font size given in kwargs or use default
      if self.label != None:
         if self.labelkwargs is None:
            self.cb.set_label(self.label)
         else:
            self.cb.set_label(self.label, **self.labelkwargs)


   def set_label(self, label, **kwargs):
      #--------------------------------------------------------------------
      """
      Set a text label along the long side of the color bar.
      It is a convenience routine for Matplotlib's *set_label()*
      but this one needs a plotted colorbar while we postpone plotting.
      """
      #--------------------------------------------------------------------
      self.label = label
      self.labelkwargs = kwargs


class Beam(object):
   #--------------------------------------------------------------------
   """
   Beam class. Usually the parameters will be provided by method
   :meth:`Annotatedimage.Beam`
   
   Objects from class Beam are graphical representations of the resolution
   of an instrument. The beam is centered at a position xc, yc.
   The major axis of the beam is the FWHM of longest distance between
   two opposite points on the ellipse. The angle between the major axis
   and the North is the position angle.

   Note that it is not correct to calculate the ellipse that represents
   the beam by applying distance 'r' (see code) as a function of
   angle, to get the new world coordinates. The reason is that the
   fwhm's are given as sizes on a sphere and therefore a correction for
   the declination is required. With method *dispcoord()* (see source code
   of class Beam)
   we sample the
   ellipse on a sphere with a correct position angle and with the correct
   sizes.

   """
   #--------------------------------------------------------------------
   def __init__(self, xc, yc, fwhm_major, fwhm_minor, pa, projection=None,
                units=None, **kwargs):
      self.ptype = "Beam"

      if units != None:
         uf, errmes = unitfactor('degree', units)
         if uf is None:
            raise ValueError(errmes)
         else:
            fwhm_major /= uf
            fwhm_minor /= uf

      semimajor = fwhm_major / 2.0
      semiminor = fwhm_minor / 2.0
      Pi = numpy.pi
      startang, endang, delta = (0.0, 360.0, 1.0)
      sinP = numpy.sin( pa*Pi/180.0 )
      cosP = numpy.cos( pa*Pi/180.0 )
      phi  = numpy.arange( startang, endang+delta, delta, dtype="f" ) 
      cosA = numpy.cos( phi*Pi/180.0 )
      sinA = numpy.sin( phi*Pi/180.0 )
      d = (semiminor*cosA) * (semiminor*cosA) + (semimajor*sinA) * (semimajor*sinA)
      r = numpy.sqrt( (semimajor*semimajor * semiminor*semiminor)/d )

      self.p1 = None
      self.p2 = None
      lon_new, lat_new = dispcoord(xc, yc, r, -1, phi+pa)
      splitlon = get_splitlon(projection)
      if splitlon is None:
         lon1, lat1 = lon_new, lat_new
         lon2 = lat2 = []
      else:
         lon1, lat1, lon2, lat2 = split_polygons(lon_new, lat_new, splitlon)

      if len(lon1):
         xp, yp = projection.topixel((lon1, lat1))
         self.p1 = Polygon(zip(xp, yp), **kwargs)
      if len(lon2):
         xp, yp = projection.topixel((lon2, lat2))
         self.p2 = Polygon(zip(xp, yp), **kwargs)

   def plot(self, frame):
      if not self.p1 is None:
         frame.add_patch(self.p1)
      if not self.p2 is None:
         frame.add_patch(self.p2)


class Skypolygon(object):
#--------------------------------------------------------------------
   """
   This class defines objects that can only be plotted onto
   spatial maps.
   Usually the parameters will be provided by method
   :meth:`Annotatedimage.Skypolygon`
   
   """
#--------------------------------------------------------------------
   def __init__(self, projection, prescription=None,
                xc=None, yc=None,
                major=None, minor=None,
                nangles=6, pa=0.0,
                units=None,
                lons=None, lats=None,
                stepsize=1.0,
                **kwargs):

      self.ptype = "Skypolygon"
      self.p1 = self.p2 = None        # Shape could be splitted into two parts
      self.patch = None
      splitlon = get_splitlon(projection)

      if prescription is None:
         if lons is None and lats is None:
            raise ValueError("No prescription entered nor longitudes and latitudes")
         elif lons is None or lats is None:
            raise ValueError("No prescription entered and missing longitudes or latitudes")
      else:    # Test minimal set of required parameters
         if xc is None or yc is None:
            raise ValueError("Missing value for center xc or yc!")
         if major is None and minor is None:
            raise ValueError("Both major and minor axes are not specified!")
         if major is None:
            major = minor
         if minor is None:
            minor = major
         if units != None:
            uf, errmes = unitfactor('degree', units)
            if uf is None:
               raise ValueError(errmes)
            else:
               major /= uf
               minor /= uf
         Pi = numpy.pi
         if prescription[0].upper() == 'E':
            semimajor = major / 2.0
            semiminor = minor / 2.0
            startang, endang, delta = (0.0, 360.0, abs(stepsize)) # should be enough samples
            sinP = numpy.sin( pa*Pi/180.0 )
            cosP = numpy.cos( pa*Pi/180.0 )
            phi  = numpy.arange( startang, endang+delta, delta, dtype="f") 
            cosA = numpy.cos( phi*Pi/180.0 )
            sinA = numpy.sin( phi*Pi/180.0 )
            d = (semiminor*cosA) * (semiminor*cosA) + (semimajor*sinA) * (semimajor*sinA)
            r = numpy.sqrt( (semimajor*semimajor * semiminor*semiminor)/d )
            lons, lats = dispcoord(xc, yc, r, -1, phi+pa)
            #for lo,la in zip(lons,lats):
            #   print "%.1f %.1f"%(lo,la)
         elif prescription[0].upper() == 'R':
            # Create rectangle with Major as the long side and aligned with North
            xs = minor/2.0
            ys = -major/2.0
            samples = 100
            deltax = minor/float(samples)
            deltay = major/float(samples)
            x = numpy.zeros(samples)
            y = numpy.zeros(samples)
            for i in range(samples):
               x[i] = xs
               y[i] = ys + i *deltay
            phi1 = numpy.arctan2(y,x)
            r1 = numpy.hypot(x,y)

            xs = minor/2.0
            ys = major/2.0
            for i in range(samples):
               x[i] = xs - i * deltax
               y[i] = ys
            phi2 = numpy.arctan2(y,x)
            r2 = numpy.hypot(x,y)

            xs = -minor/2.0
            ys =  major/2.0
            for i in range(samples):
               x[i] = xs
               y[i] = ys - i *deltay
            phi3 = numpy.arctan2(y,x)
            r3 = numpy.hypot(x,y)

            xs = -minor/2.0
            ys = -major/2.0
            for i in range(samples):
               x[i] = xs + i  * deltax
               y[i] = ys
            phi4 = numpy.arctan2(y,x)
            r4 = numpy.hypot(x,y)
            phi = numpy.concatenate((phi1, phi2, phi3, phi4))*180/Pi+90 # dispcoord wants it in degs.
            r = numpy.concatenate((r1, r2, r3, r4))
            lons, lats = dispcoord(xc, yc, r, -1, phi+pa)
            
         elif prescription[0].upper() == 'N':
            if nangles < 3:
               raise ValueError("Number of angles in regular polygon must be > 2!")
            nsamples = 360.0/nangles
            psi = numpy.linspace(0, 360, nangles+1)
            radius = major/2.0;
            xs = radius;  ys = 0.0    # Start value
            first = True
            for ang in psi[1:]:
               xe = radius * numpy.cos(ang*Pi/180)
               ye = radius * numpy.sin(ang*Pi/180)
               lambd = numpy.arange(0.0, 1.0, 1.0/nsamples)
               x = xs + lambd * (xe - xs)
               y = ys + lambd * (ye - ys)
               Phi = numpy.arctan2(y,x)*180.0/Pi
               R = numpy.hypot(x,y)
               xs = xe
               ys = ye
               if first:
                  phi = Phi.copy()
                  r = R.copy()
                  first = False
               else:
                  phi = numpy.concatenate((phi, Phi))
                  r = numpy.concatenate((r, R))
            lons, lats = dispcoord(xc, yc, r, -1, phi+pa)
               
      if lons is None or lats is None:
         raise ValueError("Unknown prescription")

      if splitlon is None:
         lon1, lat1 = lons, lats
         lon2 = lat2 = []
      else:
         lon1, lat1, lon2, lat2 = split_polygons(lons, lats, splitlon)
      if len(lon1):
         xp, yp = projection.topixel((lon1, lat1))
         self.p1 = Polygon(zip(xp, yp), closed=True, **kwargs)
         self.lon1 = lon1; self.lat1 = lat1
      if len(lon2):
         xp, yp = projection.topixel((lon2, lat2))
         self.p2 = Polygon(zip(xp, yp), closed=True, **kwargs)
         self.lon2 = lon2; self.lat2 = lat2

   def plot(self, frame):
      if not self.p1 is None:
         frame.add_patch(self.p1)
         self.patch = self.p1
      if not self.p2 is None:
         frame.add_patch(self.p2)
         self.patch = self.p2



class Marker(object):
#--------------------------------------------------------------------
   """
   Marker class. Usually the parameters will be provided by method
   :meth:`Annotatedimage.Marker`

   Mark features in your map with a marker symbol. Properties of the
   marker are set with Matplotlib's keyword arguments.
   """
#--------------------------------------------------------------------
   def __init__(self, xp=None, yp=None, **kwargs):
      self.ptype = "Marker"
      self.xp = xp
      self.yp = yp
      self.kwargs = kwargs
      self.patch = None


   def plot(self, frame):
      if self.xp is None or self.yp is None:
         return
      if not issequence(self.xp):
         self.xp = [self.xp]
      if not issequence(self.yp):
         self.yp = [self.yp]
      self.patch = frame.plot(self.xp, self.yp, 'o', **self.kwargs)  # Set default marker symbol to prevent connections
      self.patch = self.patch[0]

      

class Gridframe(object):
   """
   -------------------------------------------------------------------------------
   Helper class which defines objects with properties which are read
   when pixel coordinates need to be plotted.
   -------------------------------------------------------------------------------
   """
   def __init__(self, pxlim, pylim, plotaxis, gridlines, major, minor, **kwargs):
      self.ptype = "Gridframe"
      self.pxlim = pxlim
      self.pylim = pylim
      self.plotaxis = plotaxis
      self.markerkwargs = {}
      self.kwargs = kwargs
      self.gridlines = gridlines
      self.major = major
      self.minor = minor


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

   :param major:        This number overrules the default positions
                        for the major tick marks. The tick marks and labels
                        are plotted at a multiple number of *major*.
   :type major:         Float or Integer (usually the input will be an integer).

   :param minor:        This number sets the plotting of minor tick marks on.
                        The markers are plotted at a multiple value of
                        *minor*.
   :type minor:         Float or Integer (usually the input will be an integer).
   
   :param offset:       The pixels can have an integer offset.
                        If you want the reference pixel to be pixel
                        0 then supply offset=(crpixX, crpixY).
                        These crpix values are usually read from then
                        header. In this routine the nearest integer of
                        the input is calculated to ensure that the
                        offset is an integer value.
   :type offset:        *None* or a floating point number
   
   :param kwargs:       Keyword arguments to set attributes for
                        the labels (e.g. color='g', fontsize=8)
   :type kwargs:        Matplotlib keyword argument(s)
   
   :Returns:            An object from class *Gridframe* which
                        is added to the plot container with Plotversion's
                        method :meth:`Plotversion.add`.
   
   :Notes:              Graticules and Pixellabels are plotted in their own
                        plot frame. If you want to be able to toggle grid lines
                        in a frame labeled with pixel coordinates, then you have to make sure
                        that the Pixellabels frame is plotted last. So always define
                        Pixellabels objects before Graticule objects. 
   
   :Examples:           Annotate the pixels in a plot along the right and top axis
                        of a plot. Change the color of the labels to red::
   
                           mplim = f.Annotatedimage(frame)
                           mplim.Pixellabels(plotaxis=("bottom", "right"), color="r")

                           or with separate axes:
   
                           mplim.Pixellabels(plotaxis="bottom", color="r")

                           mplim.Pixellabels(plotaxis="right", color="b", markersize=10)
                           mplim.Pixellabels(plotaxis="top", color="g", markersize=-10, gridlines=True)

   .. automethod:: setp_marker
   .. automethod:: setp_label
   """
   def __init__(self, pxlim, pylim, plotaxis=None, markersize=None,
                gridlines=False, ticks=None, major=None, minor=None, offset=None, **kwargs):

      def nint(x):
         return numpy.floor(x+0.5)

      self.ptype = "Pixellabels"       # not a gridframe object
      defkwargs = {'fontsize':7}
      defkwargs.update(kwargs)
      if plotaxis is None:
         plotaxis = [2,3]

      px = [0,0]; py = [0,0]
      px[0] = pxlim[0]; py[0] = pylim[0]    # Do not copy directly because new values must be temporary
      px[1] = pxlim[1]; py[1] = pylim[1]
      if offset != None:
         offX = nint(offset[0])
         offY = nint(offset[1])
         px[0] -= offX; px[1] -= offX;
         py[0] -= offY; py[1] -= offY;
   
      gridlabs = Gridframe(px, py, plotaxis, gridlines, major, minor, **defkwargs)
      self.gridlabs = gridlabs
      self.frame = None


   def setp_marker(self, **kwargs):
      #--------------------------------------------------------------------
      """
      Set properties of the pixel label tick markers
       
      :param kwargs:  keyword arguments to change properties of
                      the tick marks. A tick mark is a Matploltlib
                      :class:`Line2D` object with attributes like
                      *markeredgewidth* etc.
      :type kwargs:   Python keyword arguments.
      """
      #--------------------------------------------------------------------
      self.gridlabs.markerkwargs.update(kwargs)

   
   def setp_label(self, **kwargs):
      #--------------------------------------------------------------------
      """
      Set properties of the pixel label tick markers
       
      :param kwargs:  keyword arguments to change properties of
                      (all) the tick labels. A tick mark is a Matploltlib
                      :class:`Text` object with attributes like
                      *fontsize*, *fontstyle* etc.
      :type kwargs:   Python keyword arguments.
      """
      #--------------------------------------------------------------------
      self.gridlabs.kwargs.update(kwargs)


   def plot(self, frame):
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

      -----------------------------------------------------------
      """
      fig = frame.figure
      pixellabels = self.gridlabs
      plotaxes = wcsgrat.parseplotaxes(pixellabels.plotaxis)  # Is always a list with integers now!
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
      
      aspect = frame.get_aspect()
      adjust = frame.get_adjustable()
      kwargs = pixellabels.kwargs
      markerkwargs = pixellabels.markerkwargs
      xlo = pixellabels.pxlim[0]-0.5 
      ylo = pixellabels.pylim[0]-0.5
      xhi = pixellabels.pxlim[1]+0.5
      yhi = pixellabels.pylim[1]+0.5
      # Copy frame
      framelabel = randomlabel('fr_')
      try:
         r,c,n = frame.get_geometry()
         gframe = fig.add_subplot(r, c, n,
                                  aspect=aspect,
                                  adjustable=adjust,
                                  autoscale_on=False,
                                  frameon=False,
                                  label=framelabel)
         gframe.set_position(frame.get_position())
      except:
         gframe = fig.add_axes(frame.get_position(),
                               aspect=aspect,
                               adjustable=adjust,
                               autoscale_on=False,
                               frameon=False,
                               label=framelabel)

      gframe.set_xlim((xlo,xhi))
      gframe.set_ylim((ylo,yhi))
      self.frame = gframe
      
      if 3 in plotaxes or 1 in plotaxes:
         if pixellabels.major != None:
            majorLocator = MultipleLocator(pixellabels.major)
            gframe.xaxis.set_major_locator(majorLocator)
         if pixellabels.minor:
            minorLocator = MultipleLocator(pixellabels.minor)
            gframe.xaxis.set_minor_locator(minorLocator)

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
            tick.tick1On = False
            tick.tick2On = False

      setmarker = (pixellabels.markerkwargs) > 0
      for tick in gframe.xaxis.get_major_ticks():
         if 3 in plotaxes:
            tick.label2.set(**kwargs)
            if setmarker:
               tick.tick2line.set(**markerkwargs)
            tick.tick1On = False
            tick.gridOn = pixellabels.gridlines
         elif 1 in plotaxes:
            tick.label1.set(**kwargs)
            if setmarker:
               tick.tick1line.set(**markerkwargs)
            tick.tick2On = False
            tick.gridOn = pixellabels.gridlines

      if 2 in plotaxes or 0 in plotaxes:
         if pixellabels.major != None:
            majorLocator = MultipleLocator(pixellabels.major)
            gframe.yaxis.set_major_locator(majorLocator)
         if pixellabels.minor:
            minorLocator = MultipleLocator(pixellabels.minor)
            gframe.yaxis.set_minor_locator(minorLocator)
            
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
            tick.tick1On = False
            tick.tick2On = False
         
      for tick in gframe.yaxis.get_major_ticks():
         if 2 in plotaxes:
            tick.label2.set(**kwargs)
            if setmarker:
               tick.tick2line.set(**markerkwargs)
            tick.tick1line.set_visible(False)
            tick.gridOn = pixellabels.gridlines
         elif 0 in plotaxes:
            tick.label1.set(**kwargs)
            if setmarker:
               tick.tick1line.set(**markerkwargs)
            tick.tick2line.set_visible(False)
            tick.gridOn = pixellabels.gridlines
               
      # gframe.grid(pixellabels.gridlines)
      fig.sca(frame)    # back to frame from calling environment


class Annotatedimage(object):
#--------------------------------------------------------------------
   """
This is one of the core classes of this module. It sets the connection
between the FITS data (created or read from file) and the routines that
do the actual plotting with Matplotlib.
The class is usually used in the context of class :class:`FITSimage` which
has a method that prepares the parameters for the constructor of
this class.

:param frame:
   This is the frame where image and or contours will be plotted.
   If omitted then a default frame will be set
:type frame:
   Matplotlib Axes instance
:param header:
   The header data for this file. Either from a FITS header or a
   dictionary with header data.
:type header:
   Python dictionary or pyfits.NP_pyfits.Header instance
:param pxlim:
   Two integer numbers which should not be smaller than 1 and not
   bigger than the header value *NAXISn*, where n represents the
   x axis.
:type pxlim:
   Tuple with two integers
:param pylim:
   Two integer numbers which should not be smaller than 1 and not
   bigger than the header value *NAXISn*, where n represents the
   y axis.
:type pylim:
   Tuple with two integers
:param imdata:
   Image data. This data must represent the area defined by
   *pxlim* and *pylim*.
:type imdata:
   2D NumPy array
:param projection:
   The current projection object which provides this class
   with conversion methods from :mod:`wcs` like
   :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
   needed for conversions between pixel- and world coordinates.
:type projection:
   Instance of Projection class from module :mod:`wcs`
:param axperm:
   Tuple or list with the FITS axis number of the two image axes,
   e.g. axperm=(1,2)
:type axperm:
   Tuple with two integers
:param wcstypes:
   In some modules we need to know what the type of an axis in the image is
   so that for example we can find out whether two different images have swapped axes.
   The order of this list is the same as the order in the original FITS file.
   'lo' is longitude axis, 'la' is latitude axis,
   'sp' is spectral axis, 'li' is a linear axis. Appended to 'li' is an
   underscore and the ctype of that axis (e.g. 'li_stokes').
   If the original data has axes (RA, DEC, FREQ, STOKES), then FITSimage.wcstypes
   = ['lo','la', 'sp', 'li_STOKES'] and when we have an Annotatedimage
   object with axes (FREQ, DEC) then the axis permutation array is (3, 2) and
   the wcsypes list is ['sp', 'la'].
:type wcstypes:
   List of strings.
:param skyout:
   A so called sky definition (sky system, reference system, equinox)
   which is used to annotate the world coordinates and to draw
   graticule lines.
:type skyout:
   String
:param spectrans:
   The spectral translation. It sets the output system for spectral axes.
   E.g. a frequency axis can be labeled with velocities.
:type spectrans:
   String
:param alter:
   The alternative description of a world coordinate system. In a FITS header
   there is an alternative description of the world coordinate system if
   for each wcs related keyword, there is an alternative keyword which has
   a character appended to it (e.g. CRVAL1a, CDELT1a). The character that
   is appended is the one that need to be entered if one wants to use the
   alternate system. Note that this is only relevant to axis labeling and
   the plotting of graticules.
:type alter:
   Character (case insensitive)
:param mixpix:
   The axis number (FITS standard i.e. starts with 1) of the missing spatial axis for
   images with only one spatial axis (e.q. Position-Velocity plots).
:type mixpix:
   *None* or an integer
:param aspect:
   The aspect ratio. This value is used to correct the scale of the plot so
   that equal sizes in world coordinates (degrees) represent equal sizes
   in a plot. This aspect ratio is only useful for spatial maps. Its default
   value is 1.0. The aspect ratio is defined as:
   :math:`abs(cdelty/cdeltx)`. This value is automatically set in objects
   from :class:`FITSimage`
:param slicepos:
   Pixel coordinates used to slice the data
   in a data set with more than two axes. The pixel coordinates represent
   positions on the axes that do not belong to the image.
:type slicepos:
   Single value or tuple with integers
:param sliceaxnames:
   List with names of the axes outside the map. Assume we have a map of a RA-DEC
   map from a RA-DEC-FREQ cube, then *sliceaxnames* = ['FREQ']. Currently these
   names are not used.
:type sliceaxnames:
   List with strings
:param basename:
   Base name for new files on disk, for example to store a color map
   on disk. The default is supplied by method :meth:`FITSimage.Annotatedimage`.
:type basename:
   string
:param cmap:
   A colormap from class :class:`mplutil.VariableColormap` or a string
   that represents a colormap (e.g. 'jet', 'spectral' etc.).
:type cmap:
   mplutil.VariableColormap instance or string
:param blankcolor:
   Color of the undefined pixels (NaN's, blanks). It is a matplotlib color
   e.g. blankcolor='g'. In the display the color of the blanks can
   be changed with key 'B'. It loops throug a small set with predefined
   colors. Changing the colors of bad pixels helps to make them more visible.
:type blankcolor:
   Matplotlib color (e.g. 'c', '#aabb12')
:param clipmin:
   A value which sets the lower value of the interval between which the colors
   in the colormap are distributed. If omitted, the minimum data value will
   be *clipmin*.
:type clipmin:
   Float
:param clipmax:
   A value which sets the upper value of the interval between which the colors
   in the colormap are distributed. If omitted, the maximum data value will
   be *clipmin*.
:type clipmax:
   Float
:param boxdat:
   An 2dim. array with the same shape as the *boxdat* attribute of the
   input FITSimage object. 
:type boxdat:
   NumPy array
:param sourcename:
   Name of origin of data. By default set to 'unknownsource'
:type sourcename:
   String
:param gridmode:
   By default this value is set to False. Positions are written
   as pixel coordinates and input of coordinates as strings will
   parse numbers as pixel coordinates. If one sets this to True, then
   a system of grids is used. The relation between a pixel and a grid
   for axis *n* is:
   
   ``grid = pixel - NINT(CRPIXn)``

   Some (legacy) astronomy software packages use this system.
   Toolbar position information is written in grid coordinates and also numbers
   in the (string) input of positions are processed as grids.
   This option is useful when an interface is needed between the
   Kapteyn Package and another software package (e.g. GIPSY).
   Note that with positions as strings, we mean positions parsed
   with method :meth:`positions.str2pos`. These are not the
   positions which are described as **pixel** positions.
:type gridmode:
   Boolean


   
:Attributes:

    .. attribute:: alter

          Character that sets an alternate world coordinate system.

    .. attribute:: aspect

          Aspect ratio of a pixel according to the FITS header.
          For spatial maps this value is used to set and keep an
          image in the correct aspect ratio.

    .. attribute:: axperm

          Axis numbers of the two axis in this map. Axis numbers
          start with 1.

    .. attribute:: basename

          Name of data origin.

    .. attribute:: blankcolor

          Color of 'bad' pixels as a Matplotlib color.
          
    .. attribute:: box

          Coordinates of the plot box. In order to keep the entire pixel in the
          corners in the plot, one has to extend the values of *pxlim* and
          *pylim* with 0.5 pixel.

    .. attribute:: clipmin

          Value either entered or calculated, which scales the image data to the
          available colors. Clipmin is the minimum value.

    .. attribute:: clipmax

          Value either entered or calculated, which scales the image data to the
          available colors. Clipmax is the maximum value.

    .. attribute:: cmap

          The color map. This is an object from class :class:`mplutil.VariableColormap`.
          which is inherited from the Matplotlib color map class.

    .. attribute:: cmapinverse

          Boolean which store the status of the current colormap, standard or inverted.

    .. attribute:: data

          Image data. Other data containers are attibutes 'data_blur', 'data_hist',
          and 'data_orig'.

    .. attribute:: fluxfie

          Function or Lambda expression which can be used to scale the flux found with
          method *getflux()*. There must be two parameters in this function or
          expression: *a* for the area and *s* for the sum of the pixel values.
          E.g. ``Annotatedimage.fluxfie = lambda s, a: s/a``
          Note that there is no method to set this attribute.
          The attribute is used in the shapes module.

    .. attribute:: frame

          Matplotlib Axes instance where image and contours are plotted

    .. attribute:: gridmode

          Boolean that indicates when we work in pixel- or in grid coordinates.

    .. attribute:: hdr

          Header which is used to derive the world coordinate system for axis labels
          and graticule lines. The header is either a Python dictionary or a PyFITS
          header.

    .. attribute:: mixpix

          The pixel of the missing spatial axis in a Position-Velocity
          image.

    .. attribute:: objlist

          List with all plot objects for the current *Annotatedimage* object derived from classes:
          'Beam', 'Colorbar', 'Contours', 'Graticule', 'Image', 'Marker', 'Minortickmarks',
          'Pixellabels', 'RGBimage', 'Ruler', 'Skypolygon'

    .. attribute:: pixelstep

          The step size in pixels or fraction of pixels. This size is used to sample
          the area of an object. Used in the context of the shapes module.
          E.g. ``annim.pixelstep = 0.5;``

    .. attribute:: pixoffset

          Tuple with two offsets in pixels used to distinguish a pixel coordinate system
          from a grid coordinate system.

    .. attribute:: projection

          An object from the Projection class as defined in module :mod:`wcs`

    .. attribute:: ptype

          Each object in the object list has an attribute which describes the (plot) type
          of the object. The ptype of an Annotatedimage is *Annotatedimage*.

    .. attribute:: pxlim

          Pixel limits in x = (xlo, xhi)

    .. attribute:: pylim

          Pixel limits in y = (ylo, yhi)

    .. attribute:: rgbs

          Boolean which is set to True if the current image is composed of three images
          each representing one color.
          
    .. attribute:: sliceaxnames

          A list with axis names that are not part of the current image, but
          are part of the data structure from which the current Annotated image data
          is extracted.

    .. attribute:: skyout

          The sky definition for which graticule lines are plotted
          and axis annotation is made (e.g. "Equatorial FK4")

    .. attribute:: spectrans

          The translation code to transform native spectral coordinates
          to another system (e.g. frequencies to velocities)


    .. attribute:: slicepos

          Single value or tuple with more than one value representing
          the pixel coordinates on axes in the original data structure
          that do not belong to the image. It defines how the data slice
          is ectracted from the original.
          The order of these 'outside' axes is copied from the (FITS) header.

    .. attribute:: wcstypes
    
          Type of the axes in this data. The order is the same as of the axes.
          The types ara strings and are derived from attribute wcstype of the
          Projection object. The types are:
          'lo' is longitude axis. 'la' is latitude axis,
          'sp' is spectral axis. 'li' is a linear axis. Appended to 'li' is an
          underscore and the ctype of that axis (e.g. 'li_stokes').

:Methods:

.. automethod:: set_norm
.. automethod:: set_colormap
.. automethod:: write_colormap
.. automethod:: set_blankcolor
.. automethod:: set_aspectratio
.. automethod:: get_colornavigation_info
.. automethod:: Image
.. automethod:: RGBimage
.. automethod:: Contours
.. automethod:: Colorbar
.. automethod:: Graticule
.. automethod:: Pixellabels
.. automethod:: Minortickmarks
.. automethod:: Beam
.. automethod:: Marker
.. automethod:: Ruler
.. automethod:: Skypolygon
.. automethod:: plot
.. automethod:: toworld
.. automethod:: topixel
.. automethod:: inside
.. automethod:: histeq
.. automethod:: blur
.. automethod:: interact_toolbarinfo
.. automethod:: interact_imagecolors
.. automethod:: interact_writepos
.. automethod:: positionsfromfile

   """
#--------------------------------------------------------------------
   # Class variables that set the lut scales and keys
   lutscales = ['linear', 'log', 'exp', 'sqrt', 'square']
   scalekeys = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
   scales_default = 0
   blankcols = ['w', 'k', 'y', 'm', 'c', 'r', 'g', 'b']
   blanknames = ['White', 'Black', 'Yellow', 'Magenta', 'Cyan', 'Red', 'Green', 'Blue']
   blankcols_default = 0

   # Class variables for scaling the colormap
   slopetrans = 4.0   # Scale mouse value between 0 and 1 to colormap slope
   shifttrans = 0.5   # Translate mouse value between 0 and 1 to -0.5, 0.5
   
   def __init__(self, frame, header, pxlim, pylim, imdata, projection, axperm, wcstypes,
                skyout, spectrans, alter='',
                mixpix=None,  aspect=1, slicepos=None, sliceaxnames=None, basename=None,
                cmap=None, blankcolor='w', clipmin=None, clipmax=None, boxdat=None,
                sourcename='unknownsource', gridmode=False, 
                adjustable='box', anchor='C', newaspect=None, clipmode=0, clipmn=(4,5),
                callbackslist={}):

      #-----------------------------------------------------------------
      """
      """
      #-----------------------------------------------------------------
      self.ptype = "Annotatedimage"
      self.sourcename = sourcename
      self.hdr = header
      self.projection = projection
      self.pxlim = pxlim
      self.pylim = pylim
      if boxdat != None:
         # shp = (pylim[1]-pylim[0]+1, pxlim[1]-pxlim[0]+1)
         #shp = imdata.shape
         #if boxdat.shape != shp:
         #   raise ValueError("The shape of 'boxdat' is not (%d,%d)" %(shp[0],shp[1]))
         self.data = boxdat
      else:
         self.data = imdata
      self.mixpix = mixpix
      self.axperm = axperm
      self.wcstypes = wcstypes
      self.skyout = skyout
      self.spectrans = spectrans
      self.alter = alter
      if self.pxlim[0] <= self.pxlim[1]:
         deltax = 0.5
      else:
         deltax = -0.5
      if self.pylim[0] <= self.pylim[1]:
         deltay = 0.5
      else:
         deltay = -0.5
      self.box = (self.pxlim[0]-deltax, self.pxlim[1]+deltax, self.pylim[0]-deltay, self.pylim[1]+deltay)
      self.image = None                          # A Matplotlib instance made with imshow()
      self.aspect = aspect
      if not newaspect is None:
         self.aspect = newaspect
      self.adjustable = adjustable
      self.anchor = anchor
      self.slicepos = slicepos                   # Information about current slice
      self.gridmode = gridmode
      self.sliceaxnames = sliceaxnames
      self.contours = None
      self.colorbar = None
      self.contourset = None
      if cmap is None:
         cmap = cmlist.cmap_default
      self.objlist = []
      self.frame = self.adjustframe(frame, adjustable)
      self.figmanager = plt_get_current_fig_manager()
      self.messenger = None
      self.toolbarkey = None
      # Keys of class dictionary with callbacks are: slope, offset, ...
      self.callbackslist = callbackslist
      if callbackslist.has_key('exmes'):
         self.externalmessenger = callbackslist['exmes']
      else:
         self.externalmessenger = None
      # The next lines need some explanation. The toolbar message
      # is bound to a figure(manager). If a figure changes, then we need
      # new toolbar message function. For multiple frames in one figure
      # we want to use the same toolbar message function.
      # To disable Matplotlib interference with this function we
      # set the function to None with lambda x: None
      # This makes it necessary to maintain two global variables.
      # One to keep track of the figure(manager) and one that
      # stores the last message function (the messenger).
      global globalfigmanager, globalmessenger
      if not (self.figmanager is globalfigmanager):
         try: # Sphinx does something with the figure manager, so we need a try
            globalfigmanager = self.figmanager
            if not self.externalmessenger is None:
               globalmessenger = self.externalmessenger
            else:
               globalmessenger = self.figmanager.toolbar.set_message
            self.figmanager.toolbar.set_message=lambda x: None
         except:
           pass
      self.messenger = globalmessenger

      # In the constructor, the colormap should not update the image
      # otherwise we get incompatible frames which behave differently
      # if we resize the plot window. 

      self.autostate = None
      try:
         self.autostate = cmap.auto
         cmap.auto = False
      except:
         pass

      self.set_colormap(cmap)
      self.set_blankcolor(blankcolor)

      # Calculate defaults for clips if nothing is given
      # If no clip values are entered, then we calculate them either from
      # the minimum and maximum values of the current data (self.data) or
      # we use mean-n*rms and mean+m*rms as the default. This depends on the
      # option given in parameter 'clipmode'.
      needstats = False
      datmin = datmax = mean = rms = None
      if clipmode == 0:
         if None in [clipmin, clipmax]:
            needstats = True
      else:
         needstats = True
      if needstats:
         datmin, datmax, mean, rms = self.get_stats()
      
      if clipmin is None:
         if clipmode in [0, 1]:
            clipmin = datmin
         else:
            if None in [mean, rms]:
               clipmin = None
            else:
               # Clipmode 2 uses mean and rms
               clipmin = mean - rms*clipmn[0]
      if clipmax is None:
         if clipmode in [0, 1]:
            clipmax = datmax
         else:
            if None in [mean, rms]:
               clipmax = None
               # Clipmode 2 uses mean and rms
            else:
               clipmax = mean + rms*clipmn[1]
      flushprint("I calculated clip min max=%s %s %d"%(str(clipmin), str(clipmax), clipmode))
      self.clipmin = clipmin
      self.clipmax = clipmax
      self.datmin = datmin
      self.datmax = datmax
      self.mean = mean
      self.rms = rms

      # Give defaults if clips are still None:
      if self.clipmin is None:
         self.clipmin = 0.0
      if self.clipmax is None:
         self.clipmax = self.clipmin + 1.0

      self.AxesCallback_ids = []                 # A list with 'mpl connect id's. Used for disconnecting
      self.norm = Normalize(vmin=self.clipmin, vmax=self.clipmax, clip=True)
      self.histogram = False                     # Is current image equalized?
      self.data_hist = None                      # There is not yet a hist. eq. version of the data
      self.blurred = False
      self.data_blur = None
      self.blurfac = (self.pxlim[1]-self.pxlim[0]+1)/200.0
      self.blurindx = 0
      self.X_lastvisited = None
      self.Y_lastvisited = None
      self.data_orig = self.data                 # So we can toggle between image versions
      if basename is None:
         self.basename = "Unknown"               # Default name for file with colormap lut data
      else:
         self.basename = basename
      self.pixelstep = 1.0                       # Sub divide pixel for flux
      self.fluxfie = lambda s, a: s/a
      annotatedimage_list.append(self)
      self.pixoffset = [0.0, 0.0]                # Use this offset to display pixel positions
      if self.gridmode:
         self.set_pixoffset(nint(self.projection.crpix[0]), nint(self.projection.crpix[1]))
      self.rgbs = None                           # If not None, then the image is a
                                                 # composed image with r,g & b components.
      # Restore the update status of the colormap
      if not self.autostate is None:
         cmap.auto = self.autostate
         self.cmap.auto = self.autostate

      # An Annotatedimage object could be an object for a slice panel.
      # Then it has its own callback for a motion_notify event to
      # update a plot.
      self.regcb = None
      #if regcallback:
      #   self.regcb = AxesCallback(regcallback, self.frame, 'motion_notify_event')
      #   self.AxesCallback_ids.append(self.regcb)


   def callback(self, cbid, *arg):
      #-----------------------------------------------------------------
      """
      Helper function for registered callbacks
      """
      #-----------------------------------------------------------------
      if self.callbackslist.has_key(cbid):
         self.callbackslist[cbid](*arg)


   def get_stats(self):
      #-----------------------------------------------------------------
      """
      Get necessary statistics e.g. to set clip levels
      """
      #-----------------------------------------------------------------      
      if self.data is None:
         return None, None, None, None

      # If the data is copied from the FITSobject then
      # it does not contain -inf and inf values, because these are
      # replaced by NaN's. If this class is used with an external image
      # you should replace these values in that image data before creating an object
      # otherwise Matplotlib will fail to plot anything (image, contours).
      # If somehow the inf values still exist, then we still want to see
      # an image and therefore discard inf, -inf and nan to find the clip values.
      datmin = datmax = mean = rms = mask = None
      validmask = False
      datmin = numpy.nanmin(self.data)       # Fast method first
      if not numpy.isfinite(datmin):         # value is inf or -inf, try again...
         mask = numpy.isfinite(self.data)    # We need a mask
         validmask = numpy.any(mask)         # At least one must be a valid number
         if validmask:                      
            datmin = float(self.data[mask].min())
         else:
            datmin = None         
      if validmask:                          # We are sure that there are inf's
         if numpy.any(mask):                 # At least one must be a valid number
            datmax = float(self.data[mask].max())
         else:
            datmax = None
      else:      
         datmax = numpy.nanmax(self.data)    # The fast method

      if validmask:
         mean = float(self.data[mask].mean())
         rms  = float(self.data[mask].std())
      else:
         # Here we know that there are no inf's, but there could be still NaN's
         mean = float(self.data.mean())
         rms  = float(self.data.std())      
         if not (numpy.isfinite(rms) and numpy.isfinite(mean)):
            mask = numpy.isfinite(self.data)
            if numpy.any(mask):                 # At least one must be a valid number
               mean = float(self.data[mask].mean())
               rms  = float(self.data[mask].std())

      
      flushprint("Calculated min, max=%f %f"%(datmin, datmax))
      flushprint("Calculated std, mean=%f %f"%(rms, mean))
                  
      return datmin, datmax, mean, rms


      
   def set_pixoffset(self, xoff=0.0, yoff=0.0):
      #-----------------------------------------------------------------
      """
      For conversions from FITS to other pixel based coordinate systems.
      """
      #-----------------------------------------------------------------
      self.pixoffset[0] = xoff
      self.pixoffset[1] = yoff
      

   def set_norm(self, clipmin=None, clipmax=None):
      #-----------------------------------------------------------------
      """
      Matplotlib scales image values between 0 and 1 for its distribution
      of colors from the color map. With this method we set the image values
      which we want to scale between 0 and 1. The default image values
      are the minimum and maximum of the data in :attr:`Annotatedimage.data`.
      If you want to inspect a certain range of data values you need more
      colors in a smaller intensity range, then use different *clipmin*
      and *clipmax* in the constructor of :class:`Annotatedimage` or
      in this method.


      :param clipmin:
         Image data below this threshold will get the same color
         Value None will be replaced by 'clipmin'.
      :type clipmin:
         Float
      :param clipmax:
         Image data above this threshold will get the same color
         Value None will be replaced by 'clipmax'.
      :type clipmax:
         Float

      :Examples:

         >>> fitsobj = maputils.FITSimage("m101.fits")
         >>> annim = fitsobj.Annotatedimage(frame, cmap="spectral")
         >>> annim.Image(interpolation='spline36')
         >>> annim.set_norm(10000, 15500)

         or:

         >>> fitsobj = maputils.FITSimage("m101.fits")
         >>> annim = fitsobj.Annotatedimage(frame, cmap="spectral", clipmin=10000, clipmax=15500)

      :notes:
      
         It is also possible to change the norm after an image has been displayed.
         This enables a programmer to setup key interaction for
         changing the clip levels in an image for example when the default clip levels
         are not suitable to inspect a certain data range.
         Usually the color editing (with :meth:`Annotatedimage.interact_imagecolors`)
         can do this job very well so we think there is not much demand in a scripting
         environment. With GUI's it will be different.
      """
      #-----------------------------------------------------------------
      #if clipmin is None and clipmax is None:
      #   # Nothing to do
      #   return
      if clipmin is None:
         clipmin = self.clipmin
      if clipmax is None:
         clipmax = self.clipmax
      if clipmin > clipmax:
         clipmin, clipmax = clipmax, clipmin  # Swap, to prevent ValueError
      self.norm = Normalize(vmin=clipmin, vmax=clipmax, clip=True)
      #self.clipmin = clipmin; self.clipmax = clipmax
      if self.cmap != None:
         self.cmap.update()
      if self.image != None:
         self.image.norm = self.norm
         if self.image.im != None:
            self.image.im.set_clim(clipmin, clipmax)
      if self.colorbar != None:
         #self.colorbar.cb.set_norm(self.norm)
         self.colorbar.cb.set_clim(clipmin, clipmax)


   def set_colormap(self, cmap, blankcolor=None):
      #-----------------------------------------------------------------
      """
      Method to set the initial color map for images, contours and colorbars.
      These color maps are either strings (e.g. 'jet' or 'spectral')
      from a list with Matplotlib color maps or it is a path to
      a color map on disk (e.g. cmap="/home/user/luts/mousse.lut").
      If the color map is not found in the list with known color maps
      then it is added to the list, which is a global variable called
      *cmlist*.

      The Kapteyn Package has also a list with useful color maps. See
      example below or example 'mu_luttest.py' in the
      :doc:`maputilstutorial`.
      
      If you add the event handler *interact_imagecolors()* it is
      possible to change colormaps with keyboard keys and mouse.

      :param cmap:
         The color map to be used for image, contours and colorbar
      :type cmap:
         String or instance of VariableColormap
      :param blankcolor:
         Color of the bad pixels in your image.
      :type blankcolor:
          Matplotlib color

      :Examples:

         >>> fitsobj = maputils.FITSimage("m101.fits")
         >>> annim = fitsobj.Annotatedimage(frame, clipmin=10000, clipmax=15500)
         >>> annim.set_colormap("spectral")

         or use the constructor as in:

         >>> annim = fitsobj.Annotatedimage(frame, cmap="spectral", clipmin=10000, clipmax=15500)

         Get extra lookup tables from Kapteyn Package (by default, these
         luts are appended at creation time of cmlist)
      
         >>> extralist = mplutil.VariableColormap.luts()
         >>> maputils.cmlist.add(extralist)

      """
      #-----------------------------------------------------------------
      if cmap is None:
         cmap = cmlist.cmap_default
      if isinstance(cmap, Colormap):
         self.cmap = cmap                           # Either a string or a Colormap instance
         # What to do with the index. This is not a string from the list.
         self.cmindx = 0
      elif type(cmap) == types_StringType:
         try:
            # Is this colormap registered in our colormap list?
            self.cmindx = cmlist.colormaps.index(cmap)
         except:
            # then register it now
            cmlist.add(cmap)
            self.cmindx = cmlist.colormaps.index(cmap)
         self.cmap = VariableColormap(cmap)
      else:
         raise Exception, "Color map is not of type Colormap or string"
      if self.image != None:
         self.cmap.set_source(cmap)
      
      self.startcmap = self.cmap         # This could be one that is not in the list with color maps
      #self.startcmindx = self.cmindx     # Use the start color map if a reset is requested
      self.startcmindx = cmlist.colormaps.index(cmlist.cmap_default)
      self.cmapinverse = False
      if blankcolor != None:
         self.set_blankcolor(blankcolor)
      try:
         self.cmap.auto = False
      except:
         pass


   def write_colormap(self, filename):
      #-----------------------------------------------------------------
      """
      Method to write current colormap rgb values to file on disk.
      If you add the event handler *interact_imagecolors()*, this
      method is automatically invoked if you press key 'm'.

      This method is only useful if the colormap changes i.e.
      in an interactive environment.
      """
      #-----------------------------------------------------------------
      #print self.cmap.name
      #print self.cmap._lut[0:-3,0:3]
      a = tabarray(self.cmap._lut[0:-3,0:3])
      a.writeto(filename)


   def set_blankcolor(self, blankcolor, alpha=1.0):
      #-----------------------------------------------------------------
      """
      Set the color of bad pixels.
      If you add the event handler *interact_imagecolors()*, this
      method steps through a list of colors for the bad pixels in an
      image.

      :param blankcolor:
         The color of the bad pixels (blanks) in your map
      :type blankcolor:
         Matplotlib color
      :param alpha:
         Make the color of bad pixels transparent with *alpha < 1*
      :type alpha:
         Float in interval [0,1]
         
      :Example:

         >>> annim.set_blankcolor('c')
      """
      #-----------------------------------------------------------------
      self.cmap.set_bad(blankcolor, alpha)
      self.blankcol = blankcolor

      
   def set_aspectratio(self, aspect):
      #-----------------------------------------------------------------
      """
      Set the aspect ratio. Overrule the default aspect ratio which corrects
      pixels that are not square in world coordinates. Can be useful if you
      want to stretch images for which the aspect ratio doesn't matter
      (e.g. XV maps).

      :param aspect:
         The aspect ratio is defined as *pixel height / pixel width*.
         With this value one can stretch an image in x- or y direction.
         The default is such that 1 arcmin in x has the same length
         in cm as 1 arcmin in y.
      :type aspect:
         Float
         
      :Example:

         >>> annim = fitsobj.Annotatedimage(frame)
         >>> annim.set_aspectratio(1.2)
      """
      #-----------------------------------------------------------------
      if aspect <= 0.0:
         raise ValueError("Only aspect ratios > 0 are allowed")
      self.aspect = abs(aspect)
      self.frame.set_aspect(aspect=self.aspect, adjustable=self.adjustable, anchor=self.anchor)


   def adjustframe(self, frame, adjustable='box'):
      #-----------------------------------------------------------------
      """
      Method to change the frame for the right aspect ratio and
      how to react on a resize of the plot window.
      """
      #-----------------------------------------------------------------
      frame.set_aspect(aspect=self.aspect, adjustable=adjustable, anchor=self.anchor)
      frame.set_autoscale_on(False)
      frame.xaxis.set_visible(False)
      frame.yaxis.set_visible(False)
      frame.set_xlim((self.box[0], self.box[1]))   # Initialize in case no objects are created
      frame.set_ylim((self.box[2], self.box[3]))   # then we still can navigate with the mouse
      return frame


   def get_colornavigation_info(self):
      #-----------------------------------------------------------------
      """
      This method compiles and returns a help text for
      color map interaction.
      """
      #-----------------------------------------------------------------
      helptext = colornavigation_info()
      return helptext


   def blur(self, nx, ny=None):
      #-----------------------------------------------------------------
      """
      Blur the image by convolving with a gaussian kernel of typical
      size nx (pixels). The optional keyword argument ny allows for a different
      size in the y direction.
      nx, ny are the sigma's for the gaussian kernel.
      """
      #-----------------------------------------------------------------
      if ny is None:
         ny = nx
      if self.data is None:       # Current data
         raise Exception, "Cannot plot image because image data is not available!"
      if self.data_blur is None:  # Blurred data
         self.data_blur = numpy.zeros(self.data.shape) # Prevent byte order problems
      gaussian_filter(self.data_orig, sigma=(nx,ny), order=0, output=self.data_blur, mode='reflect', cval=0.0)

       
   def histeq(self, nbr_bins=256):
      #-----------------------------------------------------------------
      """
      Create a histogram equalized version of the data.
      The histogram equalized data is stored in attribute *data_hist*.
      """
      #-----------------------------------------------------------------
      if self.data is None:
         raise Exception, "Cannot plot image because image data is not available!"
      # Algorithm by Jan Erik Solem
      im = self.data
      ix = numpy.isfinite(im)
      #im = numpy.ma.masked_where(numpy.isfinite(self.data), self.data)
      #get image histogram
      imhist,bins = numpy.histogram(im[ix].flatten(), nbr_bins, normed=True,
                                    range=(self.clipmin, self.clipmax))
      cdf = imhist.cumsum() #cumulative distribution function
      #cdf = 255 * cdf / cdf[-1] #normalize
      cdf = cdf / cdf[-1] #normalize
      #use linear interpolation of cdf to find new pixel values
      im2 = numpy.interp(im.flatten(),bins[:-1],cdf)
      # Rescale between clip levels of original data
      m = (self.clipmax-self.clipmin) / (cdf[-1]-cdf[0])
      im2 = m * (im2-cdf[0]) + self.clipmin
      self.data_hist = im2.reshape(im.shape)


   def Image(self, **kwargs):
      #-----------------------------------------------------------------
      """
      Setup of an image.
      This method is a call to the constructor of class :class:`Image` with
      a default value for most of the keyword parameters.
      This created object has an attribute *im* which is
      an instance of Matplotlib's *imshow()* method.
      This object has a plot method. This method is used by the
      more general :meth:`Annotatedimage.plot` method.

      :param kwargs:
         From the documentation
         of Matplotlib we learn that for method *imshow()* (used in the plot method
         if an Image) a few interesting keyword arguments remain:

         * *interpolation* - From Matplotlib's documentation: Acceptable values are
           None, 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
           'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel',
           'mitchell', 'sinc', 'lanczos'
         * *visible* - Switch the visibility of the image
         * *alpha* - Value between 0 and 1 which sets the transparency of the image.

      :type kwargs:
         Python keyword parameters

      :Attributes:

         .. attribute:: im

         The object generated after a call to Matplotlib's *imshow()*.

      :Examples:

         >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
         >>> annim = fitsobject.Annotatedimage()
         >>> annim.Image(interpolation="spline36")

         or create an image but make it invisible:

         >>> annim.Image(visible=False)
      """
      #-----------------------------------------------------------------
      if self.image != None:
         raise Exception, "Only 1 image allowed per Annotatedimage object"
      image = Image(self.data, self.box, self.cmap, self.norm, **kwargs)
      self.objlist.append(image)
      self.image = image
      return image


   def RGBimage(self, f_red, f_green, f_blue, fun=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Matplotlib's method imshow() is able to produce RGB images.
      To create a real RGB image, we need three arrays with identical
      shape representing the red, green and blue components.
      Method imshow() requires data scaled between 0 and 1.

      This utility method prepares a composed and scaled data array
      derived from three FITSimage objects.
      It scales the composed array and not the individual image arrays.
      The method allows for a function or lambda expression to
      be entered to process the scaled data.
      The world coordinate system (e.g. to plot graticules) is copied
      from the current :class:`Annotatedimage` object. Note that for the
      three images only the shape of the array must be equal to the
      shape of the data of the current :class:`Annotatedimage` object.


      :param f_red:
         This object describes a two dimensional data structure which represents
         the red part of the composed image.
      :type f_red:
         Object from class :class:`FITSimage`
      :param f_green:
         This object describes a two dimensional data structure which represents
         the green part of the composed image.
      :type f_green:
         Object from class :class:`FITSimage`
      :param f_blue:
         This object describes a two dimensional data structure which represents
         the blue part of the composed image.
      :type f_blue:
         Object from class :class:`FITSimage`
      :param fun:
         A function or a Lambda expression to process the scaled data.
      :type fun:
         Function or Lambda expression
      :param kwargs:
         See description method :meth:`Annotatedimage.Image`.
      :type kwargs:
         Python keyword parameters

      :Note:
         A RGB image does not interact with a colormap. Interacting with a colormap
         (e.g. after adding annim.interact_imagecolors() in the example below)
         is not forbidden but it gives weird results. To rescale the data, for
         instance for a better view of interesting data, you need to enter a
         function or Lambda expression with parameter *fun*.

      :Example:

       ::
      
         from kapteyn import maputils
         from numpy import sqrt
         from matplotlib import pyplot as plt
         
         f_red = maputils.FITSimage('m101_red.fits')
         f_green = maputils.FITSimage('m101_green.fits')
         f_blue = maputils.FITSimage('m101_blue.fits')
         
         fig = plt.figure()
         frame = fig.add_subplot(1,1,1)
         annim = f_red.Annotatedimage(frame)
         annim.RGBimage(f_red, f_green, f_blue, fun=lambda x:sqrt(x), alpha=0.5)
         
         grat = annim.Graticule()
         annim.interact_toolbarinfo()
         
         maputils.showall()

      """
      #-----------------------------------------------------------------
      if self.image != None:
         raise Exception, "Only 1 image allowed per Annotatedimage object"
      if f_red.boxdat.shape != self.data.shape:
         raise Exception, "Shape of red image is not equal to shape of Annotatedimage object!"
      if f_green.boxdat.shape != self.data.shape:
         raise Exception, "Shape of green image is not equal to shape of Annotatedimage object!"
      if f_blue.boxdat.shape != self.data.shape:
         raise Exception, "Shape of blue image is not equal to shape of Annotatedimage object!"
      # Compose a new array. Note that this syntax implies a real copy of the data.
      rgbim = numpy.zeros((self.data.shape+(3,)))
      rgbim[:,:,0] = f_red.boxdat
      rgbim[:,:,1] = f_green.boxdat
      rgbim[:,:,2] = f_blue.boxdat
      # Scale the composed array to values between 0 and 1
      dmin = rgbim.min()
      dx = rgbim.max() - dmin
      if dx == 0.0:
         dx = 1.0
      rgbim = (rgbim - dmin)/dx
      # Apply the function or lambda expression if any is given, to
      # rescale the data.
      if not fun is None:
         rgbim = fun(rgbim)
      image = Image(rgbim, self.box, self.cmap, self.norm, **kwargs)
      self.objlist.append(image)
      self.image = image
      # Set a flag to indicate that this Annotatedimage object has a rgb
      # data as image. We store the FITSimage objects as an attribute.
      # We need this data later if we wnat to display the image values of the
      # three maps in an informative message.
      self.rgbs = (f_red.boxdat, f_green.boxdat, f_blue.boxdat)
      return image


   def Contours(self, levels=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Setup of contour lines.
      This method is a call to the constructor of class :class:`Contours`
      with a number of default parameters.
      Either it plots single contour lines or a combination
      of contour lines and filled regions between the contours.
      The colors are taken from the current colormap.

      :param levels:
         Image values for which contours must be plotted. The default
         is *None* which results in a list with values calculated
         by the Contour constructor.
      :type levels:
         None or a list with floating point numbers
      :param kwargs:
         There are a number of keyword arguments that
         are useful:
       
         * *filled* - if set to True the area between two contours will
           get a color (close to the color of the contour.
         * negative - one of Matplotlib's line styles
           'solid', 'dashed', 'dashdot', 'dotted' in which contours are
           plotted which represent negative image values.
         * *colors* - If None, the current colormap will be used.
           If a character or string, all levels will be plotted in this color.
           If a tuple of matplotlib colors then different levels will be
           plotted in different colors in the order specified.
         * *linewidths* - If a number, all levels will be plotted with this linewidth.
           If a tuple, different levels will be plotted with different
           linewidths in the order specified
         * *linestyles* - One of 'solid', 'dashed', 'dashdot', 'dotted', which
           sets the style of the contour. It can also be given in a list. See the
           Matplotlib documentation for its behaviour.
      :type kwargs:
         Python keyword parameters

      :Methods:

         * :meth:`Contours.setp_contour` - Set properties for individual contours
         * :meth:`Contours.setp_label` - Plot labels for individual contours

      :Examples:

         >>> fitsobj = maputils.FITSimage("m101.fits")
         >>> annim = fitsobj.Annotatedimage()
         >>> annim.Image(alpha=0.5)
         >>> cont = annim.Contours()
         >>> print "Levels=", cont.clevels
         Levels= [  4000.   6000.   8000.  10000.  12000.  14000.]

         >>> annim.Contours(filled=True)

         In the next example note the plural form of the standard Matplotlib
         keywords. They apply to all contours:
         
         >>> annim.Contours(colors='w', linewidths=2)

         Set levels and the line style for negative contours:
      
         >>> annim.Contours(levels=[-500,-300, 0, 300, 500], negative="dotted")

         A combination of keyword parameters with less elements than
         the number of contour levels:
      
         >>> cont = annim.Contours(linestyles=('solid', 'dashed', 'dashdot', 'dotted'),
                                   linewidths=(2,3,4), colors=('r','g','b','m'))

         Example of setting of properties for all and 1 contour with
         *setp_contour()*:
      
         >>> cont = annim.Contours(levels=range(10000,16000,1000))
         >>> cont.setp_contour(linewidth=1)
         >>> cont.setp_contour(levels=11000, color='g', linewidth=3)

         Plot a (formatted) label near a contour with *setp_label()*:
      
         >>> cont2 = annim.Contours(levels=(8000,9000,10000,11000))
         >>> cont2.setp_label(11000, colors='b', fontsize=14, fmt="%.3f")
         >>> cont2.setp_label(fontsize=10, fmt="$%g \lambda$")
      """
      #-----------------------------------------------------------------
      contourset = Contours(self.data, self.box, levels, cmap=self.cmap, norm=self.norm, **kwargs)
      self.objlist.append(contourset)
      self.contourset = contourset
      return contourset


   def Colorbar(self, frame=None, clines=False, **kwargs):
      #-----------------------------------------------------------------
      """
      This method is a call to the constructor of class :class:`Colorbar`
      with a number of default parameters.
      A color bar is an image which represents the current color scheme.
      It annotates the colors with image values so that it is possible
      to get an idea of the distribution of the values in your image.

      :param frame:
         By default a colorbar will 'steal' some space from its parent frame
         but this behaviour can be overruled by setting an explicit frame (Matplotlib Axes object).
      :type frame:
         Matplotlib Axes object
      :param clines:
          If set to true AND a contour set (an :meth:`Annotatedimage.Contours` object)
          is available, then lines will be plotted in the colorbar
          at positions that correspond to the contour levels
      :type clines:
          Boolean
      :param kwargs:
          Specific keyword arguments and Keyword arguments for Matplotlib's method *ColorbarBase()*

          * *label* - A text that will be plotted along the long axis of the colorbar.
          * *linewidths* - One number that sets the line width of all the contour lines
            in the colorbar.

          From Matplotlib:
             
          * *orientation* - 'horizontal' or 'vertical'
          * *fontsize* - Size of numbers along the colorbar
          * *ticks* - Levels which are annotated along the colorbar
          * *visible* - Make image in colorbar invisible
           
      :type kwargs:
          Python keyword arguments

      :methods:
          :meth:`Colorbar.set_label` - Plot a title along the long side of the colorbar.

      :Examples:

          A basic example were the font size for the ticks are set:
             
          >>> fitsobj = maputils.FITSimage("m101.fits")
          >>> annim = fitsobj.Annotatedimage(cmap="spectral")
          >>> annim.Image()
          >>> colbar = annim.Colorbar(fontsize=8)
          >>> annim.plot()
          >>> plt.show()

          Set frames for Image and Colorbar:

          >>> frame = fig.add_axes((0.1, 0.2, 0.8, 0.8))
          >>> cbframe = fig.add_axes((0.1, 0.1, 0.8, 0.1))
          >>> annim = fitsobj.Annotatedimage(cmap="Accent", clipmin=8000, frame=frame)
          >>> colbar = annim.Colorbar(fontsize=8, orientation='horizontal', frame=cbframe)

          Create a title for the colorbar and change its font size:

          >>> units = r'$ergs/(sec.cm^2)$'
          >>> colbar.set_label(label=units, fontsize=24)
      """
      #------------------------------------------------------------------
      if self.colorbar != None:
         raise Exception, "Only 1 colorbar allowed per Annotatedimage object"
      colorbar = Colorbar(self.cmap, frame=frame, norm=self.norm, contourset=self.contourset, clines=clines, **kwargs)
      self.objlist.append(colorbar)
      self.colorbar = colorbar
      return colorbar


   def Graticule(self, visible=True, pxlim=None, pylim=None, **kwargs):
      #-----------------------------------------------------------------
      """
      This method is a call to the constructor of class :class:`wcsgrat.Graticule`
      with a number of default parameters.
      
      It calculates and plots graticule lines of constant longitude or constant
      latitude. The description of the parameters is found in
      :class:`wcsgrat.Graticule`. An extra parameter is *visible*.
      If visible is set to False than we can plot objects derived from this
      class such as 'Rulers' and 'Insidelabels' without plotting unwanted
      graticule lines and labels.

      :Methods:

         * :meth:`wcsgrat.Graticule.Ruler`
         * :meth:`wcsgrat.Graticule.Insidelabels`

      Other parameters such as *hdr*, *axperm*, *pxlim*, *pylim*, *mixpix*,
      *skyout* and *spectrans* are set to defaults in the context of this method
      and should not be overwritten.

      :Examples:

         >>> fitsobj = maputils.FITSimage('m101.fits')
         >>> annim = fitsobj.Annotatedimage()
         >>> grat = annim.Graticule()
         >>> annim.plot()
         >>> plt.show()

         Set the range in world coordinates and set the positions for
         the labels with (X, Y):
         
         >>> X = arange(0,360.0,15.0)
         >>> Y = [20, 30,45, 60, 75, 90]
         >>> grat = annim.Graticule(wylim=(20.0,90.0), wxlim=(0,360), 
                                    startx=X, starty=Y)

         Add a ruler, based on the current Annotatedimage object:
         
         >>> ruler3 = annim.Ruler(23*15,30,22*15,15, 0.5, 1, world=True,
                                 fmt=r"$%4.0f^\prime$",
                                 fun=lambda x: x*60.0, addangle=0)
         >>> ruler3.setp_labels(color='r')

         Add world coordinate labels inside the plot. Note that these are
         derived from the current Graticule object.
         
         >>> grat.Insidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10,
                               color='r', ha='right')

      """
      #-----------------------------------------------------------------
      class Gratdata(object):
         
         def __init__(self, hdr, axperm, wcstypes, pxlim, pylim, mixpix, skyout, spectrans, alter):
            self.hdr = hdr
            self.axperm = axperm
            self.pxlim = pxlim
            self.pylim = pylim
            self.mixpix = mixpix
            self.skyout = skyout
            self.spectrans = spectrans
            self.alter = alter
            self.wcstypes = wcstypes

      # One can overrule the pixel limits of the Annotatedimage object
      if pxlim is None:
         pxlim = self.pxlim
      if pylim is None:
         pylim = self.pylim
       
      gratdata = Gratdata(self.hdr, self.axperm, self.wcstypes, pxlim, pylim,
                          self.mixpix, self.skyout, self.spectrans, self.alter)
      graticule = wcsgrat.Graticule(graticuledata=gratdata, **kwargs)
      graticule.visible = visible     # A new attribute only for this context
      self.objlist.append(graticule)
      return graticule



   def Minortickmarks(self, graticule, partsx=10, partsy=10, **kwargs):
      #-------------------------------------------------------------
      """
      Drawing minor tick marks is as easy or as difficult as finding
      the positions of major tick marks. Therefore we decided that
      the best way to draw minor tick marks is to calculate a new
      (but invisible) graticule object. Only the tick marks are visible.

      :param graticule:
         Graticule object from which we change the step size
         between tick marks to create a new Graticule object for which
         most components (graticule lines, labels, ...) are made invisible.
      :type graticule:
         Object from class :class:`wcsgrat.Graticule`
      :param partsx:
         Divide major tick marks in this number of parts.
         This method forces this number to be an integer between 2 and 20
         If the input is None then nothing is plotted. For example for maps with
         only one spatial axis, one can decide to plot tick marks for only
         one of the axes.
      :type partsx:
         Integer or None
      :param partsx:
         See description for parameter *partsx*
      :type partsx:
         Integer or None
      :param kwargs:
         Parameters for changing the attributes of the tick mark symbols.
         Useful keywords are *color*, *markersize* and *markeredgewidth*.
      :type kwargs:
         Matplotlib keyword arguments related to a Line2D object.
   
      :Notes:
   
         Minor tick marks are also plotted at the positions of major tick marks.
         By default this will not be visible. It is visible if you user
         a longer marker size, a different color or a marker with increased
         width.
   
         The default marker size is set to 2.
   
      :Returns:
   
         This method returns a graticule object. Its properties can be
         changed in the calling environment with the appropriate methods
         (e.g. :meth:`wcsgrat.Graticule.setp_tickmark`).
   
      :Examples:
   
         ::
   
            from kapteyn import maputils
            from matplotlib import pyplot as plt
            fitsobj = maputils.FITSimage("m101.fits")
            mplim = fitsobj.Annotatedimage()
            grat = mplim.Graticule()
            grat2 = grat.minortickmarks()
            mplim.plot()
            plt.show()
   
         Adding parameters to change attributes:
   
            >>> grat2 = grat.minortickmarks(3, 5,
                     color="#aa44dd", markersize=3, markeredgewidth=2)
   
         Minor tick marks only along x axis:
   
            >>> grat2 = minortickmarks(grat, 3, None)
      """
      #-------------------------------------------------------------
      # Get the position of the first label in the original graticule
      startx = graticule.xstarts[0]
      starty = graticule.ystarts[1]
      skyout = graticule.skyout
      spectrans=graticule.spectrans
      wxlim = graticule.wxlim
      wylim = graticule.wylim
      # Separate the kwargs to be able to set just one of them
      # to invisible (partsx/y is None)
      if not kwargs.has_key('markersize'):
         kwargs.update(markersize=2)
      kwargs1 = kwargs.copy()
      kwargs2 = kwargs.copy()
      # Get the steps along the axes
      deltax = graticule.delta[0]
      deltay = graticule.delta[1]
      # Adjust the steps.
      if partsx is None:
         deltax = None
         kwargs1.update(visible=False)
      else:
         deltax /= float(max(2,min(20,int(abs(partsx)))))
      if partsy is None:
         deltay = None
         kwargs2.update(visible=False)
      else:
         deltay /= float(max(2,min(20,int(abs(partsy)))))
      # Build the new graticule
      minorgrat = self.Graticule(startx=startx, starty=starty,
                                 deltax=deltax, deltay=deltay,
                                 wxlim=wxlim, wylim=wylim,
                                 skyout=skyout, spectrans=spectrans)
      # Make unnecessary elements invisible
      minorgrat.setp_gratline(wcsaxis=[0,1], visible=False)
      minorgrat.setp_ticklabel(wcsaxis=[0,1], visible=False)
      minorgrat.setp_axislabel(plotaxis=[0,1,2,3], visible=False)
      # Change attributes of tick mark (i.e. a Matplotlib Line2D object)
      minorgrat.setp_tickmark(wcsaxis=0, **kwargs1)
      minorgrat.setp_tickmark(wcsaxis=1, **kwargs2)
      # Return the graticule so that its attributes can also be changed with
      # methods in the calling environment.
      # Note that there is need to store the new object because
      # method Graticule() has aready done that.
      return minorgrat



   def Pixellabels(self, **kwargs):
      #-----------------------------------------------------------------
      """
      This method is a call to the constructor of class
      :class:`wcsgrat.Pixellabels`
      with a number of default parameters.
      It sets the annotation along a plot axis to pixel coordinates.

      :param plotaxis:     The axis name of one or two of the axes of the
                           plot rectangle:
                           or 'left', 'bottom', 'right', 'top'
                           Combinations are always between
                           'left' and 'bottom' and 'right' and 'top'.
      :type  plotaxis:     String
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
                           header (e.g. as CRPIX1 and CRPIX2).
                           In this routine the nearest integer of
                           the input is calculated to ensure that the
                           offset is an integer value.
      :type offset:        *None* or floating point numbers

      Other parameters are related to Matplotlib label attributes.

      :Examples:

      >>> fitsobject = maputils.FITSimage("m101.fits")
      >>> annim = fitsobject.Annotatedimage()
      >>> annim.Pixellabels(plotaxis=("top","right"), color="b", markersize=10)

      or separate the labeling so that you can give different properties
      for different axes. In this case we shift the labels along the
      top axis towards the axis line with *va='top'*:
         
      >>> annim.Pixellabels(plotaxis='top', va='top')
      >>> annim.Pixellabels(plotaxis='right')

      :Notes:

      If a pixel offset is given for this Annimated object, then plot
      the pixel labels with this offset.

      """
      #-----------------------------------------------------------------
      pixlabels = Pixellabels(self.pxlim, self.pylim, offset=self.pixoffset, **kwargs)
      self.objlist.append(pixlabels)
      return pixlabels


   def Beam(self, major, minor, pa=0.0, pos=None, xc=None, yc=None, units=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Objects from class Beam are graphical representations of the resolution
      of an instrument. The beam is centered at a position xc, yc.
      The major axis of the beam is the FWHM of the longest distance between
      two opposite points on the ellipse. The angle between the major axis
      and the North is the position angle.

      A beam is an ellipse in world coordinates.
      To draw a beam given the input parameters, points are calculated in world
      coordinates so that angle and required distance of sample points on the
      ellipse are correct on a sphere.
       
      
      :param major: Full width at half maximum of major axis of beam in degrees.
      :type major:  Float
      :param minor: Full width at half maximum of minor axis of beam in degrees.
      :type minor:  Float
      :param pa:    Position angle in degrees. This is the angle between
                    the positive y-axis and the major axis of the beam.
                    The default value is 0.0.
      :type pa:     Float
      :param pos:   A string that represents the position of the center
                    of the beam. If two numbers are available then
                    one can also use parameters *xc* and *yc*.
                    The value in parameter *pos* supersedes the values
                    in *xc* and *yc*.
      :type pos:    String
      :param xc:    X value in **world** coordinates of the center position
                    of the beam.
      :type xc:     Float
      :param yc:    Y value in **world** coordinates of the center position
                    of the beam.
      :type yc:     Float

      :Examples:
      
       ::
      
         fwhm_maj = 0.08
         fwhm_min = 0.06
         lat = 54.347395233845
         lon = 210.80254413455
         beam = annim.Beam(fwhm_maj, fwhm_min, 90, xc=lon, yc=lat,
                           fc='g', fill=True, alpha=0.6)
         pos = '210.80254413455 deg, 54.347395233845 deg'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='m', fill=True, alpha=0.6)
         pos = '14h03m12.6105s 54d20m50.622s'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='y', fill=True, alpha=0.6)
         pos = 'ga 102.0354152 {} 59.7725125'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='g', fill=True, alpha=0.6)
         pos = 'ga 102d02m07.494s {} 59.7725125'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='b', fill=True, alpha=0.6)
         pos = '{ecliptic,fk4, j2000} 174.3674627 {} 59.7961737'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='r', fill=True, alpha=0.6)
         pos = '{eq, fk4-no-e, B1950} 14h01m26.4501s {} 54d35m13.480s'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
         pos = '{eq, fk4-no-e, B1950, F24/04/55} 14h01m26.4482s {} 54d35m13.460s'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
         pos = '{ecl} 174.367764 {} 59.79623457'
         beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
         pos = '53 58'     # Pixels
         beam = annim.Beam(0.04, 0.02, pa=30, pos=pos, fc='y', fill=True, alpha=0.4)
         pos = '14h03m12.6105s 58' # World coordinate and a pixel coordinate
         beam = annim.Beam(0.04, 0.02, pa=-30, pos=pos, fc='y', fill=True, alpha=0.4)

      :Properties:

         A selection of keyword arguments for the beam (which is a
         Matplotlib :class:`Polygon` object) are:

         * alpha - float (0.0 transparent through 1.0 opaque)
         * color - Matplotlib color arg or sequence of rgba tuples
         * edgecolor or ec - Matplotlib color spec, or None for default, or 'none' for no color
         * facecolor or fc - Matplotlib color spec, or None for default, or 'none' for no color
         * linestyle  or ls - ['solid' | 'dashed' | 'dashdot' | 'dotted']
         * linewidth or lw - float or None for default

      """
      #-----------------------------------------------------------------
      if pos != None:
         poswp = str2pos(pos, self.projection, mixpix=self.mixpix, gridmode=self.gridmode)
         if poswp[3] != "":
            raise Exception, poswp[3]
         world = poswp[0][0]      # Only first element is used
      else:
         world = (xc, yc)
      spatials = [self.projection.lonaxnum, self.projection.lataxnum]
      spatialmap = self.axperm[0] in spatials and self.axperm[1] in spatials
      if not spatialmap:
         raise Exception, "Can only plot a beam in a spatial map"

      beam = Beam(world[0], world[1], major, minor, pa, projection=self.projection,
                  units=units, **kwargs)
      self.objlist.append(beam)
      return beam


   def Skypolygon(self, prescription=None, xc=None, yc=None,
                  cpos=None, major=None, minor=None,
                  nangles=6, pa=0.0,
                  units=None, lons=None, lats=None, stepsize=1.0, **kwargs):
   #-----------------------------------------------------------------
      """
      Construct an object that represents an area in the sky.
      Usually this is an ellipse, rectangle or regular polygon with
      given center and other parameters to define its size or number
      of angles and the position angle. The object is plotted in a
      way that the sizes and angles, as defined on a sphere, are
      preserved.
      The objects need a 'prescription'. This is a recipe to calculate
      a distance to a center point (0,0) as function of an angle
      in a linear and flat system. Then the object perimeter is
      re-calculated for a given center (xc,yc) and for corresponding
      angles and distances on a sphere.

      If *prescription=None*, then this method expects two arrays
      *lons* and *lats*.  These are copied unaltered as vertices for
      an irregular polygon.

      For cylindrical projections it is possible that a polygon in a
      all sky plot
      crosses a boundary (e.g. 180 degrees longitude if the projection
      center is at 0 degrees). Then the object is splitted into two parts
      one for the region 180-phi and one for the region 180+phi where phi
      is an arbitrary positive angle.
      This splitting is done for objects with and without a prescription.

      :param prescription:
         How should the polygon be created? The prescriptions are
         "ellipse", "rectangle", "npolygon" or None.
         This method only checks the first character of the string.
      :type prescription:
         String or None
      :param xc:
         Coordinate in degrees to set the center of the shape in X
      :type xc:
         Floating point number
      :param yc:
         Coordinate in degrees to set the center of the shape in Y
      :type yc:
         Floating point number
      :param cpos:
         Instead of a position in world coordinates (xc, yc),
         supply a string with a position. The syntax is described
         in the positions module. For example:
         ``cpos='20h30m10s -10d10m20.23s`` or ``cpos=ga 110.3 ga 33.4``
      :type cpos:
         String
      :param major:
         Major axis of ellipse in degrees. This parameter is also used as
         height for rectangles and as radius for regular polygons.
      :type major:
         Floating point number
      :param minor:
        Minor axis of ellipse in degrees. This parameter is also used as
         width for rectangles. If the prescription is an ellipse then
         a circle is defined if *major*=*minor*
      :type minor:
         Floating point number
      :param nangles:
         The number of angles in a regular polygon. The radius of this
         shape is copied from parameter *major*.
      :type nangles:
         Integer
      :param pa:
         Position angle. This is an astronomical angle i.e. with respect
         to the north in the direction of the east. For an ellipse the angle is
         between the north and the major axis. For a rectangle it is the
         angle between the north and the parameter that represents the height.
         For a regular polygon, it is the angle between the north and the line that
         connects the center with the first angle in the polygon.
      :type pa:
         Floating point number
      :param units:
         A case insensitive minimal matched string that sets the units for the
         values in *major* and *minor* (e.g. arcmin, arcsec).
      :type units:
         String
      :param lons:
         Sequence with longitudes in degrees that (together with matching
         latitudes) are used to define the vertices of a polygon.
         If nothing is entered for *prescription* or *prescription=None*
         then these positions are used unaltered.
      :type lons:
         Sequence of floating point numbers.
      :param lats:
         See description at *lons*.
      :type lats:
         Sequence of floating point numbers.
      :param stepsize:
         Step in degrees to calculate samples (applies to ellipse only)
      :type stepsize:
         Floating point number
      :param kwargs:
         Plot parameters
      :type kwargs:
         Matplotlib keyword arguments
      """
   #-----------------------------------------------------------------
      if cpos != None:
         poswp = str2pos(cpos, self.projection, mixpix=self.mixpix, gridmode=self.gridmode)
         if poswp[3] != "":
            raise Exception, poswp[3]
         world = poswp[0][0]      # Only first element is used
      else:
         world = (xc, yc)
      spatials = [self.projection.lonaxnum, self.projection.lataxnum]
      spatialmap = self.axperm[0] in spatials and self.axperm[1] in spatials
      if not spatialmap:
         raise Exception, "Can only plot a sky polygon in a spatial map"

      spoly = Skypolygon(projection=self.projection,
                         prescription=prescription,
                         xc=world[0], yc=world[1],
                         major=major, minor=minor,
                         nangles=nangles, pa=pa,
                         units=units, 
                         lons=lons, lats=lats, stepsize=stepsize,
                         **kwargs)

      self.objlist.append(spoly)
      return spoly


   def Marker(self, pos=None, x=None, y=None, mode='', **kwargs):
      #-----------------------------------------------------------------
      """
      Plot marker symbols at given positions. This method creates
      objects from class Marker. The constructor of that class needs
      positions in pixel coordinates. Here we allow positions to
      be defined in a string which can contain either world- or pixel
      coordinates (or a mix of both). If x and y coordinates are
      known, or read from file, one can also enter this data without
      parsing. The keyword arguments *x* and *y* can be used to enter
      pixel coordinates or world coordinates.

      :param pos:
         A definition of one or more positions for the current
         image. The string is parsed by :mod:`positions`.
      :type pos:
         String
      :param x:
         If keyword argument *pos* is not used, then this method
         expects numbers in parameters *x* and *y*. Advantage of
         using this parameter, is that it skips the position
         parser and therefore it is much faster.
      :type x:
         Float or a sequence of floating point numbers
      :param y:
         If keyword argument *pos* is not used, then this method
         expects numbers in parameters *x* and *y*
      :type y:
         Float or a sequence of floating point numbers
      :param world:
         Flag to set the conversion mode. If True then the numbers
         in *x* and *y* are world coordinates. Else, they are
         processed as pixel coordinates.

      :Returns:

         Object from class :class:`Marker`

      :Examples:
         In the first example we show 4 markers plotted in the
         projection center (given by header values CRPIX)::
      
            f = maputils.FITSimage("m101.fits")
            fig = plt.figure()
            frame = fig.add_subplot(1,1,1)
            annim = f.Annotatedimage(frame, cmap="binary")
            annim.Image()
            grat = annim.Graticule()
            annim.Marker(pos="pc", marker='o', markersize=10, color='r')
            annim.Marker(pos="ga 102.035415152 ga 59.772512522", marker='+',
                        markersize=20, markeredgewidth=2, color='m')
            annim.Marker(pos="{ecl,fk4,J2000} 174.367462651 {} 59.796173724",
                        marker='x', markersize=20, markeredgewidth=2, color='g')
            annim.Marker(pos="{eq,fk4-no-e,B1950,F24/04/55} 210.360200881 {} 54.587072397",
                        marker='o', markersize=25, markeredgewidth=2, color='c',
                        alpha=0.4)

         In the second example we show how to plot a sequence of markers.
         Note the use of the different keyword arguments and the role
         of flag *world* to force the given values to be processed in pixel
         coordinates::
       
            # Use pos= keyword argument to enter sequence of
            # positions in pixel coordinates
            pos = "[200+20*sin(x/20) for x in range(100,200)], range(100,200)"
            annim.Marker(pos=pos, marker='o', color='r')
            
            # Use x= and y= keyword arguments to enter sequence of
            # positions in pixel coordinates
            xp = [400+20*numpy.sin(x/20.0) for x in range(100,200)]
            yp = range(100,200)
            annim.Marker(x=xp, y=yp, mode='pixels', marker='o', color='g')

            # Single position in pixel coordinates
            annim.Marker(x=150, y=150, mode='pixels', marker='+', color='b')
            
         In the next example we show how to use method :meth:`positionsfromfile`
         in combination with this Marker method to read positions
         from a file and to plot them. The positions in the file
         are world coordinates. Method :meth:`positionsfromfile`
         returns pixel coordinates::
      
            fn = 'smallworld.txt'
            xp, yp = annim.positionsfromfile(fn, 's', cols=[0,1])
            annim.Marker(x=xp, y=yp, mode='pixels', marker=',', color='b')

      """
      #-----------------------------------------------------------------
      if x is None and y is None and pos is None:
         # Nothing to do
         return None

      # For parameters *x* and *y* a *mode* should be given
      if pos is None:
         p = mode.upper().startswith('P')
         if p:
            w = False
         else:
            w = mode.upper().startswith('W')
         if not p and not w:
            raise Exception, "Marker(): Mode not or incorrectly specified!"
         else:
            world = w

      if (x is None and not y is None) or (not x is None and y is None):
         raise Exception, "Marker(): One of the arrays is None and the other is not!"
      if not pos is None and (not x is None or not y is None):
         raise Exception, "Marker(): You cannot enter values for both pos= and x= and/or y="

      if not pos is None:
         poswp = str2pos(pos, self.projection, mixpix=self.mixpix, gridmode=self.gridmode)
         if poswp[3] != "":
            raise Exception, poswp[3]
         xp = poswp[1][:,0]
         yp = poswp[1][:,1]
      else:
         if world:
            xp, yp = self.topixel(x, y)
         else:
            xp, yp = x, y
      markers = Marker(xp, yp, **kwargs)
      self.objlist.append(markers)
      return markers


   def Ruler(self, pos1=None, pos2=None,
             x1=None, y1=None, x2=None, y2=None, lambda0=0.5, step=None,
             world=False, angle=None, addangle=0.0,
             fmt=None, fun=None, fliplabelside=False, mscale=None,
             labelsintex=True, **kwargs):
      #-----------------------------------------------------------------
      """
      This method prepares arguments for a call to function
      :func:`rulers.Ruler` in module :mod:`rulers`

      Note that this method sets a number of parameters which
      cannot be changed like *projection*, *mixpix*, *pxlim*, *pylim*
      and *aspectratio*, which are all derived from the properties of the current
      :class:`maputils.Annotatedimage` object.
      """
      #-----------------------------------------------------------------
      ruler = rulers.Ruler(self.projection, self.mixpix, self.pxlim, self.pylim,
                           aspectratio=self.aspect, pos1=pos1, pos2=pos2,
                           x1=x1, y1=y1, x2=x2, y2=y2, lambda0=lambda0, step=step,
                           world=world, angle=angle, addangle=addangle,
                           fmt=fmt, fun=fun, fliplabelside=fliplabelside, mscale=mscale,
                           labelsintex=labelsintex, **kwargs)
      self.objlist.append(ruler)
      return ruler


   def plot(self):
      #-----------------------------------------------------------------
      """
      Plot all objects stored in the objects list for this Annotated image.

      :Example:

         >>> fitsobj = maputils.FITSimage('m101.fits')
         >>> annim = fitsobj.Annotatedimage()
         >>> grat = annim.Graticule()
         >>> annim.plot()
         >>> plt.show()

      """
      #-----------------------------------------------------------------
      needresize = False
      for obj in self.objlist:
         try:
            pt = obj.ptype
            if pt == "Colorbar":
               if obj.frame is None:             # No frame set by user
                  needresize = True
                  orientation = obj.kwargs['orientation']
               else:
                  self.cbframe = obj.frame
         except:
            raise Exception, "Unknown object. Cannot plot this!"
      
      if needresize:                             # because a colorbar must be included
         self.cbframe = make_axes(self.frame, orientation=orientation)[0]
         # A frame is created for the colorbar. The original frame has been changed too.
         # Make sure that it gets the same properties as before the change.
         self.frame = self.adjustframe(self.frame)
      

      for obj in self.objlist:
         try:
            pt = obj.ptype
         except:
            raise Exception, "Unknown object. Cannot plot this!"
         if pt in ["Image", "Contour", "Graticule", "Insidelabels", "Pixellabels", "Beam",
                   "Marker", "Ruler", "Skypolygon"]:
            try:
               visible = obj.visible
            except:
               visible = True
            flushprint("Graticule plot nu object %s"%(str(obj.ptype)))            
            obj.plot(self.frame)
            # If we want to plot derived objects (e.g. ruler) and not the graticule
            # then set visible to False in the constructor.

      for obj in self.objlist:
         pt = obj.ptype
         if pt == "Colorbar":
            if not self.image is None:
               obj.plot(self.cbframe, self.image.im)
            elif not self.contourset is None:
               obj.plot(self.cbframe, self.contourset.CS)
            else:
               raise Exception, "A color bar could not find an image or contour set!"

      self.cmap.add_frame(self.frame)   # Add to list in mplutil



   def toworld(self, xp, yp, matchspatial=False):
      #--------------------------------------------------------------------
      """
      This is a helper method for method :meth:`wcs.Projection.toworld`.
      It converts pixel positions from a map to world coordinates.
      The difference with that method is that this method has its focus
      on maps, i.e. two dimensional structures.
      It knows about the missing spatial axis if your data slice has only one
      spatial axis.
      Note that pixels in FITS run from 1 to *NAXISn* and that the pixel
      coordinate equal to *CRPIXn* corresponds to the world coordinate
      in *CRVALn*.
   
      :param xp:
         Pixel value(s) corresponding to the x coordinate of a position.
      :type xp:
         Single Floating point number or sequence
      :param yp:
         A pixel value corresponding to the y coordinate of a position.
      :type yp:
         Single Floating point number or sequence
      :param matchspatial:
         If True then also return the world coordinate of the matching spatial
         axis. Usually this is an issue when the map is a slice with only
         one spatial axis (XV- or Position-Velocity map)
      :type matchspatial:
         Boolean

      :Note:
         If somewhere in the process an error occurs,
         then the return values of the world
         coordinates are all *None*.

      :Returns:
         Three world coordinates: *xw* which is the world coordinate for
         the x-axis, *yw* which is the world coordinate for
         the y-axis and (if *matchspatial=True*) *missingspatial* which
         is the world coordinate
         that belongs to the missing spatial axis.
         If there is not a missing spatial axis, then the value of this
         output parameter is *None*. So you don't need to know the structure
         of the map beforehand. You can test whether the last value
         is *None* or not *None* in the calling environment.
   
      :Examples:
         We have a test set with:
         
         * RA:   crpix1=51  - crval1=-51,28208479590
         * DEC:  crpix2=51  - crval2=+60.15388802060
         * VELO: crpix3=-20 - crval3=-243000 (m/s)
   
         Now let us try to find the world coordinates
         of a RA-VELO map at (crpix1, crpix3) at slice position DEC=51.
         We should get three numbers which are all equal
         to the value of *CRVALn*
   
         >>> from kapteyn import maputils
         >>> fig = figure()
         >>> fitsobject = maputils.FITSimage('ngc6946.fits')
         >>> fitsobject.set_imageaxes(1,3, slicepos=51)
         >>> annim = fitsobject.Annotatedimage()
         >>> annim.toworld(51,-20)
         (-51.282084795899998, -243000.0, 60.1538880206)
         >>> annim.topixel(-51.282084795899998, -243000.0)
         (51.0, -20.0)

         Or work with a sequence of numbers (list, tuple of NumPy ndarray
         object) as in this example::

            from kapteyn import maputils

            f = maputils.FITSimage("ngc6946.fits")
            # Get an XV slice at DEC=51
            f.set_imageaxes(1, 3, slicepos=51)
            annim = f.Annotatedimage()
            
            x = [10, 50, 300, 399]
            y = [1, 44, 88, 100]
            
            # Convert these to world coordinates
            lon, velo, lat = annim.toworld(x, y, matchspatial=True)
            print "lon, velo lat=", lon, velo, lat

            # We are not interested in the pixel coordinate of the slice
            # because we know it is 51. Therefore we omit 'matchspatial'
            x, y = annim.topixel(lon, velo)
            print "Back to pixel coordinates: x, y =", x, y

            #Output:
            #lon, velo lat= [-50.691745281033555,
            #                -51.267685761904154,
            #                -54.862775451370837,
            #                -56.280231731192607]
            #               [-154800.00401099998,
            #                 25799.987775999994,
            #                 210599.97937199997,
            #                 260999.97707999998]
            #               [ 60.152142940138205,
            #                 60.153886982461088,
            #                 60.089564526325667,
            #                 60.028325686860846]

            #Back to pixel coordinates: x, y = [  10.   50.  300.  399.]
            #                                  [   1.   44.   88.  100.]
      """
      #--------------------------------------------------------------------
      xw = yw = None
      missingspatial = None
      try:
         if (self.mixpix is None):
            xw, yw = self.projection.toworld((xp, yp))
         else:
            if issequence(xp):
               mixpix = numpy.zeros(len(xp)) + self.mixpix
            else:
               mixpix = self.mixpix
            xw, yw, missingspatial = self.projection.toworld((xp, yp, mixpix))
      except:
         pass
      if matchspatial:
         return xw, yw, missingspatial
      else:
         return xw, yw



   def topixel(self, xw, yw, matchspatial=False):
      #--------------------------------------------------------------------
      """
      This is a helper method for method :meth:`wcs.Projection.topixel`.
      It knows about the missing spatial axis if a data slice has only one
      spatial axis. It converts world coordinates in units (given by the
      FITS header, or the spectral translation) from a map to pixel coordinates.
      Note that pixels in FITS run from 1 to *NAXISn*.
   
      :param xw:
         A world coordinate corresponding to the x coordinate of a position.
      :type xw:
         Floating point number
      :param yw:
         A world coordinate corresponding to the y coordinate of a position.
      :type yw:
         Floating point number
      :param matchspatial:
         If set to *True* then return the pixel coordinates and the value of the
         pixel on the missing spatial axis.
      :type matchspatial:
         Boolean

      :Returns:
         Two pixel coordinates: *x* which is the world coordinate for
         the x-axis and *y* which is the world coordinate for
         the y-axis.

         If somewhere in the proces an error occurs, then the return values of the pixel
         coordinates are all *None*.
   
      :Notes:
         This method knows about the pixel on the missing spatial axis
         (if there is one). This pixel is usually the pixel coordinate of
         the slice if the dimension of the data is > 2.

      :Examples:
         See example at :meth:`toworld`
      """
      #--------------------------------------------------------------------
      x = y = None
      try:
         if (self.mixpix is None):
            x, y = self.projection.topixel((xw, yw))
         else:
            # Note that we have a fixed grid for the missing spatial axis,
            # but this does not imply a constant world coordinate. Therefore
            # we have to use the mixed transformation method.
            if issequence(xw):
               mixpix = numpy.zeros(len(xw)) + self.mixpix
               unknown = numpy.zeros(len(xw)) + numpy.nan
            else:
               mixpix = self.mixpix
               unknown = numpy.nan
            wt = (xw, yw, unknown)
            pixel = (unknown, unknown, mixpix)
            (wt, pixel) = self.projection.mixed(wt, pixel)
            x = pixel[0]; y = pixel[1]
      except:
         pass
      if matchspatial:
         return x, y, self.mixpix
      else:
         return x, y


   def inside(self, x=None, y=None, pos=None, mode=''):
      #--------------------------------------------------------------------
      """
      This convenience method belongs to class :class:`Annotatedimage` which
      represents a two dimensional map which could be a slice
      (*slicepos*) from a
      bigger data structure and/or could be limited by limits on the
      pixel ranges of the image axes (*pxlim*, *pylim*).
      Then, for a sequence of coordinates in x and y, return a sequence with
      Booleans with *True* for a coordinate within the boundaries of this
      map and *False* when it is outside the boundaries of this
      map. This method can work with either sequences of coordinates
      (parameters *x* and *y*)
      or a string with a position (parameter *pos*).
      If parameters *x* and *y* are used then parameter *world* sets
      these coordinates to world- or pixel coordinates.

      :param x:
         Single number of a sequence representing the x coordinates
         of your input positions. These coordinates are world coordinates
         if *mode='world'* (or *mode='w'*) and pixel
         coordinates if *mode='pixels* (or *mode='p'*).
      :type x:
         Floating point number or sequence of floating point numbers.
      :param y:
         Single number of a sequence representing the x coordinates
         of your input positions. See description for parameter *x*
      :type y:
         Floating point number or sequence of floating point numbers.
      :param mode:
         Input in *x* and*y* represent either pixel coordinates or
         world coordinates. Is the first character is 'p' or 'P' then
         the mode is set to pixels. If it starts with 'w' or 'W' the input
         in *x* and *y* are world coordinates.
      :param pos:
         A description of one or a number of positions entered as a string.
         The syntax is described in module :mod:`positions`.
         The value of parameter *mode* is ignored.
      :type pos:
         String
      :param world:
         If parameters *x* and *y* are used then the step of coordinate
         interpretation as with *pos* is skipped. These coordinates can be either
         pixel- or world coordinates depending on the value of *world*.
         By default this value is *True*.
      :type world:
         Boolean

      :Raises:
         :exc:`Exception`
            *One of the arrays is None and the other is not!*

         :exc:`Exception`
            *You cannot enter values for both pos= and x= and/or y=*

      :Returns:

         * None -- there was nothing to do
         * Single Boolean -- Input was a single position
         * NumPy array of Booleans -- Input was a sequence of positions

      :Note:
         
         For programmers: note the similarity to method :meth:`Marker`
         with respect to the use of method :meth:`positions.str2pos`.

         This method is tested with script *mu_insidetest.py* which
         is part of the examples tar file.

      :Examples:

      >>> fitsobj = maputils.FITSimage("m101.fits")
      >>> fitsobj.set_limits((180,344), (100,200))
      >>> annim = fitsobj.Annotatedimage()

      >>> pos="{} 210.870170 {} 54.269001"
      >>> print annim.inside(pos=pos)
      >>> pos="ga 101.973853, ga 59.816461"
      >>> print annim.inside(pos=pos)

      >>> x = range(180,400,40)
      >>> y = range(100,330,40)
      >>> print annim.inside(x=x, y=y, mode='pixels')

      >>> print annim.inside(x=crval1, y=crval2, mode='w')

      """
      #--------------------------------------------------------------------
      if x is None and y is None and pos is None:
         # Nothing to do
         return None

      # For parameters *x* and *y* a *mode* should be given
      if pos is None:
         p = mode.upper().startswith('P')
         if p:
            w = False
         else:
            w = mode.upper().startswith('W')
         if not p and not w:
            raise Exception, "Inside(): Mode not or incorrectly specified!"
         else:
            world = w
            
      if (x is None and not y is None) or (not x is None and y is None):
         raise Exception, "Inside(): One of the arrays is None and the other is not!"
      if not pos is None and (not x is None or not y is None):
         raise Exception, "Inside(): You cannot enter values for both pos= and x= and/or y="
      if not pos is None:
         world, pixels, units, errmes = str2pos(pos, self.projection, mixpix=self.mixpix, gridmode=self.gridmode)
         if errmes != '':
            raise Exception, errmes
         else:
            xp = pixels[:,0]
            yp = pixels[:,1]
      else:
         if world:
            xp, yp = self.topixel(x, y)
            # When conversions fail then None, None is returned
            if not issequence(x):
               x = [x]
            if not issequence(y):
               y = [y]
            if xp is None:
               xp = numpy.array([None]*len(x))
            if yp is None:
               yp = numpy.array([None]*len(y))
         else:
            if not isinstance(x, (numpy.ndarray)):
               xp = numpy.array(x)
            else:
               xp = x
            if not isinstance(y, (numpy.ndarray)):
               yp = numpy.array(y)
            else:
               yp = y
               
      # At this stage we have an array for the x and y coordinates.
      # Now check whether they are inside a map
      
      b = numpy.where((xp>self.pxlim[0]-0.5) &
                      (xp<self.pxlim[1]+0.5) &
                      (yp>self.pylim[0]-0.5) &
                      (yp<self.pylim[1]+0.5), True, False)
      if b.shape != () and len(b) == 1:
         b = b[0]
      return b
         

   def positionmessage(self, x, y, posobj, parts=False):
      #--------------------------------------------------------------------
      """
      Display cursor position in pixels and world coordinates together
      with the image value in the message area of a Matplotlib window.
      This function is used internally by :meth:`on_move` and
      :meth:`on_click`.

      :param x:
         The x coordinate of a position in the image
      :type x:
         Floating point number
      :param y:
         The y coordinate of a position in the image
      :type y:
         Floating point number
      :param posobj:
         A description of how the string should be formatted is
         stored in this object.
      :type posobj:
         An object from class :class:`Positionmessage`

      :Returns:
         A formatted string with position information and
         the image value of the corresponding pixel. E.g.
         ``x,y= 56.91, 25.81  wcs=-51.366665, 59.973195  Z=-4.45e-04``

         The WCS coordinates are not formatted in hms/dms. They are
         printed in the units of the FITS header.

      :Notes:
         This method knows about the object's missing axis if
         only one spatial axis was part of the image.
         It takes also care of NaNs in your map.
      """
      #--------------------------------------------------------------------
      s = ''
      if self.pxlim[0]-0.5 < x < self.pxlim[1]+0.5 and self.pylim[0]-0.5 < y < self.pylim[1]+0.5:
         if 1: #try:            
            xw, yw, missingspatial = self.toworld(x, y, matchspatial=True)
            #xi = numpy.round(x) - (self.pxlim[0]-1)
            #yi = numpy.round(y) - (self.pylim[0]-1)
            xi = numpy.round(x - (self.pxlim[0]-1))
            yi = numpy.round(y - (self.pylim[0]-1))
            x -= self.pixoffset[0]; y -= self.pixoffset[1];
            spix = posobj.pix2str(x, y)
            swcs = posobj.wcs2str(xw, yw, missingspatial)
            swcsuf = posobj.wcs2str(xw, yw, missingspatial, unformatted=True)
            if self.rgbs is None:
               if self.data is None:
                  z = numpy.nan
               else:
                  z = self.data[yi-1, xi-1]
            else:
               z = (self.rgbs[0][yi-1, xi-1], self.rgbs[1][yi-1, xi-1], self.rgbs[2][yi-1, xi-1])
            s = ''
            if not spix is None:
               s = "x,y=%s" % spix
            if not swcs is None:
               s += "  wcs=%s" % swcs
            sz = None
            if not (self.rgbs is None and self.data is None):
               sz   = posobj.z2str(z)
               s += "  z=%s" % sz
         else: #except:
            spix = self.posobj.pix2str(x,y)
            s = "pix:%s" % spix
      if parts:
         return(spix, swcs, swcsuf, sz)
      else:
         return s.strip()


   def mouse_toolbarinfo(self, axesevent):
      #--------------------------------------------------------------------
      """
      *Display position information:*

      Note that the events (motion_notify_event) are blocked from the
      moment you changed mode in the toolbar to zoom or pan mode.
      So we display a message in our modified Matplotlib routines
      that deal with callbacks for zoom- and pan actions.
      """
      #--------------------------------------------------------------------
      s = ''      
      x, y = axesevent.xdata, axesevent.ydata
      self.X_lastvisited = x
      self.Y_lastvisited = y
      mode = self.figmanager.toolbar.mode

      s = self.positionmessage(x, y, axesevent.posobj)
      if s != '':
         if self.externalmessenger is None and not backend.startswith('QT'):
            # For a non QT canvas with width 19.5 cm, there is a fixed width for
            # toolbar buttons (7.5 cm = 2.95 inch). For the rest (12 cm) we have
            # space to set a message. Unfortunately we don't have any information
            # about the real width of the string with information.
            # In this 12 cm we have 57 characters. On average the size of a
            # character is 12/57/2.54 inch (0.083 inch).
            charw = 0.12    # Average character width (better than 0.083)
            xsize = self.frame.figure.get_figwidth()
            messagemax = xsize - 2.5
            if len(s)*charw > messagemax:
               l = int(messagemax/charw)
               s = s[:l]
         self.messenger(s)
         

   def interact_toolbarinfo(self, pixfmt="%.1f", wcsfmt="%.3e", zfmt="%+.3e",
                            hmsdms=True, dmsprec=1):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
      It reacts to mouse movements. A message is prepared with position information
      in both pixel coordinates and world coordinates. The world coordinates
      are in the units given by the (FITS) header.

      :param pixfmt:
         Python number format for pixel coordinates
      :type pixfmt:
         String
      :param wcsfmt:
         Python number format for wcs coordinates if the coordinates
         are not spatial or if parameter *hmsdms* is False.
      :type wcsfmt:
         String
      :param zfmt:
         Python number format for image value(s)
      :type pixfmt:
         String
      :param hmsdms:
         If True (default) then spatial coordinates will be formatted
         in hours/degrees, minutes and seconds according to the current sky system.
         The precision in seconds is entered with parameter *dmsprec*.
      :type hmsdms:
         Boolean
      :param dmsprec:
         Number of decimal digits in seconds for coordinates formatted in
         in HMS/DMS
      :type dmsprec:
         Integer
         
      :Notes:

         If a format is set to *None*, its corresponding number(s) will
         not appear in the informative message.

         If a message does not fit in the toolbar then only a part is
         displayed. We don't have control over the maximum size of that message
         because it depends on the backend that is used (GTK, QT,...).
         If nothing appears, then a manual resize of the window will suffice.

      :Example: 

         Attach to an object from class :class:`Annotatedimage`:
      
         >>> annim = f.Annotatedimage(frame)
         >>> annim.interact_toolbarinfo()

         or:

         >>> annim.interact_toolbarinfo(wcsfmt=None, zfmt="%g")
         
         A more complete example::

            from kapteyn import maputils
            from matplotlib import pyplot as plt

            f = maputils.FITSimage("m101.fits")
            
            fig = plt.figure(figsize=(9,7))
            frame = fig.add_subplot(1,1,1)
            
            annim = f.Annotatedimage(frame)
            ima = annim.Image()
            annim.Pixellabels()
            annim.plot()
            annim.interact_toolbarinfo()
            
            plt.show()

      """
      #--------------------------------------------------------------------
      posobj = Positionmessage(self.projection.skysys, self.skyout, self.projection.types)
      flushprint("Ik maak hier een positionmessge object met skyout=%s"%(str(self.skyout)))
      posobj.pixfmt = pixfmt
      posobj.wcsfmt = wcsfmt
      posobj.zfmt = zfmt
      posobj.hmsdms = hmsdms
      posobj.dmsprec = dmsprec      
      self.toolbarkey = AxesCallback(self.mouse_toolbarinfo, self.frame,
                                     'motion_notify_event', posobj=posobj)
      flushprint("Add mouse toolbar callback %d for object %d"%(id(self.toolbarkey), id(self)))                                     
      self.AxesCallback_ids.append(self.toolbarkey)


   def disconnectCallbacks(self):
      #--------------------------------------------------------------------
      """
      Disconnect callbacks made with AxecCallback()
      The id's are stored in a list 'AxesCallback_ids'
      """
      #--------------------------------------------------------------------
      flushprint("\n\n-------------------------------------\nRemoving Axescallbacks from %d"%(id(self)))
      for acid in self.AxesCallback_ids:
         acid.deschedule()
         flushprint("Removing Axescallback %d"%(id(acid)))
         del acid
      self.AxesCallback_ids = []   # Reset the list
      flushprint("-----------------------------------------")


   def mouse_imagecolors(self, axesevent):
      #--------------------------------------------------------------------
      """
      *Reset color limits:*
   
      If you move the mouse in this image and press the **right mouse button**
      at the same time, then the color limits for image and colorbar are
      set to a new value.
      
      :param axesevent:
         AxesCallback event object with pixel position information.
      :type axesevent:
         AxesCallback instance
   
      :Example:
         Given an object from class :class:`Annotatedimage` called *annim*,
         register this callback function for the position information with:
   
         >>> annim.interact_imagecolors()
   
      :Notes:
         See *interact_imagecolors()* for more info.
      """
      #--------------------------------------------------------------------
      # Keys are often used as modifiers. So require an empty key
      if axesevent.event.button == 3 and axesevent.event.key in ['', None]:
         x, y = axesevent.xdata, axesevent.ydata
         if self.image is None or self.image.im is None:               # There is no image to adjust
            return
         # 1. event.xdata and event.ydata are the coordinates of the mouse location in
         # data coordinates (i.e. in screen pixels)
         # 2. transData converts these coordinates to display coordinates
         # 3. The inverse of transformation transAxes converts display coordinates to
         # normalized coordinates for the current frame.
         xy = self.frame.transData.transform((x,y))
         self.image.xyn_mouse = self.frame.transAxes.inverted().transform(xy)
         x , y = self.image.xyn_mouse
         slope = Annotatedimage.slopetrans * x   # i.e. at center: slope=0.5*slopetrans
         offset = y - Annotatedimage.shifttrans  # i.e. at center: offset=0-shifttrans
         self.cmap.modify(slope, offset)
         self.callback('slope', slope)
         self.callback('offset', offset)



   def key_imagecolors(self, axesevent, externalkey=None):
   #--------------------------------------------------------------------
      """
   This method catches keys which change the color setting of an image.
   See also documentation at *interact_imagecolors()*.

   :param axesevent:
      AxesCallback event object with pixel position information.
   :type axesevent:
      AxesCallback instance

   :Examples:
      If *image* is an object from :class:`Annotatedimage` then register
      this function with:

      >>> annim = f.Annotatedimage(frame)
      >>> annim.interact_imagecolors()
      """
   #--------------------------------------------------------------------      
      if externalkey is not None:
         eventkey = externalkey
      else:
         eventkey = axesevent.event.key

      if eventkey is None:
         # This can happen when the caps lock is on.
         return

      # Use the class variables !
      scales = Annotatedimage.lutscales
      scalekeys = Annotatedimage.scalekeys
      scales_default = Annotatedimage.scales_default
               
      # Request for another color map with page up/down keys
      if eventkey in ['pageup', 'pagedown']:
         lm = len(cmlist.colormaps)
         if eventkey == 'pagedown':
            self.cmindx += 1
            if self.cmindx >= lm:
               self.cmindx = 0
         if eventkey == 'pageup':
            self.cmindx -= 1
            if self.cmindx < 0:
               self.cmindx = lm - 1
         newcolormapstr = cmlist.colormaps[self.cmindx]
         self.messenger(newcolormapstr)
         self.cmap.set_source(newcolormapstr)     # Keep original object, just change the lut
         self.callback('lut', self.cmindx)
         #self.cmap.update()
      # Request for another scale, linear, logarithmic etc.
      elif eventkey in scalekeys:
         key = eventkey 
         #self.messenger(scales[key])
         s_indx = scalekeys[key]
         self.cmap.set_scale(scales[s_indx])
         mes = "Color map scale set to '%s'" % scales[s_indx]
         self.messenger(mes)
         self.callback('scale', s_indx)
      # Invert the color map colors
      #elif eventkey.upper() == 'I':
      elif eventkey == '9':
         if self.cmapinverse:
            self.cmap.set_inverse(False)
            self.cmapinverse = False
            mes = "Color map not inverted"
         else:
            self.cmap.set_inverse(True)
            self.cmapinverse = True
            mes = "Color map inverted!"
         self.messenger(mes)
         self.callback('inverse', self.cmapinverse)
      # Reset all color map parameters
      #elif eventkey.upper() == 'R':
      elif eventkey == '0':          # Reset all at once
         self.cmap.auto = False      # Postpone updates of the canvas.
         self.messenger('Reset color map to default')
         #self.cmap.set_source(self.startcmap)
         self.cmap.set_source(cmlist.colormaps[self.startcmindx])
         self.cmap.modify(1.0, 0.0)
         if self.cmapinverse:
            self.cmap.set_inverse(False)
            self.cmapinverse = False
         #colmap_start = cmlist.colormaps[self.image.startcmap]
         self.cmindx = self.startcmindx
         self.cmap.set_scale(scales[0])
         self.blankcol = Annotatedimage.blankcols[0]
         self.cmap.set_bad(self.blankcol)
         self.cmap.auto = True
         self.cmap.update()          # Update all
         self.callback('lut', self.cmindx)
         self.callback('inverse', False)
         self.callback('scale', scales_default)
         self.callback('slope', 1.0)
         self.callback('offset', 0.0)
         self.callback('blankcol', 0)
         """
         # Template for user interaction to set clip levels in existing image
         elif eventkey.upper() == 'C':
               cmin = self.clipmin + 200.0
               cmax = self.clipmax - 200.0
               print "cmin, max voor norm:", cmin, cmax
               self.set_norm(cmin, cmax)
         """
         
      elif eventkey.upper() == 'B':
         # Toggle colors for bad pixels (blanks) 
         blankcols = Annotatedimage.blankcols
         try:
            indx = blankcols.index(self.blankcol)
         except:
            indx = 0
         if indx + 1 == len(blankcols):  # Start at the beginning again
            indx = 0
         else:
            indx += 1
         self.blankcol = blankcols[indx]
         self.cmap.set_bad(self.blankcol)
         mes = "Color of bad pixels changed to '%s'" % (Annotatedimage.blanknames[indx])
         self.messenger(mes)
         self.callback('blankcol', indx)
      elif eventkey.upper() == 'M':
         stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
         filename = self.basename + "_" + stamp + ".lut"
         self.write_colormap(filename)
         mes = "Save color map to file [%s]" % filename
         self.messenger(mes)
      elif eventkey.upper() == 'H':
         # Set data to histogram equalized version
         if self.histogram:
            # Back to normal
            self.set_histogrameq(False)
            self.messenger('Original image displayed')
         else:
            if self.data_hist is None:
               self.messenger('Calculating histogram')
            self.set_histogrameq()
            self.messenger('Histogram eq. image displayed')
      elif eventkey.upper() == 'X':
         # Change the smoothing factor and go to blur mode.
         self.blurindx += 1
         if self.blurindx >= 10:
            self.blurindx = 1
         self.blurfac = self.blurindx * 0.5
         mes = "Blur index: %d blur sigma: %g" % (self.blurindx, self.blurfac)
         self.messenger(mes)
         self.set_blur(nx=self.blurfac, new=True)
      elif eventkey.upper() == 'Z':
         # Set data to blurred version
         if self.blurred:
            # Back to normal
            self.set_blur(False)
            self.messenger('Original image displayed')
         else:
            if self.data_blur is None:
               self.messenger('Calculating smoothed version')
            self.set_blur(nx=self.blurfac)
            self.messenger('Smoothed eq. image displayed')

   def set_histogrameq(self, on=True):
      if not on:
         # Back to normal
         self.data = self.data_orig
         self.histogram = False
      else:
         if self.data_hist is None:
            #self.data_hist = self.histeq()
            self.histeq()         # It sets attribute data_hist to new image
         self.data = self.data_hist
         self.histogram = True
      #self.norm = Normalize(vmin=self.clipmin, vmax=self.clipmax)
      if self.image.im != None:
         self.image.im.set_data(self.data)
      else:
         # An image was not yet 'plotted'. Then we adjust some
         # parameters first to prepare for the new image.
         self.image.data = self.data
      self.cmap.update()


   def set_blur(self, on=True, nx=10, ny=None, new=False):      
      if not on:
         # Back to normal
         self.data = self.data_orig
         self.blurred = False
      else:
         if self.data_blur is None or new:
            self.blur(nx, ny)
         self.data = self.data_blur
         self.blurred = True
      #self.norm = Normalize(vmin=self.clipmin, vmax=self.clipmax)
      if not self.image.im is None:
         self.image.im.set_data(self.data)
         #self.image.im.changed()
      else:
         # An image was not yet 'plotted'. Then we adjust some
         # parameters first to prepare for the new image.
         self.image.data = self.data
      #print "In blur id image, image.im, self.data", id(self.image), id(self.image.im), id(self.data)
      self.cmap.update()


   def interact_imagecolors(self):
      #--------------------------------------------------------------------
      """
      Add mouse interaction (right mouse button) and keyboard interaction
      to change the colors in an image.

      **MOUSE**

      If you move the mouse in the image for which you did register this
      callback function and press the **right mouse button**
      at the same time, then the color limits for image and colorbar are
      set to a new value.

      The new color setting is calculated as follows: first the position
      of the mouse (x, y) is transformed into normalized coordinates
      (i.e. between 0 and 1) called (xn, yn).
      These values are used to set the slope and offset for a function that
      sets an color for an image value according to the relations:
      ``slope = 2.0 * xn; offset = yn - 0.5``.
      The minimum and maximum values of the image are set by parameters
      *clipmin* and *clipmax*. For a mouse position exactly in the center
      (xn,yn) = (0.5,0.5) the slope is 1.0 and the offset is 0.0 and the
      colors will be divided equally between *clipmin* and *clipmax*.

      **KEYBOARD**

         * **page-down** move forwards through a list with known color maps.
         * **page-up** move backwards through a list with known color maps.
         * **0** resets the colors to the original colormap and scaling.
           The default color map is 'jet'.
         * **i** (or 'I') toggles between **inverse** and normal scaling.
         * **1** sets the colormap scaling to **linear**
         * **2** sets the colormap scaling to **logarithmic**
         * **3** sets the colormap scaling to **exponential**
         * **4** sets the colormap scaling to **square root**
         * **5** sets the colormap scaling to **square**
         * **b** (or 'B') changes color of **bad** pixels.
         * **h** (or 'H') replaces the current data by a **histogram equalized**
           version of this data. This key toggles between the original
           data and the equalized data.
         * **z** (or 'Z') replaces the current data by a **smoothed**
           version of this data. This key is a toggle between
           the original data and the blurred version, smoothed
           with a value of sigma set by key 'x'. Pressing 'x' repeatedly
           increases the smoothing factor. Note that Not a Number (NaN) values
           are smoothed to 0.0.
         * **x** (or 'X") increases the smoothing factor. The number of steps is
           10. Then is starts again with step 1.
         * **m** (or 'M') saves current colormap look up data to a file.
           The default name of the file is the name of file from which the data
           was extracted or the name given in the constructor. The name is
           appended with '.lut'. This data is written in the
           right format so that it can be be (re)used as input colormap.
           This way you can fix a color setting and reproduce the same setting
           in another run of
           a program that allows one to enter a colormap from file.


      If *annim* is an object from class :class:`Annotatedimage` then activate
      color editing with:

      >>> fits = maputils.FITSimage("m101.fits")
      >>> fig = plt.figure()
      >>> frame = fig.add_subplot(1,1,1)
      >>> annim = fits.Annotatedimage(frame)
      >>> annim.Image()
      >>> annim.interact_imagecolors()
      >>> annim.plot()
      """
      #--------------------------------------------------------------------
      self.imagecolorsmouse = AxesCallback(self.mouse_imagecolors, self.frame, 'motion_notify_event')
      self.imagecolorskey = AxesCallback(self.key_imagecolors, self.frame, 'key_press_event')
      #self.imagecolorskey = AxesCallback(self.key_imagecolors, self.frame, 'motion_notify_event')
      #self.imagecolorsmouse = AxesCallback(self.mouse_imagecolors, self.frame, 'motion_notify_event')
      # Set interaction with colormap on. We postponed this until now because
      # setting it before plot operations ruins your graticule frames with versions
      # of Matplotlib >= 0.99. This seems to be caused by an unvoluntary canvas.draw()
      # in module mplutil for color map updates for a sequence of images with the same
      # color map (subsets/slices). The constructor of the Annotatedimage class
      # sets auto to False. Here we set it to True because only here we require
      # interaction.
      self.cmap.auto = True
      flushprint("Add imagecolors mouse and key callbacks %d %d for object %d"%(id(self.imagecolorskey),
      id(self.imagecolorsmouse), id(self)))
      self.AxesCallback_ids.append(self.imagecolorskey)
      self.AxesCallback_ids.append(self.imagecolorsmouse)


   def mouse_writepos(self, axesevent):
      #--------------------------------------------------------------------
      """
      Print position information of the position where
      you clicked with the right mouse button while pressing
      the SHIFT key. Print the info on the command line or
      GIPSY command line cq. Log file.
      Register this function with an event handler,
      see example.
   
      :param axesevent:
         AxesCallback event object with pixel position information.
      :type axesevent:
         AxesCallback instance
   
      :Example:
         Register this callback function for object *annim* with:
   
         >>> annim.interact_writepos()
      """
      #--------------------------------------------------------------------
      condition = axesevent.event.button == 1 and axesevent.event.key == 'shift'
      if not condition:
         return

      if self.figmanager.toolbar.mode == '':
         x, y = axesevent.xdata, axesevent.ydata
         spix, swcs, swcsuf, sz = self.positionmessage(x, y, axesevent.posobj, parts=True)
         s = ' '   # One space to separate multiple entries on one line
         if spix:
            s += spix
         if swcs:
            if s:
               s += ' '
            s += swcs
         if swcsuf:            # The unformatted world coordinate
            if s:
               s += ' '
            s += swcsuf            
         if sz:
            if s:
               s += ' '
            s += sz
        
         if s == ' ':
            return
         if axesevent.gipsy and gipsymod:
            if axesevent.g_appendcr:
               s += '\r'   # Add carriage return
            if axesevent.g_typecli:
               typecli(s)  # Write to Hermes command line
            if axesevent.g_tolog:
               anyout(s)   # Write to Hermes log file and screen
         else:
            print s


   def interact_writepos(self, pixfmt="%.1f", dmsprec=1, wcsfmt="%.3g", zfmt="%.3e",
                         hmsdms=True, grids=True, world=True, worlduf=False, imval=True,
                         gipsy=False, g_typecli=False,
                         g_tolog=False, g_appendcr=False ):                            
      #--------------------------------------------------------------------
      """
      Add mouse interaction (left mouse button) to write the position
      of the mouse to screen. The position is written both in pixel
      coordinates and world coordinates.

      
      :param pixfmt:
         Python number format for pixel coordinates
      :type pixfmt:
         String
      :param wcsfmt:
         Python number format for wcs coordinates if the coordinates
         are not spatial or if parameter *hmsdms* is False.
      :type wcsfmt:
         String
      :param zfmt:
         Python number format for image value(s)
      :type pixfmt:
         String
      :param hmsdms:
         If True (default) then spatial coordinates will be formatted
         in hours/degrees, minutes and seconds according to the current sky system.
         The precision in seconds is entered with parameter *dmsprec*.
      :type hmsdms:
         Boolean
      :param dmsprec:
         Number of decimal digits in seconds for coordinates formatted in
         in HMS/DMS
      :type dmsprec:
         Integer
      :param gipsy:
         If set to True, the output is written with GIPSY function anyout() to
         screen and a log file.
      :type gipsy:
         Boolean
      :param typecli:
         If True then write the positions on the Hermes command line instead of the
         log file and screen.
      :type typecli:
         Boolean
      
      :Example:

      >>> fits = maputils.FITSimage("m101.fits")
      >>> fig = plt.figure()
      >>> frame = fig.add_subplot(1,1,1)
      >>> annim = fits.Annotatedimage(frame)
      >>> annim.Image()
      >>> annim.interact_writepos()
      >>> annim.plot()

      For a formatted output one could add parameters to *interact_writepos()*.
      The next line writes no pixel coordinates, writes spatial coordinates
      in degrees (not in HMS/DMS format) and adds a format for
      the world coordinates and the image value(s).

      >>> annim.interact_writepos(pixfmt=None, wcsfmt="%.12f", zfmt="%.3e",
                                  hmsdms=False)
      """
      #--------------------------------------------------------------------
      posobj = Positionmessage(self.projection.skysys, self.skyout, self.projection.types)
      posobj.pixfmt = posobj.wcsfmt = posobj.wcsuffmt = posobj.zfmt = None
      if grids:
         posobj.pixfmt = pixfmt
      if world:
         posobj.wcsfmt = wcsfmt
      if worlduf:               # User wants world coordinates in decimal degrees
         if wcsfmt:
            posobj.wcsuffmt = wcsfmt
      if imval:         
         posobj.zfmt = zfmt
      posobj.hmsdms = hmsdms
      posobj.dmsprec = dmsprec
      if not gipsy:
         g_typecli = False      
      self.writeposmouse = AxesCallback(self.mouse_writepos, self.frame,
                           'button_press_event',
                            gipsy=gipsy, g_typecli=g_typecli,
                            g_tolog=g_tolog, g_appendcr=g_appendcr,
                            posobj=posobj)
      self.AxesCallback_ids.append(self.writeposmouse)
      flushprint("Add mouse writeposmouse callback %d for object %d"%(id(self.writeposmouse), id(self)))


   def motion_events(self):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
      It reacts on mouse movements. During these movements, a message
      with information about the position of the cursor in pixels
      and world coordinates is displayed on the toolbar.
      """
      #--------------------------------------------------------------------
      #      
      self.cidmove = AxesCallback(self.on_move, self.frame, 'motion_notify_event')
      self.AxesCallback_ids.append(self.cidmove)



   def key_events(self):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
      using keys.
      """
      #--------------------------------------------------------------------
      #self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.key_pressed)
      self.cidkey = AxesCallback(self.key_pressed, self.frame, 'key_press_event')
      self.AxesCallback_ids.append(self.cidkey)


   def click_events(self):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
      It reacts on pressing the left mouse button and prints a message
      to stdout with information about the position of the cursor in pixels
      and world coordinates.
      """
      #--------------------------------------------------------------------
      # self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
      self.cidclick = AxesCallback(self.on_click, self.frame, 'button_press_event')
      self.AxesCallback_ids.append(self.cidclick)
      

   def positionsfromfile(self, filename, comment, skyout=None, **kwargs):
      #--------------------------------------------------------------------
      """
      Read positions from a file with world coordinates and convert them to
      pixel coordinates. The interface is exactly the same as from method
      :meth:`tabarray.readColumns()`

      It expects that the first column you specify contains the longitudes and
      the second column that is specified the latitudes.

      :param filename:
         Name (and pahth if necessary) of the file which contains longitudes
         and latitudes.
      :type filename:
         String
      :param comment:
         Comment characters. If a line starts with a comment character,
         it will be skipped.
      :type comment:
         String
      :param skyout:
         Tell the system in what sky system your longitudes and latitudes are.
      :type skyout:
         Sky definition
      :param kwargs:
         Keywords for Tabarray's method readColumns.
      :type kwargs:
         Python keyword arguments

      :Examples:
      
      >>> fn = 'smallworld.txt'
      >>> xp, yp = annim.positionsfromfile(fn, 's', cols=[0,1])
      >>> frame.plot(xp, yp, ',', color='#FFDAAA')

      Or: your graticule is equatorial but the coordinates in the file are
      galactic:
      
      >>> xp, yp = annim.positionsfromfile(fn, 's', skyout='ga', cols=[0,1])
      """
      #--------------------------------------------------------------------
      ainv = self.projection.allow_invalid
      if not skyout is None:
         skyout_old = self.projection.skyout
         self.projection.skyout = skyout
      self.projection.allow_invalid = True
      lon, lat= readColumns(filename, comment, **kwargs)
      xp, yp = self.projection.topixel((lon,lat))
      xp = numpy.ma.masked_where(numpy.isnan(xp) | (xp > self.pxlim[1]) | (xp < self.pxlim[0]), xp)
      yp = numpy.ma.masked_where(numpy.isnan(yp) | (yp > self.pylim[1]) | (yp < self.pylim[0]), yp)
      self.projection.allow_invalid = ainv       # Reset status for invalid transformations
      if not skyout is None:
          # Restore
         self.projection.skyout = skyout_old
      return xp, yp


   def getflux(self, xy, pixelstep=None):
      # Return Area in pixels and sum of image values in
      # polygon defined by xy

      if pixelstep is None:
         pixelstep = self.pixelstep
      poly = numpy.asarray(xy)
      mm = poly.min(0)
      xmin = mm[0]; ymin = mm[1]
      mm = poly.max(0)
      xmax = mm[0]; ymax = mm[1]
      xmin = numpy.floor(xmin); xmax = numpy.ceil(xmax)
      ymin = numpy.floor(ymin); ymax = numpy.ceil(ymax)
      Y = numpy.arange(ymin,ymax+1, pixelstep)
      X = numpy.arange(xmin,xmax+1, pixelstep)
      l = int(xmax-xmin+1); b = int(ymax-ymin+1)
      numpoints = len(X)*len(Y)
      x, y = numpy.meshgrid(X, Y)
      pos = zip(x.flatten(), y.flatten())
      pos = numpy.asarray(pos)
      mask = nxutils.points_inside_poly(pos, poly)

      # Get indices for positions inside shape using mask
      i = numpy.arange(len(pos), dtype='int')
      # Get a filtered array
      posm = pos[i[numpy.where(mask)]]

      xy = posm.T
      x = xy[0]
      y = xy[1]
      # Check inside pixels:
      # self.frame.plot(x, y, 'r')
      # Correction of array index consists of three elements:
      # 1) The start position of the box must map on the start position of the array
      # 2) Positions all > 0.5 so add 0.5 and take int to avoid the need of a round function
      # 3) The data could be a limited part of the total, but the mouse positions
      #    are relative to (1,1).
      xcor = self.pxlim[0] - 0.5
      ycor = self.pylim[0] - 0.5
      x = numpy.asarray(x-xcor, dtype='int')
      y = numpy.asarray(y-ycor, dtype='int')

      # Compose the array with the intensities at positions inside the polygon
      z = self.data[y,x]
      area = len(posm)*pixelstep*pixelstep
      sum = z.sum()*(pixelstep*pixelstep)

      """
      # Old algorithm
      count = 0
      sum = 0.0
      for i, xy in enumerate(pos):
         if mask[i]:
            xp = int(xy[0] - xcor)
            yp = int(xy[1] - ycor)
            z = self.data[yp,xp]
            sum += z
            count += 1
      c1 = time.clock()
      print "2: Calculated in %f cpu seconds" % ((c1-c0))
      print count*(pixelstep*pixelstep), sum*(pixelstep*pixelstep)
      """
      return area, sum


class FITSaxis(object):
#-----------------------------------------------------------------
   """
This class defines objects which store WCS information from a FITS
header. It includes axis number and alternate header information
in a FITS keyword.
    
:param axisnr:
   FITS axis number. For this number the relevant keys in the header
   are read.
:type axisnr:
   Integer
:param hdr:
   FITS header
:type hdr:
   pyfits.NP_pyfits.Header instance

   
:Methods:

.. automethod:: printattr
.. automethod:: printinfo

   """  
#--------------------------------------------------------------------
    
   def __init__(self, axisnr, hdr, alter):
      def makekey(key, alternate=True):
         s = "%s%d" %(key, axisnr)
         if alter != '' and alternate:
            s += alter
         return s

      ax = makekey("CTYPE")
      self.ctype = 'Unknown'
      self.axname = 'Unknown'
      self.axext = ''
      if hdr.has_key(ax):
         self.ctype = hdr[ax].upper()
         info = hdr[ax].split('-')
         self.axname = info[0].upper()
         if len(info) > 1:
            ext = info[-1].upper()  # Last splitted string must be extension. Skip the hyphens
            extlist = {    'ARC':'Zenithal equidistant projection',
                           'AIR':'Airy projection',
                           'AZP':'Slant zenithal (azimuthal) perspective projection',
                           'AIT':"Hammer Aitoff projection",
                           'BON':"Bonne's equal area projection",
                           'CAR':'Plate Carree',
                           'CEA':"Lambert's equal area projection",
                           'COD':"Conic equidistant projection",
                           'COE':"Conic equal area projection",
                           'COO':"Conic orthomorfic projection",
                           'COP':"Conic perspective projection",
                           'CSC':"COBE quadrilateralized spherical cube projection",
                           'CYP':"Gall's stereographic projection",
                           'MER':"Mercator's projection",
                           'MOL':"Mollweide's projection",
                           'NCP':"Northern celestial pole projection",
                           'PAR':'Parabolic projection',
                           'PCO':"Polyconic",
                           'POL':"Polyconic projection",
                           'QSC':"Quadrilateralized spherical cube projection",
                           'SIN':'Slant orthograpic projection',
                           'SFL':'Sanson-Flamsteed projection',
                           'STG':'Stereographic projection',
                           'SZP':'Slant zenithal perspective',
                           'TAN':'Gnomonic projection',
                           'TSC':"Tangential spherical cube projection",
                           'ZEA':'Zenith equal area projection',
                           'ZPN':'Zenithal polynomial projection'}
            self.axext = ext
            flushprint("ext, haskey=%s %s"%(ext, str(extlist.has_key(ext))))
            if extlist.has_key(ext):
               self.axext += ' (' + extlist[ext] +')'
      else:
         self.ctype = "X%d"%axisnr
         self.axname = self.ctype

      ai = makekey("NAXIS", alternate=False)
      self.axlen = hdr[ai]
      self.axisnr = axisnr
      self.axstart = 1
      self.axend = self.axlen

      ai = makekey("CDELT")
      if hdr.has_key(ai):
         self.cdelt = hdr[ai]
      else:
         self.cdelt = 1.0
      ai = makekey("CRVAL")
      if hdr.has_key(ai):
         self.crval = hdr[ai]
      else:
         self.crval = 1.0
      ai = makekey("CUNIT")          # Is sometimes omitted, so check first.
      if hdr.has_key(ai):
         self.cunit = hdr[ai]
      else:
         self.unit = '?'
      ai = makekey("CRPIX")
      if hdr.has_key(ai):
         self.crpix = hdr[ai]
      else:
         self.crpix = self.axlen/2
      ai = makekey("CROTA")          # Not for all axes
      self.crota = 0
      if hdr.has_key(ai):
         self.crota = hdr[ai]

      self.wcstype = None
      self.wcsunits = None
      self.outsidepix = None


   def printattr(self):
   #----------------------------------------------------------
      """
   Print formatted information for this axis.
   
   :Examples:
      
   >>> from kapteyn import maputils
   >>> fitsobject = maputils.FITSimage('rense.fits')
   >>> fitsobject.hdr
   <pyfits.NP_pyfits.Header instance at 0x1cae3170>
   >>> ax1 = maputils.FITSaxis(1, fitsobject.hdr)
   >>> ax1.printattr()
   axisnr     - Axis number:  1
   axlen      - Length of axis in pixels (NAXIS):  100
   ctype      - Type of axis (CTYPE):  RA---NCP
   axname     - Short axis name:  RA
   cdelt      - Pixel size:  -0.007165998823
   crpix      - Reference pixel:  51.0
   crval      - World coordinate at reference pixel:  -51.2820847959
   cunit      - Unit of world coordinate:  DEGREE
   wcstype    - Axis type according to WCSLIB:  None
   wcsunits   - Axis units according to WCSLIB:  None
   outsidepix - A position on an axis that does not belong to an image:  None
   
   If we set the image axes in *fitsobject* then the WCS attributes
   will get a value also. This object stores its FITSaxis objects in a list
   called *axisinfo[]*. The index is the required FITS axis number.
   
   >>> fitsobject.set_imageaxes(1, 2, 30)
   >>> fitsobject.axisinfo[1].printattr()
   ..........
   wcstype    - Axis type according to WCSLIB:  longitude
   wcsunits   - Axis units according to WCSLIB:  deg
   outsidepix - A position on an axis that does not belong to an image:  None

      """
   #----------------------------------------------------------
      s = "axisnr     - Axis number: %s\n" % self.axisnr \
        + "axlen      - Length of axis in pixels (NAXIS): %s\n" % self.axlen \
        + "axstart    - First pixel coordinate: %s\n" % self.axstart \
        + "axend      - Last pixel coordinate: %s\n" % self.axend \
        + "ctype      - Type of axis (CTYPE): %s\n" % self.ctype \
        + "axname     - Short axis name: %s\n" % self.axname \
        + "axext      - Axis extension: : %s\n" % self.axext \
        + "cdelt      - Pixel size: %s\n" % self.cdelt \
        + "crpix      - Reference pixel: %s\n" % self.crpix \
        + "crval      - World coordinate at reference pixel: %s\n" % self.crval \
        + "cunit      - Unit of world coordinate: %s\n" % self.cunit \
        + "crota      - Rotation of axis: %s\n" % self.crota \
        + "wcstype    - Axis type according to WCSLIB: %s\n" % self.wcstype \
        + "wcsunits   - Axis units according to WCSLIB: %s\n" % self.wcsunits \
        + "outsidepix - A position on an axis that does not belong to an image: %s\n" % self.outsidepix

      return s
   

   def printinfo(self):
   #----------------------------------------------------------
      """
   Print formatted information for this axis.
   
   :Examples:
   
      >>> from kapteyn import maputils
      >>> fitsobject = maputils.FITSimage('rense.fits')
      >>> ax1 = maputils.FITSaxis(1, fitsobject.hdr)
      >>> ax1.printinfo()
      Axis 1: RA---NCP  from pixel 1 to   512
      {crpix=257 crval=178.779 cdelt=-0.0012 (DEGREE)}
      {wcs type=longitude, wcs unit=deg}
      Axis 2: DEC--NCP  from pixel 1 to   512
      {crpix=257 crval=53.655 cdelt=0.00149716 (DEGREE)}
      {wcs type=latitude, wcs unit=deg}
      

   :Notes:
      if attributes for a :class:`maputils.FITSimage` object are changed
      then the relevant axis properties are updated.
      So this method can return different results depending
      on when it is used.
      """
   #----------------------------------------------------------
      a = self
      s = "Axis %d: %-9s from pixel %5d to %5d\n  {crpix=%d crval=%G cdelt=%g (%s)}\n  {wcs type=%s, wcs unit=%s}" % (
                  a.axisnr,
                  a.ctype,
                  a.axstart,
                  a.axend,
                  a.crpix,
                  a.crval,
                  a.cdelt,
                  a.cunit,
                  a.wcstype,
                  a.wcsunits)
      return s


class FITSimage(object):
#-----------------------------------------------------------------
   """
This class extracts 2D image data from FITS files. It allows for
external functions to prompt users for relevant input like the
name of the FITS file, which header in that file should be used,
the axis numbers of the image axes, the pixel
limits and a spectral translation if one of the selected axes
is a spectral axis.
All the methods in this class that allow these external functions
for prompting can also be used without these functions. Then one needs
to know the properties of the FITS data beforehand.
   
:param filespec:
   A default file either to open directly
   or to be used in a prompt as default file. This variable
   should have a value if no external function is used to prompt a user.
:type filespec:
   String
:param promptfie:
   A user supplied function which should prompt a
   user for some data, opens the FITS file and returns the hdu
   list and a user selected index for the header from this hdu
   list. An example of a function supplied by
   :mod:`maputils` is function :func:`prompt_fitsfile`
:type promptfie:
   Python function
:param hdunr:
   A preset of the index of the header from the hdu list.
   If this variable is set then it should not prompted for in the
   user supplied function *promptfie*.
:type hdunr:
   Integer
:param alter:
   Selects an alternate header for the world coordinate system.
   Default is the standard header.
   Keywords in alternate headers end on a character A..Z
:type alter:
   Empty or a single character. Input is case insensitive.
:param memmap:
   Set the memory mapping for PyFITS. The default is copied from the
   default in your version of PyFITS. If you want to be sure it is on
   then specify *memmap=1*
:type memmap:
   Boolean
:param externalheader:
   If defined, then it is a header from an external source e.g. a user
   defined header.
:type externalheader:
   Python dictionary
:param externaldata:
   If defined, then it is data from an external source e.g. user
   defined data or processed data in a numpy array. A user/programmer
   should check if the shape of the numpy array fits the sizes given
   in FITS keywords *NAXISn*.
:type externaldata:
   Numpy array
:param parms:
   Extra parameters for PyFITS's *open()* method, such as
   *uint16*, *ignore_missing_end*, *checksum*, see PyFITS documentation
   for their meaning.
:type parms:
   keyword arguments

:Attributes:
   
    .. attribute:: filename

       Name of the FITS file (read-only).
       
    .. attribute:: hdr

       Header as read from the header (read-only).
       
    .. attribute:: naxis

       Number of axes (read-only).
       
    .. attribute:: dat

       The raw image data (not sliced, swapped or limited in range).
       The required sliced image data is stored in attribute :attr:`boxdat`.
       This is a read-only attribute. 
       
    .. attribute:: axperm

       Axis permutation array. These are the (FITS) axis numbers of your
       image x & y axis.

    .. attribute:: wcstypes

       Type of the axes in this data. The order is the same as of the axes.
       The types ara strings and are derived from attribute wcstype of the
       Projection object. The types are:
       'lo' is longitude axis. 'la' is latitude axis,
       'sp' is spectral axis. 'li' is a linear axis. Appended to 'li' is an
       underscore and the ctype of that axis (e.g. 'li_stokes').
       
    .. attribute:: mixpix

       The missing pixel if the image has only one spatial axis. The other
       world coordinate could be calculated with a so called *mixed* method
       which allows for one world coordinate and one pixel.
       
    .. attribute:: axisinfo

       A list with :class:`FITSaxis` objects. One for each axis. The index is
       an axis number (starting at 1).
       
    .. attribute:: slicepos

       A list with position on axes in the FITS file which do not belong to
       the required image.
       
    .. attribute:: pxlim

       Axis limit in pixels. This is a tuple or list (xlo, xhi).
       
    .. attribute:: pylim

       Axis limit in pixels. This is a tuple or list (xlo, xhi).
       
    .. attribute:: boxdat

       The image data. Possibly sliced, axis swapped and limited in axis range.
       
    .. attribute:: imshape

       Sizes of the 2D array in :attr:`boxdat`.
       
    .. attribute:: spectrans

       A string that sets the spectra translation. If one uses the prompt function
       for the image axes, then you will get a list of possible translations for the
       spectral axis in your image.

    .. attribute:: proj

       An object from :class:`wcs.Projection`. This object is the result of the call:
       ``proj = wcs.Projection(self.hdr)``, so it is the Projection object that involves all
       the axes in the FITS header.
    
    .. attribute:: convproj

       An object from :class:`wcs.Projection`. This object is needed to
       be able to use methods *toworld()* and *topixel()* for the
       current image.
       
    .. attribute:: figsize

       A suggested figure size (inches) in X and Y directions.
       
    .. attribute:: aspectratio

       Plot a circle in **world coordinates** as a circle. That is, if the
       pixel size in the FITS header differs in X and Y, then correct
       the (plot) size of the pixels with value *aspectratio*
       so that features in an image have the correct sizes in longitude and
       latitude in degrees.
      
:Notes:
   The object is initialized with a default position for a data slice if
   the dimension of the FITS data is > 2. This position is either the value
   of CRPIX from the header or 1 if CRPIX is outside the range [1, NAXIS].

   Values -inf and +inf in a dataset are replaced by NaN's (not a number number).
   We know that Matplotlib's methods have problems with these values, but these
   methods can deal with NaN's.

:Examples:
   PyFITS allows URL's to retrieve FITS files. It can also read gzipped files e.g.:
   
      >>> f = 'http://www.atnf.csiro.au/people/mcalabre/data/WCS/1904-66_ZPN.fits.gz'
      >>> fitsobject = maputils.FITSimage(f)
      >>> print fitsobject.str_axisinfo()
      Axis 1: RA---ZPN  from pixel 1 to   192
        {crpix=-183 crval=0 cdelt=-0.0666667 (Unknown)}
        {wcs type=longitude, wcs unit=deg}
      Axis 2: DEC--ZPN  from pixel 1 to   192
        {crpix=22 crval=-90 cdelt=0.0666667 (Unknown)}
        {wcs type=latitude, wcs unit=deg}

   Use Maputil's prompt function :func:`prompt_fitsfile` to get
   user interaction for the FITS file specification.
   
      >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)

:Methods:

.. index:: Select image data from FITS file
.. automethod:: set_imageaxes
.. index:: Set pixel limits of image axes
.. automethod:: set_limits
.. index:: Set spectral translation
.. automethod:: set_spectrans
.. index:: Set output sky
.. automethod:: set_skyout
.. index:: A class for plotting FITS data
.. automethod:: Annotatedimage
.. index:: Aspect ratio from FITS header data
.. automethod:: get_pixelaspectratio
.. index:: Set default figure size for Matplotlib
.. automethod:: get_figsize
.. index:: Print information from FITS header
.. automethod:: str_header
.. automethod:: str_axisinfo
.. automethod:: str_wcsinfo
.. automethod:: str_spectrans
.. automethod:: get_dataminmax
.. automethod:: slice2world
.. automethod:: header2classic
.. automethod:: reproject_to
.. automethod:: writetofits

   """

#--------------------------------------------------------------------
   def __init__(self, filespec=None, promptfie=None, prompt=True, hdunr=None, alter='', memmap=None,
                externalheader=None, externaldata=None, externalname="artificial",
                **parms):
      #----------------------------------------------------
      # Usually the required header and data are extracted
      # from a FITS file. But it is also possible to provide
      # header and data from an external source. Then these
      # must be processed instead of other keyword parameters
      #-----------------------------------------------------
      if externalheader is not None:
         self.hdr = externalheader
         self.bitpix, self.bzero, self.bscale, self.blank = getscale(self.hdr)
         self.filename = externalname
         if externaldata is not None:
            if 'int' in externaldata.dtype.name:
               # So this is integer data without bscale, bzero or blank
               self.dat = externaldata.astype(numpy.float32)
               flushprint("I just converted data to float for externaldata")
            else:
               self.dat = externaldata
         else:
            self.dat = None
         #self.dat = externaldata
      else:
         # Not an external header, so a file is given or user wants to be prompted.
         if promptfie:
            if memmap is None:  # Use default of current PyFITS version
               hdulist, hdunr, filename, alter = promptfie(filespec, prompt, hdunr, alter)
            else:
               hdulist, hdunr, filename, alter = promptfie(filespec, prompt, hdunr, alter, memmap)
         else:
            try:
               if memmap is None:
                  hdulist = pyfits.open(filespec, **parms)
               else:
                  hdulist = pyfits.open(filespec, memmap=memmap, **parms)
               filename = filespec
            except IOError, (errno, strerror):
               print "Cannot open FITS file: I/O error(%s): %s" % (errno, strerror)
               raise
            except:
               print "Cannot open file, unknown error!"
               raise
            if hdunr is None:
               hdunr = 0

         self.hdunr = hdunr
         hdu = hdulist[hdunr]
         self.filename = filename
         self.hdr = hdu.header
         self.bitpix, self.bzero, self.bscale, self.blank = getscale(self.hdr)

         # If an input array is of type integer then it is converted to
         # float32. Then we can use NaN's in the data as a replacement for BLANKS
         # Note that if scaled data is read then the scaling is applied first
         # and the type is converted to float32
         # Due to added code in PyFITS 1.3 to deal with
         # blanks, the conversion also takes place if we have
         # integer data and keyword BLANK is available. In that case
         # the integer array is automatically converted to float64 and
         # the BLANK values are converted to NaN.
         if externaldata is None:
            if 'int' in hdu.data.dtype.name:
               # So this is integer data without bscale, bzero or blank
               self.dat = hdu.data.astype(numpy.float32)
            else:
               self.dat = hdu.data
         else:
            #Do not check type
            self.dat = externaldata
         hdulist.close()             # Close the FITS file

      # IMPORTANT
      # Some FITS writers write their -32 bitpix blanks as -inf,
      # but there can also be other sources that insert +inf or -inf
      # values in a data set. Most methods in Matplotlib (e.g. to
      # create images or contours) cannot cope with these -inf and inf
      # values. So to be save, we treat those as NaN and therefore 
      # replace +/-inf's by NaN's
      """
      try:
         self.dat[-numpy.isfinite(self.dat)] = numpy.nan
      except:
         pass
      """
      # An alternate header can also be specified for an external header
      self.alter = alter.upper()

      # Test on the required minimum number of axes (2)
      self.naxis = self.hdr['NAXIS']
      if self.naxis < 2:
         print "You need at least two axes in your FITS file to extract a 2D image."
         print "Number of axes in your FITS file is %d" % (self.naxis,)
         hdulist.close()
         raise Exception, "Number of data axes must be >= 2."
      

      self.axperm = [1,2]
      self.mixpix = None 
      # self.convproj = wcs.Projection(self.hdr).sub(self.axperm)
      axinf = {}
      for i in range(self.naxis):
         # Note that here we define the relation between index (i) and
         # the axis number (i+1)
         # The axinf dictionary has this axis number as key values 
         axisnr = i + 1         
         axinf[axisnr] = FITSaxis(axisnr, self.hdr, self.alter)
      self.axisinfo = axinf

      slicepos = []                  # Set default positions (CRPIXn) on axes outside image for slicing data
      sliceaxnames = []
      sliceaxnums = []
      n = self.naxis
      if (n > 2):
         for i in range(n):
            axnr = i + 1
            if axnr not in self.axperm:
               crpix = self.axisinfo[axnr].crpix
               # If CRPIX in the header is negative (e.g. in a
               # manipulated header) then this corresponds to a non-existing data
               # slice. In that case we take the first slice in
               # a sequence as the default.
               if crpix < 1 or crpix > self.axisinfo[axnr].axlen:
                  crpix = 1
               slicepos.append(crpix)
               sliceaxnames.append(self.axisinfo[axnr].axname)
               sliceaxnums.append(axnr)
      self.slicepos = slicepos
      self.sliceaxnames = sliceaxnames
      self.sliceaxnums = sliceaxnums

      n1 = self.axisinfo[self.axperm[0]].axlen
      n2 = self.axisinfo[self.axperm[1]].axlen
      self.pxlim = [1, n1]
      self.pylim = [1, n2]
      self.proj = wcs.Projection(self.hdr, alter=self.alter)
      # Attribute 'gridmode' is False by default
      for i in range(n):
         ax = i + 1
         # The index for axinfo starts with 1.
         self.axisinfo[ax].wcstype = self.proj.types[i]
         self.axisinfo[ax].wcsunits = self.proj.units[i]
         #self.axisinfo[ax].cdelt = self.proj.cdelt[i]
         #if self.alter != '':
         self.axisinfo[ax].cdelt = self.proj.cdelt[i]
         self.axisinfo[ax].crval = self.proj.crval[i]
         self.axisinfo[ax].ctype = self.proj.ctype[i]
         self.axisinfo[ax].cunit = self.proj.cunit[i]
         self.axisinfo[ax].crpix = self.proj.crpix[i]

      # Set the type of axis in the original system.
      wcstypes = []
      for i in range(n):
         ax = i + 1
         if not self.proj.lonaxnum is None and ax == self.proj.lonaxnum:
            wcstypes.append('lo')
         elif not self.proj.lataxnum is None and ax == self.proj.lataxnum:
            wcstypes.append('la')
         elif not self.proj.specaxnum is None and ax == self.proj.specaxnum:
            wcstypes.append('sp')
         else:
            # To distinguish linear types we append the ctype of this axis.
            wcstypes.append('li_' + self.proj.ctype[i])
      self.wcstypes = wcstypes

      self.spectrans = None    # Set the spectral translation
      self.skyout = None       # Must be set before call to set_imageaxes
      self.boxdat = self.dat
      # We set two axes as the default axes of an image. Then the attribute
      # 'convproj' becomes available after a call to set_imageaxes()
      self.set_imageaxes(self.axperm[0], self.axperm[1], self.slicepos)
      self.aspectratio = None
      self.pixelaspectratio = self.get_pixelaspectratio()
      self.figsize = None      # TODO is dit nog belangrijk??
      

   def slice2world(self, skyout=None, spectra=None, userunits=None):
      #-----------------------------------------------------------------
      """
      Given the pixel coordinates of a slice, return the world
      coordinates of these pixel positions and their units.
      For example in a 3-D radio cube with axes RA-DEC-FREQ one can have
      several RA-DEC images as function of FREQ. This FREQ is given
      in pixels coordinates in attribute *slicepos*. The world coordinates
      are calculated using the Projection object which is
      also an attribute.

      :param skyout:
         Set current projection object in new output sky mode
      :type skyout:
         String or tuple representing sky definition

      :param spectra:
         Use this spectral translation for the output world coordinates
      :type spectra:
         String

      :param userunits:
         A sequence of units as the user wants to have it appear
         in the slice info string. The order of these units must
         be equal to the order of the axes outside the slice/subset.
         Both the world coordinates and the units are adjusted. 
      :type userunits:
         String
          
      :Returns:
         A tuple with two elements: *world* and *units*.
         Element *world* is either an empty list or a list with
         one or more world coordinates. The number of coordinates is
         equal to the number of axes in a data set that do not
         belong to the extracted data which can be a slice.
         For each world coordinate there is a unit in element *units*.

      :Note:
         This method first calculates a complete set of world coordinates.
         Where it did not define a slice position, it takes the header
         value *CRPIXn*. So if a map is defined with only one spatial axes and
         the missing spatial axis is found in *slicepos* than we have two
         matching pixel coordinates for which we can calculate world coordinates.
         So by definition, if a slice is a function of a spatial
         coordinate, then its world coordinate is found by using the matching
         pixel coordinate which, in case of a spatial map, corresponds to the
         projection center.

      :Example:
      
         >>> vel, uni = fitsobj.slice2world(spectra="VOPT-???")
         >>> velinfo = "ch%d = %.1f km/s" % (ch, vel[0]/1000.0)

         or:
         >>> vel, uni = fitsobj.slice2world(spectra="VOPT-???", userunits="km/s")
      """
      #-----------------------------------------------------------------
      # Is there something to do?
      if self.slicepos is None:
         return None, None
      pix = []
      units = []
      world = []
      j = 0
      skyout_old = self.proj.skyout
      if skyout != None:
         self.proj.skyout = skyout
      if spectra != None:
         newproj = self.proj.spectra(spectra)
      else:
         newproj = self.proj
      for i in range(self.naxis):
         if (i+1) in self.axperm:
            pix.append(self.proj.crpix[i])
         else:
            pix.append(self.slicepos[j])
            j += 1
            # Note that we assumed the axis order in slicepos is the same as in the Projection object
      s = "TEST set 1dim tuple pix=%s"%str(tuple(pix))
      flushprint(s)
      wor = newproj.toworld(tuple(pix))
      for i in range(self.naxis):
         if not (i+1) in self.axperm:
            world.append(wor[i])
            units.append(newproj.cunit[i])
      if skyout != None:
         self.proj.skyout = skyout_old

      # This method accepts a tuple with units given by a user.
      # The units are entered in the same order as the axes outside a subset/slice
      # Scaled versions
      if not userunits is None:
         sworld = []
         sunits = []
         if not issequence(userunits):
            userunits = [userunits]
         for w, u, uu in zip(world, units, userunits):
            if not uu is None:
               uf, errmes = unitfactor(u, uu)
               if uf is None:
                  raise ValueError(errmes)
               else:
                  w *= uf
                  u = uu
            sworld.append(w)
            sunits.append(u)
      else:
         # The original (native) values and units
         sworld = world
         sunits = units
      return sworld, sunits


   def get_dataminmax(self, box=False):
   #------------------------------------------------------------
      """
      Get minimum and maximum value of data in entire data structure
      defined by the current FITS header or in a slice. 
      These values can be important if
      you want to compare different images from the same source
      (e.g. channel maps in a radio data cube).

      :param box:
         Find min, max in data or if set to True in data slice (with limits).
      :type box:
         Boolean
         
      :Returns:
         min, max, two floating point numbers representing the minimum
         and maximum data value in data units of the header (*BUNIT*).

      :Note:
         We assume here that the data is read when a FITSobject was created.
         Then the data is filtered and the -inf, inf values are replaced 
         by NaN's.
         
      :Example:
         Note the difference between the min, max of the entire data or the
         min, max of the slice (limited by a box)::
   
            fitsobj = maputils.FITSimage('ngc6946.fits')
            vmin, vmax = fitsobj.get_dataminmax()
            for i, ch in enumerate(channels):
               fitsobj.set_imageaxes(lonaxnum, lataxnum, slicepos=ch)
               print "Min, max in this channel: ", fitsobj.get_dataminmax(box=True)


      """
   #------------------------------------------------------------
      if box:
         if self.boxdat is None:
            return 0, 1
         #mi = numpy.nanmin(self.boxdat)
         #ma = numpy.nanmax(self.boxdat)
         mask = numpy.isfinite(self.boxdat)
         mi = numpy.min(self.boxdat[mask])
         ma = numpy.max(self.boxdat[mask])
      else:
         #mi = numpy.nanmin(self.dat)
         #ma = numpy.nanmax(self.dat)
         mask = numpy.isfinite(self.dat)
         mi = numpy.min(self.dat[mask])
         ma = numpy.max(self.dat[mask])
      return mi, ma


   def str_header(self):
   #------------------------------------------------------------
      """
      Print the meta information from the selected header.
      Omit items of type *HISTORY*. It prints both real FITS headers
      and headers given by a dictionary.

      :Returns:
         A string with the header keywords

      :Examples:
         If you think a user needs more information from the header than
         can be provided with method :meth:`str_axisinfo` it can be useful to
         display the contents of the selected FITS header.
         This is the entire header and not a selected alternate header.

         >>> from kapteyn import maputils
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> print fitsobject.str_header()
         SIMPLE  =                    T / SIMPLE FITS FORMAT
         BITPIX  =                  -32 / NUMBER OF BITS PER PIXEL
         NAXIS   =                    3 / NUMBER OF AXES
         NAXIS1  =                  100 / LENGTH OF AXIS
         NAXIS2  =                  100 / LENGTH OF AXIS
         NAXIS3  =                  101 / LENGTH OF AXIS
         BLOCKED =                    T / TAPE MAY BE BLOCKED
         CDELT1  =  -7.165998823000E-03 / PRIMARY PIXEL SEPARATION
         CRPIX1  =   5.100000000000E+01 / PRIMARY REFERENCE PIXEL
         CRVAL1  =  -5.128208479590E+01 / PRIMARY REFERENCE VALUE
         CTYPE1  = 'RA---NCP          ' / PRIMARY AXIS NAME
         CUNIT1  = 'DEGREE            ' / PRIMARY AXIS UNITS
         etc. etc.

      """
   #------------------------------------------------------------
      st = ''
      if isinstance(self.hdr, dict):
         keylist = self.hdr.keys()
         keylist.sort()
         for k in keylist:
            s = self.hdr[k]
            if not str(s).startswith('HISTORY'):
               if type(s) == types_StringType:
                  val = "'" + "%-18s"%s + "'"
               else:
                  val = "%+20s"%str(s)
               st += "%-8s"%k + "= " + val + " /\n"
      else:
         for s in self.hdr.ascardlist():
            if not str(s).startswith('HISTORY'):
               st += "%s\n" % s
      return st



   def str_axisinfo(self, axnum=None, long=False):
   #------------------------------------------------------------
      """
      For each axis in the FITS header, return a string with the data related
      to the World Coordinate System (WCS).

      :param axnum:
         A list with axis numbers for which one wants to print information.
         These axis numbers are FITS numbers i.e. in range [1,NAXIS].
         To display information about the two image axes one should use
         attribute :attr:`maputils.FITSimage.axperm` as in the second example
         below.
      :type axnum:
         None, Integer or list with Integers
      :param long:
         If *True* then more verbose information is printed.
      :type long:
         Boolean

      :Returns:
         A string with WCS information for each axis in *axnum*.

      :Examples:
         Print useful header information after the input of the FITS file
         and just before the specification of the image axes:

         >>> from kapteyn import maputils
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> print fitsobject.str_axisinfo()
         Axis 1: RA---NCP  from pixel 1 to   100
           {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
           {wcs type=longitude, wcs unit=deg}
         Axis 2: DEC--NCP  from pixel 1 to   100
           {crpix=51 crval=60.1539 cdelt=0.007166 (DEGREE)}
           {wcs type=latitude, wcs unit=deg}
         Axis 3: VELO-HEL  from pixel 1 to   101
           {crpix=-20 crval=-243 cdelt=4200 (km/s)}
           {wcs type=spectral, wcs unit=m/s}

         Print extended information for the two image axes only:
      
         >>> print str_axisinfo(axnum=fitsobject.axperm, long=True)

        
      :Notes:
         For axis numbers outside the range of existing axes
         in the FITS file, nothing will be printed. No exception
         will be raised.
      """
   #------------------------------------------------------------
      if axnum is None:
         axnum = range(1, self.naxis+1)
      if not issequence(axnum):
         axnum = [axnum]
      s = ''
      l = len(axnum)
      for i, ax in enumerate(axnum):  # Note that the dictionary is unsorted. We want axes 1,2,3,...
         if ax >= 1 and ax <= self.naxis:
            a = self.axisinfo[ax]
            if long:
               s += a.printattr()
            else:
               s += a.printinfo()
            if i < l-1:
               s += '\n'
      return s
            


   def str_wcsinfo(self):
   #------------------------------------------------------------
      """
      Compose a string with information about
      the data related to the current World Coordinate System (WCS)
      (e.g. which axes are
      longitude, latitude or spectral axes)

      :Returns:
         String with WCS information for the current Projection
         object.

      :Examples: Print information related to the world coordinate system:

         >>> print fitsobject.str_wcsinfo()
         Current sky system:                 Equatorial
         reference system:                   ICRS
         Output sky system:                  Equatorial
         Output reference system:            ICRS
         projection's epoch:                 J2000.0
         Date of observation from DATE-OBS:  2002-04-04T09:42:42.1
         Date of observation from MJD-OBS:   None
         Axis number longitude axis:         1
         Axis number latitude axis:          2
         Axis number spectral axis:          None
         Allowed spectral translations:      None

      """
   #------------------------------------------------------------
      s = ''
      sys, ref, equinox, epoch = skyparser(self.convproj.skysys)
      if sys != None:     s +=  "Native sky system:                 %s\n" % skyrefsystems.id2fullname(sys)
      if ref != None:     s +=  "Native reference system:           %s\n" % skyrefsystems.id2fullname(ref)
      if equinox != None: s +=  "Native Equinox:                    %s\n" % equinox
      if epoch   != None: s +=  "Native date of observation:        %s\n" % epoch

      sys, ref, equinox, epoch = skyparser(self.convproj.skyout)
      if sys != None:     s +=  "Output sky system:                 %s\n" % skyrefsystems.id2fullname(sys)
      if ref != None:     s +=  "Output reference system:           %s\n" % skyrefsystems.id2fullname(ref)
      if equinox != None: s +=  "Output Equinox:                    %s\n" % equinox
      if epoch   != None: s +=  "Output date of observation:        %s\n" % epoch

      s +=  "Projection's epoch:                %s\n" % self.convproj.epoch
      s +=  "Date of observation from DATE-OBS: %s\n" % self.convproj.dateobs
      s +=  "Date of observation from MJD-OBS:  %s\n" % self.convproj.mjdobs
      s +=  "Axis number longitude axis:        %s\n" % self.convproj.lonaxnum
      s +=  "Axis number latitude axis:         %s\n" % self.convproj.lataxnum
      s +=  "Axis number spectral axis:         %s\n" % self.convproj.specaxnum
      s +=  "Selected spectral translation:     %s\n" % self.spectrans

      return s


   def str_spectrans(self):
   #------------------------------------------------------------
      """
      Compose a string with the possible spectral translations for this data.

      :Returns:
         String with information about the allowed spectral
         translations for the current Projection object.

      :Examples: Print allowed spectral translations:

         >>> print fitsobject.str_spectrans()
      
      """
   #------------------------------------------------------------
      s = ''
      isspectral = False
      for ax in self.axperm:
         if self.axisinfo[ax].wcstype == 'spectral':
            isspectral = True
      if not isspectral:
         return         # Silently
      i = 0
      for st, un in self.convproj.altspec:
         s += "%d   %s (%s)\n" % (i, st, un)
         i += 1 

      return s


   def getaxnumberbyname(self, axname):
      #--------------------------------------------------------------
      """
      Given an axis specification,  
      convert to the corresponding axis number. If the input was a number,
      then return this number. Note that there is no check on
      multiple matches of the minimal matched string.
      The string matching is case insensitive.

      :param axname:   The name of one of the axis in the header of this
                       object.
      :type axname:    String or Integer
      """
      #--------------------------------------------------------------
      if type(axname) != types_StringType:
         # Probably an integer
         return axname

      n = self.naxis
      # Get the axis number
      axnumber = None
      for i in range(n):
         ax = i + 1
         str2 = self.axisinfo[ax].axname
         if str2.find(string_upper(axname), 0, len(axname)) > -1:
             axnumber = ax
             break

      # Check on validity
      if axnumber is None:
         raise ValueError("Axis name [%s] is not found in the header"%axname)
      return axnumber



   def set_imageaxes(self, axnr1=None, axnr2=None, slicepos=None, promptfie=None):
   #--------------------------------------------------------------
      """
      A FITS file can contain a data set of dimension n.
      If n < 2 we cannot display the data without more information.
      If n == 2 the data axes are those in the FITS file, Their numbers are 1 and 2.
      If n > 2 then we have to know the numbers of those axes that
      are part of the image. For the other axes we need to know a
      pixel position so that we are able to extract a data slice.

      Attribute :attr:`dat` is then always a 2D array.

      :param axnr1:
         Axis number of first image axis (X-axis). If it is a string, then
         the number of the first axis which matches is returned. The string match
         is minimal and case insensitive. 
      :type axnr1:
         Integer or String
      :param axnr2:
         Axis number of second image axis (Y-axis). If it is a string, then
         the number of the first axis which matches is returned. The string match
         is minimal and case insensitive.
      :type axnr2:
         Integer or String
      :param slicepos:
         list with pixel positions on axes outside the image at which
         an image is extracted from the data set. Applies only to data sets with
         dimensions > 2. The length of the list must be equal to the number
         of axes in the data set that are not part of the image.
      :type slicepos:
         Integer or sequence of integers
      :param spectrans:
         The spectral translation to convert between different spectral types
         if one of the image axes has spectral type.
      :type spectrans:
         Integer
      :param promptfie:
         A Function that for in an Interactive Environment (fie),
         supplied by the user, that can
         prompt a user to enter the values for axnr1, axnr2
         and slicepos. An example of a function supplied by
         :mod:`maputils` is function :func:`prompt_imageaxes`

      :Raises:
         :exc:`Exception`
            *One axis number is missing and no prompt function is given!*

         :exc:`Exception`
            *Missing positions on axes outside image!* -- Somehow there are not
            enough elements in parameter *slicepos*. One should supply
            as many pixel positions as there are axes in the FITS data
            that do not belong to the selected image.

         :exc:`Exception`
            *Cannot find a matching axis for the spatial axis!* -- The matching
            spatial axis for one of the image axes could not be found
            in the FITS header. It will not be possible to get
            useful world coordinates for the spatial axis in your image.

      **Modifies attributes:**
          .. attribute:: axisinfo
          
                A dictionary with objects from class :class:`FITSaxis`. One object
                for each axis. The dictionary keys are the axis numbers.
                See also second example at method :meth:`FITSaxis.printattr`.
              
          .. attribute:: allowedtrans

                A list with strings representing the spectral translations
                that are possible for the current image axis selection.

          .. attribute:: spectrans

                The selected spectral translation

          .. attribute:: slicepos

                One or a list with integers that represent pixel positions on
                axes in the data set that do not belong to the image.
                At these position, a slice with image data is extracted.

          .. attribute:: map

                Image data from the selected FITS file. It is always a
                2D data slice and its size can be found in attribute
                :attr:`imshape`.

          .. attribute:: imshape

                The shape of the array :attr:`map`.

          .. attribute:: mixpix

                Images with only one spatial axis, need another spatial axis
                to produces useful world coordinates. This attribute is
                extracted from the relevant axis in attribute :attr:`slicepos`.

          .. attribute:: convproj

                An object from class Projection as defined in :mod:`wcs`.

          .. attribute:: axperm

                The axis numbers corresponding with the X-axis and Y-axis in
                the image.

      :Note:
         The aspect ratio is reset (to *None*) after each call to this method.

      :Examples:
         Set the image axes explicitly:

         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(1,2, slicepos=30)

         Set the images axes in interaction with the user using
         a prompt function:

         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)

         Enter (part of) the axis names. Note the minimal matching and
         case insensitivity.
                   
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes('ra','d', slicepos=30)
      """
      #-----------------------------------------------------------------
      axnr1 = self.getaxnumberbyname(axnr1)
      axnr2 = self.getaxnumberbyname(axnr2)
      n = self.naxis
      if n >= 2:
         if (axnr1 is None or axnr2 is None) and promptfie is None:
            if (axnr1 is None and axnr2 is None):
               axnr1 = self.axperm[0]
               axnr2 = self.axperm[1]
            else:
               raise Exception, "One axis number is missing and no prompt function is given!"
         if slicepos is None and promptfie is None:
            slicepos = self.slicepos

      # If there is a spectral axis in the FITS file, then get allowed
      # spectral translations
      self.allowedtrans = self.proj.altspec

      if promptfie != None:
         axnr1, axnr2, self.slicepos = promptfie(self, axnr1, axnr2)
      else:
         if slicepos != None and n > 2:
            if issequence(slicepos):
               self.slicepos = slicepos
            else:
               self.slicepos = [slicepos]
            k = 0
            for i in range(n):
               axnr = i + 1
               if axnr not in [axnr1, axnr2]:
                  self.sliceaxnames[k] = self.axisinfo[axnr].axname
                  k += 1
                  

      axperm = [axnr1, axnr2]
      wcsaxperm = [axnr1, axnr2]
      # User could have changed the order of which these axes appear 
      # in the FITS header. Sort them to assure the right order.
      axperm.sort()

      if n > 2:    # Get the axis numbers of the other axes
         if len(self.slicepos) != n-2:
            raise Exception, "Missing positions on axes outside image!"
         j = 0
         for i in range(n):
            axnr = i + 1
            if axnr != axnr1 and axnr != axnr2:
               axperm.append(axnr)
               self.axisinfo[axnr].outsidepix = self.slicepos[j]
               j += 1

      # Create a slice if the data has more dimensions than 2
      # Example: Enter pixel position on RA: 30
      # Slice =  [slice(None, None, None), slice(None, None, None), slice(30, 31, None)]
      # Fastest axis last
      if n > 2: 
         sl = []
         for ax in range(n,0,-1):     # e.g. ax = 3,2,1
            if (ax == axnr1 or ax == axnr2):
               sl.append(slice(None))
            else:
               g = self.axisinfo[ax].outsidepix
               sl.append(slice(g-1, g))             # Slice indices start with 0, pixels with 1

         # Reshape the array, assuming that the other axes have length 1
         # You can reshape with the shape attribute or with NumPy's squeeze method. With
         # squeeze there is no way to find images axis which have also length 1 so we use
         # the reshape() method.
         if self.dat != None:
            if axperm[0] != wcsaxperm[0]:
               self.boxdat = self.dat[sl].reshape((self.axisinfo[axnr1].axlen,self.axisinfo[axnr2].axlen))
            else:
               self.boxdat = self.dat[sl].reshape((self.axisinfo[axnr2].axlen,self.axisinfo[axnr1].axlen))
      else:
         self.boxdat = self.dat

      if self.boxdat != None:
         self.imshape = self.boxdat.shape
         if axperm[0] != wcsaxperm[0]:
            # The x-axis should be the y-axis vv.
            self.boxdat = numpy.swapaxes(self.boxdat, 0,1)   # Swap the x- and y-axes
            self.imshape = self.boxdat.shape

      if axperm[0] != wcsaxperm[0]:
         # The x-axis should be the y-axis vv.
         axperm[0] = wcsaxperm[0]     # Return the original axis permutation array
         axperm[1] = wcsaxperm[1]

      n1 = self.axisinfo[axperm[0]].axlen
      n2 = self.axisinfo[axperm[1]].axlen
      self.pxlim = [1, n1]
      self.pylim = [1, n2]

      #------------------------------------------------------------------
      # Here we do some analysis of the axes. We can think of many 
      # combinations of axes, but not all will be valid for WCSlib
      # WCSlib recognizes a longitude axis, a latitude axis, a spectral
      # axis and other types are identified with None. A spatial axis
      # ALWAYS needs a matching spatial axis. For the missing axis
      # we only know the pixel position and we use the mixed projection
      # method to find the corresponding world coordinate.
      # We distinguish the following situations:
      # 1) (Lon,Lat) or (Lat,Lon)  ==> Nothing to do
      # 2) (Lon,X) or (X,Lon) ==> Look in axis dict for Lat axis
      # 3) (Lat,X) or (X,Lat) ==> Look in axis dict for Lon axis
      # 4) (X,Spec) or (Spec,X) or (X,Y). ==> Nothing to do
      # In cases 1 and 4 return first two elements from axperm array
      # In cases 2 and 3 return first two elements from axperm array
      # and the third element is the axis number of the missing axis.
      #------------------------------------------------------------------
      self.mixpix = None
      ap = wcsaxperm                             # which elements could have been swapped
      mix = False
      matchingaxnum = None
      if n > 2:
         ax1 = wcsaxperm[0]; ax2 = wcsaxperm[1]
         if ax1 == self.proj.lonaxnum and ax2 != self.proj.lataxnum:
            matchingaxnum = self.proj.lataxnum
            mix = True
         elif ax1 == self.proj.lataxnum and ax2 != self.proj.lonaxnum:
            matchingaxnum = self.proj.lonaxnum
            mix = True
         if ax2 == self.proj.lonaxnum and ax1 != self.proj.lataxnum:
            matchingaxnum = self.proj.lataxnum
            mix = True
         elif ax2 == self.proj.lataxnum and ax1 != self.proj.lonaxnum:
            matchingaxnum = self.proj.lonaxnum
            mix = True
      if mix:
         if matchingaxnum != None:
            self.mixpix = self.axisinfo[matchingaxnum].outsidepix
            ap = (axperm[0], axperm[1], matchingaxnum)
         else:
            raise Exception, "Cannot find a matching axis for the spatial axis!"
      else:
          ap = (axperm[0], axperm[1])
      # If a spectral translation is needed, then we must apply the spectra()
      # method on the original projection object, which contains the spectral axis
      # The method does not work if the projection object is restricted to
      # two non spectral axes.
      if self.spectrans != None:
         self.proj.spectra(self.spectrans)
      self.convproj = self.proj.sub(ap)  # Projection object for selected image only
      #if self.spectrans != None:
      #   self.convproj = self.convproj.spectra(self.spectrans)
      if self.skyout != None:
         self.convproj.skyout = self.skyout
      self.axperm = wcsaxperm        # We need only the numbers of the first two axes
      self.aspectratio = None        # Reset the aspect ratio because we could have another image now
      


   def set_spectrans(self, spectrans=None, promptfie=None):
   #--------------------------------------------------------------
      """
      Set spectral translation or ask user to enter a spectral
      translation if one of the axes in the current FITSimage
      is spectral.

      :param spectrans:
         A spectral translation e.g. to convert frequencies
         to optical velocities.
      :type spectrans:
         String
      :param promptfie:
         A function, supplied by the user, that can
         prompt a user to enter a sky definition.
      :type promptfie:
         A Python function without parameters. It returns
         a string with the spectral translation.
         An example of a function supplied by
         :mod:`maputils` is function :func:`prompt_spectrans`
      
      :Examples:
         Set a spectral translation using 1) a prompt function,
         2) a spectral translation for which we don't know the code
         for the conversion algorithm and 3) set the translation explicitly:

         >>> fitsobject.set_spectrans(promptfie=maputils.prompt_spectrans)
         >>> fitsobject.set_spectrans(spectrans="VOPT-???")
         >>> fitsobject.set_spectrans(spectrans="VOPT-V2W")

      """
   #--------------------------------------------------------------
      isspectral = False
      for ax in range(1,self.naxis+1):
         if self.axisinfo[ax].wcstype == 'spectral':
            isspectral = True
      if not isspectral:
         return         # Silently
      if promptfie is None:
         #if spectrans is None:
         #   raise Exception, "No spectral translation given!"
         #else:
         # Allow for None
         self.spectrans = spectrans
      else:
         self.spectrans = promptfie(self)
      if self.spectrans != None:
         self.convproj = self.convproj.spectra(self.spectrans)
      
         
   def set_skyout(self, skyout=None, promptfie=None):
   #--------------------------------------------------------------
      """
      Set the output sky definition. Mouse positions and
      coordinate labels will correspond to the selected
      definition. The method will only work if both axes are
      spatial axes.

      :param skyout:
         The output sky definition for sky system, reference system,
         equinox and date of observation.
         For the syntax of a sky definition see the description
         at :meth:`celestial.skymatrix`
      :type skyout:
         A single value or tuple.
      :param promptfie:
         A function, supplied by the user, that can
         prompt a user to enter a sky definition.
      :type promptfie:
         A Python function without parameters. It returns
         the sky definition.
         An example of a function supplied by
         :mod:`maputils` is function :func:`prompt_skyout`
         
      :Notes:
          The method sets an output system only for data with
          two spatial axes. For XV maps the output sky system is
          always the same as the native system.

      """
   #--------------------------------------------------------------
      spatials = [self.proj.lonaxnum, self.proj.lataxnum]
      spatialmap = self.axperm[0] in spatials and self.axperm[1] in spatials

      if not spatialmap:
         return         # Silently
      
      if promptfie is None:
         if skyout is None:
            raise Exception, "No definition for the output sky is given!"
         else:
            self.skyout = skyout
      else:
         self.skyout = promptfie(self)

      if self.skyout != None:
         self.convproj.skyout = self.skyout


   def set_limits(self, pxlim=None, pylim=None, promptfie=None):
   #---------------------------------------------------------------------
      """
      This method sets the image box. That is, it sets the limits
      of the image axes in pixels. This can be a useful feature if
      one knows which part of an image contains the interesting data.
      
      :param pxlim:
         Two integer numbers which should not be smaller than 1 and not
         bigger than the header value *NAXISn*, where n represents the
         x axis.
      :type pxlim:
         Tuple with two integers
      :param pylim:
         Two integer numbers which should not be smaller than 1 and not
         bigger than the header value *NAXISn*, where n represents the
         y axis.
      :type pylim:
         Tuple with two integers
      :param promptfie:
         An external function with parameters *pxlim*, *pylim*, *axnameX*,
         and *axnameY* which are used to compose a prompt.
         If a function is given then there is no need to enter *pxlim* and *pylim*.
         The prompt function must return (new) values for *pxlim* and *pylim*.
         An example of a function supplied by
         :mod:`maputils` is function :func:`prompt_box`
      :type promptfie:
         Python function

      :Examples: Ask user to enter limits with prompt function :func:`prompt_box`
         
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(1,2, slicepos=30) # Define image in cube
         >>> fitsobject.set_limits(promptfie=maputils.prompt_box)
      """
   #---------------------------------------------------------------------
      n1 = self.axisinfo[self.axperm[0]].axlen
      n2 = self.axisinfo[self.axperm[1]].axlen
      npxlim = [None,None]
      npylim = [None,None]
      if pxlim is None:
         npxlim = [1, n1]
      else:
         if not issequence(pxlim):
            raise Exception, "pxlim must be tuple or list!"
         npxlim[0] = pxlim[0]
         npxlim[1] = pxlim[1]
      if pylim is None:
         npylim = [1, n2]
      else:
         if not issequence(pylim):
            raise Exception, "pylim must be tuple or list!"
         npylim[0] = pylim[0]
         npylim[1] = pylim[1]
      if promptfie != None:
         axname1 = self.axisinfo[self.axperm[0]].axname
         axname2 = self.axisinfo[self.axperm[1]].axname
         npxlim, npylim = promptfie(self.pxlim, self.pylim, axname1, axname2)
      # Check whether these values are within the array limits
      if npxlim[0] < 1:  npxlim[0] = 1
      if npxlim[1] > n1: npxlim[1] = n1
      if npylim[0] < 1:  npylim[0] = 1
      if npylim[1] > n2: npylim[1] = n2
      # Get the subset from the (already) 2-dim array
      if self.boxdat != None:
         self.boxdat = self.boxdat[npylim[0]-1:npylim[1], npxlim[0]-1:npxlim[1]]       # map is a subset of the original (squeezed into 2d) image
         self.imshape = self.boxdat.shape
      self.pxlim = npxlim
      self.pylim = npylim
      self.axisinfo[self.axperm[0]].axstart = npxlim[0]
      self.axisinfo[self.axperm[0]].axend = npxlim[1]
      self.axisinfo[self.axperm[1]].axstart = npylim[0]
      self.axisinfo[self.axperm[1]].axend = npylim[1]



   def get_pixelaspectratio(self):
   #---------------------------------------------------------------------
      """
      Return the aspect ratio of the pixels in the current
      data structure defined by the two selected axes.
      The aspect ratio is defined as *pixel height / pixel width*.

      :Example:

         >>> fitsobject = maputils.FITSimage('m101.fits')
         >>> print fitsobject.get_pixelaspectratio()
         1.0002571958

      :Note:

         If a header has only a cd matrix and no values for CDELT,
         then these values are set to 1. This gives an aspect ratio
         of 1.
      """
   #---------------------------------------------------------------------
      a1 = self.axperm[0]; a2 = self.axperm[1];
      cdeltx = self.axisinfo[a1].cdelt
      cdelty = self.axisinfo[a2].cdelt
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)

      # TODO do a better job calculating the aspectratio from a CD matrix.
      if self.proj.lonaxnum in self.axperm and self.proj.lataxnum in self.axperm:
         # Then is it a spatial map
         aspectratio = abs(cdelty/cdeltx)
      else:
         aspectratio = nx/ny
      self.pixelaspectratio = aspectratio
      return aspectratio



   def get_figsize(self, xsize=None, ysize=None, cm=False):
   #---------------------------------------------------------------------
      """
      Usually a user will set the figure size manually
      with Matplotlib's figure(figsize=...) construction.
      For many plots this is a waste of white space around the plot.
      This can be improved by taking the aspect ratio into account
      and adding some extra space for labels and titles.
      For aspect ratios far from 1.0 the number of pixels in x and y
      are taken into account.

      A handy feature is that you can enter the two values in centimeters
      if you set the flag *cm* to True.

      If you have a plot which is higher than its width and you want to
      fit in on a A4 page then use:

      >>> f = maputils.FITSimage(externalheader=header)
      >>> figsize = f.get_figsize(ysize=21, cm=True)
      >>> fig = plt.figure(figsize=figsize)
      >>> frame = fig.add_subplot(1,1,1)
      
      """
   #---------------------------------------------------------------------
      if xsize != None and not cm:
         xsize *= 2.54
      if ysize != None and not cm:
         ysize *= 2.54
      if xsize != None and ysize != None:
         return (xsize/2.54, ysize/2.54)

      a1 = self.axperm[0]; a2 = self.axperm[1];
      cdeltx = self.axisinfo[a1].cdelt
      cdelty = self.axisinfo[a2].cdelt
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      aspectratio = abs(cdelty/cdeltx)
      if aspectratio > 10.0 or aspectratio < 0.1:
         aspectratio = nx/ny
      extraspace = 3.0  # cm

      if xsize is None and ysize is None:
         if abs(nx*cdeltx) >= abs(ny*cdelty):
            xsize = 21.0        # A4 width
         else:
            ysize = 21.0
      if xsize != None:                       # abs(nx*cdeltx) >= abs(ny*cdelty):
         xcm = xsize
         # The extra space is to accommodate labels and titles
         ycm = xcm * (ny/nx) * aspectratio + extraspace
      else:
         ycm = ysize
         xcm = ycm * (nx/ny) / aspectratio + extraspace
      return (xcm/2.54, ycm/2.54)


   def header2classic(self):
      #--------------------------------------------------------------------
      """
      If a header contains PC or CD elements, and not all the 'classic'
      elements for a WCS then  a number of FITS readers could have a
      problem if they don't recognize a PC and CD matrix. What can be done
      is to derive the missing header items, CDELTn and CROTA from
      these headers and add them to the header.

      What is a 'classic' FITS header?
      
      (See also http://fits.gsfc.nasa.gov/fits_standard.html)
      For the transformation between pixel coordinates and world
      coordinates, FITS supports three conventions.
      First some definitions:

      An intermediate pixel coordinate :math:`q_i`  is calculated from a
      pixel coordinates :math:`p` with:

      .. math::
      
         q_i = \sum_{j=1}^N m_{ij}(p_j-r_j)

      Rj are the pixel coordinate elements of a reference point
      (FITS header item CRPIXj), j is an index for the pixel axis
      and i for the world axis
      The matrix :math:`m_{ij}` must be non-singular and its dimension is
      NxN where N is the number of world coordinate axes (given
      by FITS header item NAXIS).

      The conversion of :math:`q_i` to intermediate world coordinate :math:`x_i` is
      a scale :math:`s_i`:

      .. math::
      
         x_i = s_i q_i

      **Formalism 1 (PC keywords)**

      Formalism 1 encodes :math:`m_{ij}` in so called PCi_j keywords
      and scale factor :math:`s_i` are the values of the CDELTi keywords
      from the FITS header.

      It is obvious that the value of CDELT should not be 0.0.

      **Formalism 2 (CD keywords)**

      If the matrix and scaling are combined we get for the
      intermediate WORLD COORDINATE :math:`x_i`:

      .. math::

         x_i = \sum_{j=1}^N (s_i m_{ij})(p_j-r_j)

      FITS keywords CDi_j encodes the product :math:`s_i m_{ij}`.
      The units of :math:`x_i` are given by FITS keyword CTYPEi.

      **Formalism 3 (Classic)**
      
      This is the oldest but now deprecated formalism. It uses CDELTi
      for the scaling and CROTAn for a rotation of the image plane.
      n is associated with the latitude axis so often one sees
      CROTA2 in the header if the latitude axis is the second axis
      in the dataset

      Following the FITS standard, a number of rules is set:
      
         1. CDELT and CROTA may co-exist with the CDi_j keywords
            but must be ignored if an application supports the CD
            formalism.
         2. CROTAn must not occur with PCi_j keywords
         3. CRPIXj defaults to 0.0
         4. CDELT defaults to 1.0
         5. CROTA defaults to 0.0
         6. PCi_j defaults to 1 if i==j and to 0 otherwise. The matrix
            must not be singular
         7. CDi_j defaults to 0.0. The matrix must not be singular.
         8. CDi_j and PCi_j must not appear together in a header.


      **Alternate WCS axis descriptions**

      A World Coordinate System (WCS) can be described by an
      alternative set of keywords.
      For this keywords a character in the range [A..Z] is appended.
      In our software we made the assumption that the primary
      description contains all the necessary information
      to derive a 'classic' header and therefore we will ignore
      alternate header descriptions.


      **Conversion to a formalism 3 ('classic') header**

      Many FITS readers from the past are not upgraded to process
      FITS files with headers written using formalism 1 or 2.
      The purpose of this application is to convert a FITS file
      to a file that can be read and interpreted by old FITS readers.
      For GIPSY we require FITS headers to be written using formalism
      3. If keywords are missing, they will be derived and a
      comment will be inserted about the keyword not being an
      original keyword.

      The method that converts the header, tries to process it
      first with WCSLIB (tools to interpret the world coordinate
      system as described in a FITS header). If this fails, then we
      are sure that the header is incorrect and we cannot proceed.
      One should use a FITS tool like 'fv' (the Interactive FITS
      File Editor from Nasa) to repair the header.

      The conversion process starts with exploring the spatial part
      of the header.
      It knows which axes are spatial and it reads the corresponding
      keywords (CDELTi, CROTAn, CDi_j, PCi_j and PC00i_00j (old
      format for PC elements).
      If there is no CD or PC matrix, then the conversion routine
      returns the unaltered original header.
      If it finds a PC matrix and no CD matrix then the header should
      contain CDELT keywords. With the values of these keywords we
      create a CD matrix:

      .. math::
      
         \\begin{bmatrix}cd_{11} & cd_{12}\\\ cd_{21} & cd_{22}\\end{bmatrix} =
         \\begin{bmatrix}cdelt_1 & 0\\\ 0 & cdelt_2 \\end{bmatrix}
         \\begin{bmatrix}pc_{11} & pc_{12}\\\ pc_{21} & pc_{22}\\end{bmatrix}


      Notes:
      
         *  We replaced notation i_j by ij so cd11 == CD1_1
         *  For the moment we restricted the problem to the 2 dim.
            spatial case because that is what we need to retrieve
            a value for CROTA, the rotation of the image.)
         *  We assumed that the PC matrix did not represent
            transposed axes as in:

      .. math::

         PC = \\begin{bmatrix}0 & 1 & 0\\\ 0 & 0 & 1\\\ 1 & 0 & 0 \\end{bmatrix}
         

      If cd12 == 0.0 and cd12 == 0.0 then CROTA is obviously 0.
      There is no rotation and CDELT1 = cd11, CDELT2 = cd22

      If one or both values of cd12, cd21 is not zero then we
      expect a value for CROTA unequal to zero and/or skew.

      We calculate the scaling parameters CDELT with::

                     CDELT1 = sqrt(cd11*cd11+cd21*cd21)
                     CDELT2 = sqrt(cd12*cd12+cd22*cd22)

      The determinant of the matrix is::

                     det = cd11*cd22 - cd12*cd21

      This value cannot be 0 because we required that the matrix is
      non-singular.
      Further we distinguish two situations: a determinant < 0 and
      a determinant > 0 (zero was already excluded).
      Then we derive two rotations. If these are equal,
      the image is not skewed.
      If they are not equal, we derive the rotation from the
      average of the two calculated angles. As a measure of skew,
      we calculated the difference between the two rotation angles.
      Here is a piece of the actual code::

                     sign = 1.0
                     if det < 0.0:
                        cdeltlon_cd = -cdeltlon_cd
                        sign = -1.0
                     rot1_cd = atan2(-cd21, sign*cd11)
                     rot2_cd = atan2(sign*cd12, cd22)
                     rot_av = (rot1_cd+rot2_cd)/2.0
                     crota_cd = degrees(rot_av)
                     skew = degrees(abs(rot1_cd-rot2_cd))

      New values of CDELT and CROTA will be inserted in the new
      'classic' header only if they were not part of the original
      header.

      The process continues with non spatial axes. For these axes
      we cannot derive a rotation. We only need to find a CDELTi
      if for axis i no value could be found in the header.
      If this value cannot be derived from the a CD matrix (usually
      with diagonal element CDi_i) then the default 1.0 is assumed.
      Note that there will be a warning about this in the comment
      string for the corresponding keyword in the new 'classic' FITS
      header.

      Finally there is some cleaning up. First all CD/PC elements are
      removed for all the axes in the data set. Second, some unwanted
      keywords are removed. The current list is:
      ["XTENSION", "EXTNAME", "EXTEND"]

      
      See also: Calabretta & Greisen: 'Representations of celestial coordinates
      in FITS', section 6

      :Returns:
         A tuple with three elements:

         * *hdr* - A modified copy of the current header.
           The CD and PC elements are removed.
         * *skew* - Difference between the two calculated rotation angles
           If this number is bigger then say 0.001 then there is considerable
           skew in the data. One should reproject the data so that it fits
           a non skewed version with only a CROTA in the header
         * *hdrchanged* - A list with keywords the are changed when a
           'classic header is required.

      :Example:

         ::
         
            from kapteyn import maputils, wcs
            import pyfits
            
            
            Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
            newheader, skew, hdrchanged = Basefits.header2classic()
            if len(hdrchanged):
               print newheader
            if skew != 0.0:
               print "found skew:", skew

      :Notes:

         This method is tested with FITS files:

         * With classic header
         * With only CD matrix
         * With only PC matrix
         * With PC and CD matrix
         * With CD matrix and NAXIS > 2
         * With sample files with skew

         
      """
      #--------------------------------------------------------------------
      skew = 0.0
      hdrchanged = []
      hdr = self.hdr.copy()
      if type(hdr) == 'dict':
         dicttype = True
      else:
         dicttype = False
      comment = "Appended by Kapteyn Package module Maputils %s" % datetime.now().strftime("%dd%mm%Yy%Hh%Mm%Ss")
      lonaxnum = self.proj.lonaxnum
      lataxnum = self.proj.lataxnum
      spatial = (lonaxnum != None and lataxnum != None)
      if spatial:
         cdeltlon = None
         cdeltlat = None
         crota = None
         key = "CDELT%d" % lonaxnum
         if hdr.has_key(key):
            cdeltlon = hdr[key]
         key = "CDELT%d" % lataxnum
         if hdr.has_key(key):
            cdeltlat = hdr[key]
         key = "CROTA%d" % lataxnum
         if hdr.has_key(key):
            crota = hdr[key]
         cd11 = cd12 = cd21 = cd22 = None
         CD = [0.0]*4   # Unspecified elements default to 0.0 G+C paper I 
         k = 0
         cdmatrix = False
         for i in [lonaxnum, lataxnum]:
            for j in [lonaxnum, lataxnum]:
               key = "CD%d_%d" % (i, j)
               if hdr.has_key(key):
                  CD[k] = hdr[key]
                  del hdr[key]
                  hdrchanged.append(key)
                  cdmatrix = True
               k += 1 
         cd11, cd12, cd21, cd22 = CD

         # Clean up CROTA's, often there are more than one in the same header
         key = "CROTA%d" % (lonaxnum)
         if hdr.has_key(key):
            del hdr[key]

         pcmatrix = False
         PC = [None]*4
         k = 0
         for i in [lonaxnum, lataxnum]:
            for j in [lonaxnum, lataxnum]:
               # Set the defaults (i.e. 1 if i==j else 0)
               if i == j:
                  PC[k] = 1.0
               else:
                  PC[k] = 0.0
               key = "PC%d_%d" % (i, j)
               if hdr.has_key(key):
                  PC[k] = hdr[key]
                  del hdr[key]
                  hdrchanged.append(key)
                  pcmatrix = True
               k += 1
         pc11, pc12, pc21, pc22 = PC

         pcoldmatrix = False
         # If no PC found then try legacy numbering
         if not pcmatrix:
            PCold = [None]*4
            k = 0
            for i in [lonaxnum, lataxnum]:
               for j in [lonaxnum, lataxnum]:
                  if i == j:
                     PCold[k] = 1.0
                  else:
                     PCold[k] = 0.0
                  key = "PC%03d%03d" % (i, j)    # Like 001, 002
                  if hdr.has_key(key):
                     PCold[k] = hdr[key]
                     del hdr[key]
                     hdrchanged.append(key)
                     pcoldmatrix = True
                  k += 1
            pco11, pco12, pco21, pco22 = PCold

         if not pcmatrix and pcoldmatrix:
            PC = PCold
            pc11, pc12, pc21, pc22 = PC
            pcmatrix = True

         # If the CD is empty but the PC exists,
         # then use the PC to create a CD.

         if not cdmatrix:
            if pcmatrix and None in [cdeltlon, cdeltlat]:
               # A PC matrix cannot exist without values for CDELT
               raise ValueError("Header does not contain necessary CDELT's!")
            if pcmatrix:
               # Found no CD but use PC to create one
               # |cd11 cd12|  = |cdelt1      0| * |pc11 pc12|
               # |cd21 cd22|    |0      cdelt2|   |pc21 pc22|
               cd11 = cdeltlon*pc11
               cd12 = cdeltlon*pc12
               cd21 = cdeltlat*pc21
               cd22 = cdeltlat*pc22
               CD = [cd11, cd12, cd21, cd22]
               cdmatrix = True

         # PC and CD should not appear both in the same header
         # but if it does, use the CD.
         # If there is no CD/PC matrix then there is nothing to do
         if not cdmatrix:
            return hdr, 0, []
         else:
            from math import sqrt
            from math import atan2
            from math import degrees, radians
            from math import cos, sin, acos
            if cd12 == 0.0 and cd21 == 0.0:
               crota_cd = 0.0
               cdeltlon_cd = cd11
               cdeltlat_cd = cd22
            else:
               cdeltlon_cd = sqrt(cd11*cd11+cd21*cd21)
               cdeltlat_cd = sqrt(cd12*cd12+cd22*cd22)
               det = cd11*cd22 - cd12*cd21
               if det == 0.0:
                  raise ValueError("Determinant of CD matrix == 0")
               sign = 1.0
               if det < 0.0:
                  cdeltlon_cd = -cdeltlon_cd
                  sign = -1.0
               rot1_cd = atan2(-cd21, sign*cd11)
               rot2_cd = atan2(sign*cd12, cd22)
               rot_av = (rot1_cd+rot2_cd)/2.0
               crota_cd = degrees(rot_av)
               skew = degrees(abs(rot1_cd-rot2_cd))
               """
               print "Angles from cd matrix:", degrees(rot1_cd), degrees(rot2_cd), crota_cd
               print "Cdelt's from cd matrix:", cdeltlon_cd, cdeltlat_cd
               print "Difference in angles (deg)", skew
               """

            # At this stage we have values for the CDELTs and CROTA derived
            # from the CD/PC matrix. If the corresponding header items are
            # not available, then put them in the header and flag the header as
            # altered.

            if cdeltlon is None:
               cdeltlon = cdeltlon_cd
               key = "CDELT%d" % lonaxnum
               # Create new one if necessary
               if dicttype:
                  hdr.update(key=cdeltlon)
               else:
                  hdr.update(key, cdeltlon, comment)
               hdrchanged.append(key)
            if cdeltlat is None:
               cdeltlat = cdeltlat_cd
               key = "CDELT%d" % lataxnum
               if dicttype:
                  hdr.update(key=cdeltlat)
               else:
                  hdr.update(key, cdeltlat, comment)
               hdrchanged.append(key)
            if crota is None:
               crota = crota_cd
               key = "CROTA%d" % lataxnum
               if dicttype:
                  hdr.update(key=crota)
               else:
                  hdr.update(key, crota, comment)
               hdrchanged.append(key)

      # What if the data was not spatial and what do we do with
      # the other non spatial axes in the header?
      # Start looping over other axes:
      wasdefault = False
      naxis = hdr['NAXIS']
      if (spatial and naxis > 2) or not spatial:
         for k in range(naxis):
            axnum = k + 1
            key = "CDELT%d" % axnum
            if not spatial or (spatial and axnum != lonaxnum and axnum != lataxnum):
               if not hdr.has_key(key):
                  key_cd = "CD%d_%d"%(axnum, axnum)  # Diagonal elements only!
                  if hdr.has_key(key_cd):
                     newval = hdr[key_cd] 
                  else:
                     # We have a problem. For this axis there is no CDELT
                     # nor a CD matrix element. Set CDELT to 1.0 (FITS default)
                     newval = 1.0
                     wasdefault = True
                  if dicttype:
                     hdr[key] = newval
                  else:
                     if wasdefault:
                        s = "Inserted default 1.0! " + comment
                     else:
                        s = comment
                     hdr.update(key, newval, s)
                  hdrchanged.append(key)
      # Clean up left overs
      for i in range(naxis):
         for j in range(naxis):
             axi = i + 1; axj = j + 1
             keys = ["CD%d_%d"%(axi, axj), "PC%d_%d"%(axi, axj), "PC%03d%03d"%(axi, axj)]
             for key in keys:
                if hdr.has_key(key):
                   del hdr[key]
                   hdrchanged.append(key)

      # Exceptions etc.:
      ekeys = ["XTENSION", "EXTNAME", "EXTEND"]
      # Possibly EXTEND will be re-inserted when writing the FITS file
      for key in ekeys:
         if hdr.has_key(key):
            del hdr[key]

      return hdr, skew, hdrchanged


   def reproject_to(self, reprojobj=None, pxlim_dst=None, pylim_dst=None,
                    plimlo=None, plimhi=None, interpol_dict = None,
                    rotation=None, insertspatial=None, **fitskeys):
      #---------------------------------------------------------------------
      """
      The current FITSimage object must contain a number of spatial maps.
      This method then reprojects these maps so that they conform to
      the input header.

      Imagine an image and a second image of which you want to overlay contours
      on the first one. Then this method uses the current data to reproject
      to the input header and you will end up with a new FITSimage object
      which has the spatial properties of the input header and the reprojected
      data of the current FITSimage object.

      Also more complicated data structures can be used. Assume you have a
      data cube with axes RA, Dec and Freq. Then this method will reproject
      all its spatial subsets to the spatial properties of the input header.

      The current FITSimage object tries to keep as much of its original
      FITS keywords. Only those related to spatial data are copied from the
      input header. The size of the spatial map can be limited or extended.
      The axes that are not spatial are unaltered.

      The spatial information for both data structures are extracted from
      the headers so there is no need to specify the spatial parts of the
      data structures.

      The image that is reprojected can be limited in size with parameters
      pxlim_dst and pylim_dst.
      If the input is a FITSimage object, then these parameters (pxlim_dst
      and pylim_dst) are copied
      from the axes lengths set with method :meth:`set_limits()` for that
      FITSimage object.

      :param reprojobj:
          *  The header which provides the new information to reproject to.
             The size of the reprojected map is either copied from
             the NAXIS keywords in the header or entered with parameters
             *pxlim_dst* and *pylim_dst*. The reprojections are done for all
             spatial maps in the current FITSimage object or for a selection
             entered with parameters *plimlo* and *plimhi* (see examples).
          *  The FITSimage object from which relevant information is
             extracted like the header and the new sizes of the spatial axes
             which otherwise should have been provided in parameters
             *pxlim_dst* and *pylim_dst*. The reprojection is restricted to
             one spatial map and its slice information is copied
             from the current FITSimage. This option is selected if you
             want to overlay e.g. contours from the current FITSimage
             data onto data from another WCS.
          *  If None, then the current header is used. Modifications to this header
             are done with keyword arguments.
      :type reprojobj:
          Python dictionary or PyFITS header. Or a :class:`maputils.FITSimage`
          object
      :param pxlim_dst:
          Limits in pixels for the reprojected box.
      :type pxlim_dst:
          Tuple of integers
      :param plimlo:
          One or more pixel coordinates corresponding to axes outside
          the spatial map in order as found in the header 'reprojobj'.
          The values set the lower limits of the axes.
          There is no need to specify all limits but the order is important.
      :type plimlo:
          Integer or tuple of integers
      :param plimhi:
          The same as plimhi, but now for the upper limits.
      :type plimhi:
          Integer or tuple of integers
      :param interpol_dict:
          This parameter is a dictionary with parameters for the interpolation routine
          which is used to reproject data. The interpolation routine
          is based on SciPy's *map_coordinates*. The most important parameters with
          the maputils defaults are:

          .. tabularcolumns:: |p{10mm}|p{100mm}|

          =========  ===============================================================        
            order :  Integer, optional

                     The order of the spline interpolation, default is 1.
                     The order has to be in the range 0-5.
            mode :   String, optional

                     Points outside the boundaries of the input are filled according
                     to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
                     Default is 'constant'.
            cval :   scalar, optional

                     Value used for points outside the boundaries of the input if
                     mode='constant'. Default is numpy.NaN
          =========  ===============================================================
          
      :type interpol_dict:
         Python dictionary
      :param rotation:
         Sets a rotation angle. If this method encounters this keyword, it will
         create a so called 'classic' header. That is a header without
         CD or PC elements. Then the rotation angle of the current spatial
         map is only given by FITS keyword *CROTAn*. The value of *rotation* is added
         to *CROTAn* to create a new value which is inserted in the new header.
         Note that values for *CROTAn* in the *fitskeys* parameter list
         overwrite this calculated value.
      :type rotation:
         Floating point number or *None*
      :param insertspatial:
         If True, then replace spatial part of current header by spatial part
         of new (destination) header.
         Assume you start with a data cube with a number of spatial maps
         as function of frequency (the third axis in the data set). If
         you use the header of another FITS file as the definition of
         a new world coordinate system, then it could be that this header
         represents a two dimensional data set. This is not an incompatible
         set because we only require a description of the spatial part.
         To keep the description of the original three dimensional
         structure we insert the new spatial(only) information into the
         current header. The default then is *insertspatial=True*.
         In other cases where we use the original header as the base
         header, one just inserts new values and there is no need to
         set insert something, so then the default is *insertspatial=False*.
         A user can change the default. This can be useful. For example
         in the maputils tutorial we have a script mu_reproj2classic.py where
         we use the header of a FITS file to make some changes and use the
         changed header as an external header to re-project to. The default
         then is *insertspatial=True*, but the external header is already
         a complete header, so there is no need to insert something.
      :type insertspatial:
         Boolean
      :param fitskeys:
         Parameters containing FITS keywords and values which are written
         in the reprojection header. 
      :type fitskeys:
         Python keyword arguments.


      .. warning::

            Values for *CROTAn* in parameter *fitskeys* overwrite
            values previously set with keyword *rotation*.

      .. warning::

            Changing values of *CROTAn* will not always result in
            a rotated image. If the world coordinate system was defined using
            CD or PC elements, then changing *CROTAn* will only add the keyword
            but it is never read because CD & PC transformations have precedence.


      :Examples:

            -Set limits for axes outside the spatial map. Assume a data structure
            with axes RA-DEC-FREQ-STOKES for which the RA-DEC part is reprojected to
            a set RA'-DEC'-FREQ-STOKES. The ranges for FREQ and STOKES set the
            number of spatial maps in this data structure. One can limit these
            ranges with *plimlo* and *plimhi*.
            
            * *plimlo=(20,2)*, *plimhi=(40,2)*

              we restrict the reprojections for spatial maps
              at frequencies 20 to 40 at one position on the STOKES axis
              (at pixel coordinate 2).
   
            * *plimlo=(None,2)*, *plimhi=(None,2)*
            
              If one wants to reproject all the maps at all frequencies
              but only for STOKES=2 and 3 then use:
              *plimlo=(None,2)* and *plimhi=(None,2)* where None implies no limits.
   
            * *plimlo=40*
              
              No *plimhi* is entered. Then there are no upper limits. Only one value
              (40) is entered so this must represent the FREQ axis at pixel
              coordinate 40. It represents all spatial maps from FREQ pixel
              coordinate 40 to the end of the FREQ range, repeated for all
              pixels on the STOKES axis.
   
            * *plimlo=(55,1)*, *plimhi=(55,1)*
            
              This reprojects just one map at FREQ pixel coordinate 55
              and STOKES pixel coordinate 1. This enables a user/programmer
              to extract one spatial map, reproject it and write it as a single
              map to a FITS file while no information about the FREQ and STOKES
              axes is lost. The dimensionality of the new data remains 4 but
              the length of the 'repeat axes' is 1.
   
            Note that if the data structure was represented by axes
            FREQ-RA-STOKES-DEC then the examples above are still valid because
            these set the limits on the repeat axes FREQ and POL whatever the
            position of these axes in the data structure.


            -Use and modify the current header to change the data.
            The example shows how to rotate an image and display the result.

            ::

               Basefits = maputils.FITSimage("m101.fits")
               Rotfits = Basefits.reproject_to(rotation=40.0,
                                             naxis1=800, naxis2=800,
                                             crpix1=400, crpix2=400)
   
               # If copy on disk required:
               # Rotfits.writetofits("m10rot.fits", clobber=True, append=False)
   
               annim = Rotfits.Annotatedimage()
               annim.Image()
               annim.Graticule()
               annim.interact_toolbarinfo()
               maputils.showall()

            -Use an external header and change keywords in that header
            before the re-projection:
       
            >>> Rotfits = Basefits.reproject_to(externalheader,
                                                naxis1=800, naxis2=800,
                                                crpix1=400, crpix2=400)


            -Use the FITS files own header. Change it and use it as an
            external header

            ::

               from kapteyn import maputils, wcs

               Basefits = maputils.FITSimage("m101.fits")
               classicheader, skew, hdrchanged = Basefits.header2classic()

               lat = Basefits.proj.lataxnum
               key = "CROTA%d"%lat
               classicheader[key] = 0.0 # New value for CROTA
               fnew = Basefits.reproject_to(classicheader, insertspatial=False)
               fnew = maputils.FITSimage(externalheader=classicheader,
                                         externaldata=Basefits.dat)
               fnew.writetofits("classic.fits", clobber=True, append=False)

            
      .. note::

            If you want to align an image with the direction of the north,
            then the value of *CROTAn* (e.g. CROTA2) should be set to zero.
            To ensure that the data will be rotated, use parameter
            *rotation* with a dummy value so that the header used for
            the re-projection is a 'classic' header:

            e.g.:

            >>> Rotfits = Basefits.reproject_to(rotation=0.0, crota2=0.0)


            Todo: If CTYPE's change, then also LONPOLE and LATPOLE
                  should change
         

      :Tests:

         1) The first test was a reprojection of data of *map1* to the
            spatial header of *map2*. One should observe that the result
            of the reprojection (*reproj*) has its spatial structure from
            *map2* and its non spatial structure (i.e. the repeat axes)
            from *map1*. Note that the order of longitude, latitude
            in *map1* is swapped in *map2*.
   
            map1:
            CTYPE:  RA - POL - FREQ - DEC
            NAXIS   35   5     16     41
   
            map2:
            CTYPE:  DEC - POL - FREQ - RA
            NAXIS   36    4     17     30
   
            reproj = map1.reproject_to(map2)
   
            reproj:
            CTYPE:  RA - POL - FREQ - DEC
            NAXIS   36   5     16     30

         2) Tested with values for the repeat axes
         3) Tested with values for the output box
         4) Tested with a new CTYPE (GLON-TAN, GLAT-TAN)
            and new CRVAL
      """
      #---------------------------------------------------------------------

      # Create a new header based on the current one.
      # Get the relevant keywords for the longitude axis
      # in the input header. Copy them in the new header with
      # the right number representing longitude in the copied header.
      # Create a data structure with as many spatial maps as there are
      # in the current data structure.
      # Correct the CRPIX and NAXIS values to support the new limits
      # Create a new FITSimage object with this dummy data and new header
      # Loop over all spatial maps and reproject them.
      # Store the reprojected data in the new data array at the appropriate
      # location
      # Return the result. 

      # What are the axis numbers of the spatial axis in order of their axis numbers?
      def flatten(seq):
         res = []
         for item in seq:
            if (isinstance(item, (tuple, list))):
               res.extend(flatten(item))
            else:
               res.append(item)
         return res

      plimLO = plimHI = None
      fromheader = True         # We need to distinguish header and FITSimage objects
      if isinstance(reprojobj, FITSimage):
         # It's a FITSimage object
         # Copy its attributes that are relevant in this context
         pxlim_dst = reprojobj.pxlim
         pylim_dst = reprojobj.pylim
         slicepos = self.slicepos
         plimLO = plimHI = slicepos
         repheader = reprojobj.hdr.copy()
         fromheader = False
         if insertspatial is None:
            insertspatial = True
      else:
         # It's a plain header or None
         if reprojobj is None:   # Then destination is original header. Modify later.
            repheader = self.hdr.copy()
            if insertspatial is None:
               insertspatial = False
         else:
            # A Python dict or a PyFITS header
            repheader = reprojobj.copy()
            if insertspatial is None:
               insertspatial = True

      # For a rotation (only) we convert the header to a 'classic' header
      if reprojobj is None and not rotation is None:
         # Get rid of skew and ambiguous rotation angles
         # by converting cd/pc matrix to 'classic' header, so that CROTA can
         # be adjusted.
         repheader, skew, hdrchanged = self.header2classic()
         # Look for the right CROTA (associated with latitude)
         key = "CROTA%d"%self.proj.lataxnum
         if repheader.has_key(key):
            crotanew = repheader[key] + rotation     # Rotation of latitude axis + user rot.
            repheader[key] = crotanew
         else:
            crotanew = rotation   # No CROTA, then assume CROTA=0.0
            if type(repheader) == 'dict':
               repheader[key] = crotanew
            else:
               repheader.update(key, crotanew)
           
      if len(fitskeys) > 0:
         # Note that a ROTATION= keyword changes the value of CROTAn
         # but we can overwrite this with a user supplied keyword CROTAn
         # In effect, this is a trigger that CROTA changes and a 'classic'
         # header is required.
         change_header(repheader, **fitskeys)

      p1 = wcs.Projection(self.hdr)
      
      naxis = len(p1.naxis)
      axnum = []
      axnum_out = []
      for i in range(1, naxis+1):
         if i in [p1.lonaxnum, p1.lataxnum]:
            axnum.append(i)
         else:
            axnum_out.append(i)
      if len(axnum) != 2:
         raise Exception, "No spatial maps to reproject in this data structure!"
      naxisout = len(axnum_out)
      len1 = p1.naxis[axnum[0]-1]; len2 =  p1.naxis[axnum[1]-1]
      # Now we are sure that we have a spatial map and the
      # axis numbers of its axes, in the order of the data structure
      # i.e. if lon, lat are swapped in the header then the
      # first axis in the data is lat.

      # Create a projection object for
      # the current spatial maps
      p1_spat = p1.sub(axnum)
      
      # Get a Projection object for a spatial map defined in the
      # input header, the destination
      p2 = wcs.Projection(repheader)

      naxis_2 = len(p2.naxis)
      axnum2 = []
      for i in range( 1, naxis_2+1):
         if i in [p2.lonaxnum, p2.lataxnum]:
            axnum2.append(i)

      if len(axnum2) != 2:
         raise Exception, "The input header does not contain a spatial data structure!"
      p2_spat = p2.sub(axnum2)

      # Determine the size and shape for a new data array. The new
      # shape is the shape of the current FITSimage object with
      # the spatial axes length replaced by the values from the input header.
      lenXnew = p2.naxis[axnum2[0]-1]; lenYnew =  p2.naxis[axnum2[1]-1]
      lonaxnum2 = p2.lonaxnum; lataxnum2 = p2.lataxnum
      shapenew = [0]*naxis
      # Uitbreiden met meer flexibiliteit
      if pxlim_dst is None:
         pxlim_dst = [1]*2
         pxlim_dst[1] = lenXnew
      if pylim_dst is None:
         pylim_dst = [1]*2
         pylim_dst[1] = lenYnew
      nx = pxlim_dst[1] - pxlim_dst[0] + 1
      ny = pylim_dst[1] - pylim_dst[0] + 1
      N = nx * ny

      # Next we process the ranges on the repeat axes (i.e. axes
      # outside the spatial map). See the documentation for what
      # is allowed to enter
      if naxisout > 0:
         if plimLO is None or plimHI is None:
            plimLO = [0]*(naxis-2)
            plimHI = [0]*(naxis-2)
            for i, axnr in enumerate(axnum_out):
               plimLO[i] = 1
               plimHI[i] = p1.naxis[axnr-1]
            # Make sure user given limits are list or tuple
            if plimlo != None:
               if not issequence(plimlo):
                  plimlo = [plimlo]
               if len(plimlo) > naxisout:
                  raise ValueError("To many values in plimlo, max=%d"%naxisout)
               for i, p in enumerate(plimlo):
                  plimLO[i] = p
            if plimhi != None:
               if not issequence(plimhi):
                  plimhi = [plimhi]
               if len(plimhi) > naxisout:
                  raise ValueError("To many values in plimhi, max=%d"%naxisout)
               for i, p in enumerate(plimhi):
                  plimHI[i] = p

         # Use the sizes of the original/current non spatial axes
         # to calculate size and shape of the new data array
         for axnr, lo , hi  in zip(axnum_out, plimLO, plimHI):
            n = hi - lo + 1
            shapenew[axnr-1] = n
            N *= n

      shapenew[axnum[0]-1] = nx
      shapenew[axnum[1]-1] = ny
      shapenew = shapenew[::-1]  # Invert list with lengths so that it represents a shape
      newdata = numpy.zeros(N, dtype=self.dat.dtype)  # Data is always float
      newdata.shape = shapenew

      # The following code inserts all keywords related to the spatial information
      # of the destination header (to which you want to reproject) into a new header
      # which is a copy of the current FITSimage object. The spatial axes in the
      # current header are replaced by the spatial axes in the input header
      # both in order of their axis number.
      if insertspatial:
         newheader = self.insertspatialfrom(repheader, axnum, axnum2)   #lonaxnum2, lataxnum2)
      else:
         newheader = repheader

      # In the new header the length of spatial and the repeat axes could have been
      # changed. Adjust the relevant keywords in the new header
      if nx != lenXnew:
         ax = axnum2[0]
         newheader['NAXIS%d'%ax] = nx
         newheader['CRPIX%d'%ax] += -pxlim_dst[0] + 1
      if ny != lenYnew:
         ax = axnum2[1]
         newheader['NAXIS%d'%ax] = ny
         newheader['CRPIX%d'%ax] += -pylim_dst[0] + 1
      if naxisout > 0:
         for axnr, lo, hi in zip(axnum_out, plimLO, plimHI):
            n = hi - lo + 1
            newheader['NAXIS%d'%axnr] = n
            newheader['CRPIX%d'%axnr] += -lo + 1
         # Also add CD elements if they are missing in the current
         # structure but available in the input header.
         key1 = "CD%d_%d" % (p1.lonaxnum, p1.lonaxnum)
         key2 = "CD%d_%d" % (axnum_out[0], axnum_out[0])
         if newheader.has_key(key1) and not newheader.has_key(key2):
            # Obviously there is a CD matrix in the input header
            # for the spatial axes, but these elements are not available
            # for the other axes in the current header
            for axnr in axnum_out:
               key2 = "CD%d_%d" % (axnr, axnr)
               # NO CD element then CDELT must exist
               newheader[key2] = newheader["CDELT%d"%axnr]

      # Process the dictionary for the interpolation options
      if interpol_dict != None:
         if not interpol_dict.has_key('order'):
            interpol_dict['order'] = 1
         if not interpol_dict.has_key('cval'):
            interpol_dict['cval'] = numpy.NaN
      else:
         interpol_dict = {}
         interpol_dict['order'] = 1
         interpol_dict['cval'] = numpy.NaN

      # Create the coordinate map needed for the interpolation.
      # Use the limits given in pxlim_dst, pylim_dst to set shape and offset.
      # Note that pxlim_dst is 1 based and the offsets are 0 based.
      dst_offset = (pylim_dst[0]-1, pxlim_dst[0]-1)   # Note the order!
      # In a previous version we had an offset in the source:
      # src_offset = (self.pylim[0]-1, self.pxlim[0]-1)
      # but this is not necessary if we keep 'boxdat' as big as
      # the entire spatial map. Besides that, it is not sure that
      # pxlim[0] and pylim[0] are set to the limits of the spatial axes.
      # The new solution therefore is better.
      src_offset = (0,0)
      coords = numpy.around(wcs.coordmap(p1_spat, p2_spat, dst_shape=(ny,nx),
                            dst_offset=dst_offset, src_offset=src_offset)*512.0)/512.0

      # Next we iterate over all possible slices.
      if naxisout == 0:
         # We have a two dimensional data structure and we
         # need to reproject only this one.
         boxdat = self.dat
         newdata = map_coordinates(boxdat, coords, **interpol_dict)
      else:
         perms = []
         for lo, hi in zip(plimLO, plimHI):
         #for axnr in axnum_out:
            #pixellist = range(1, p1.naxis[axnr-1]+1)
            pixellist = range(lo, hi+1)
            perms.append(pixellist)
            # Get all permutations. Last axis is slowest axis
         z = perms[0]
         for i in range(1, len(perms)):
            z = [[x,y] for y in perms[i] for x in z]  # Extend the list with permutations
            Z = []
            for l in z :
               Z.append(flatten(l))   # Flatten the lists in the list
            z = Z

         for tup in z:
            sl = []       # Initialize slice list for current data
            slnew = []    # Initialize slice list for new, reprojected data
            nout = len(axnum_out) - 1
            for ax in range(naxis,0,-1):     # e.g. ax = 3,2,1
               if ax in [p1.lonaxnum, p1.lataxnum]:
                  sl.append(slice(None))
                  slnew.append(slice(None))
               else:
                  if issequence(tup):
                     g = tup[nout]
                  else:
                     g = tup
                  sl.append(slice(g-1, g))
                  g2 = g - plimLO[nout] + 1
                  slnew.append(slice(g2-1, g2))
                  nout -= 1
            boxdat = self.dat[sl].squeeze()
            boxdatnew = newdata[slnew].squeeze()
            # Find (interpolate) the data in the source map at the positions
            # given by the destination map and 'insert' it in the
            # data structure as a copy.
            reprojecteddata = map_coordinates(boxdat, coords, **interpol_dict)
            boxdatnew[:] = reprojecteddata

      fi = FITSimage(externalheader=newheader, externaldata=newdata)
      if not fromheader:
         # The input header was a FITSimage object. Then only one spatial map is
         # reprojected. We should bring the new FITSimage object in the same
         # state that corresponds to axis numbers and slice position of
         # the input FITSimage object. In the calling environment the
         # 'boxdat' attributes are comparable and compatible then.
         # Note that in the output there is only one slice and its
         # pixel position is alway 1 for each repeat axis.
         # In the header the crpix values are adjusted so that these
         # pixel positions 1 on the repeat axes still correspond to 
         # the right world coordinates
         slicepos = [1 for x in slicepos]
         fi.set_imageaxes(reprojobj.axperm[0], reprojobj.axperm[1], slicepos)
      return fi


   def insertspatialfrom(self, header, axnum1, axnum2):
      #---------------------------------------------------------------------
      """
      Utility function, used in the context of method
      *reproject_to()*, which returns a new header based on the
      current header, but where its spatial information
      is replaced by the spatial information of the input
      header *header*. The axis numbers of the spatial axes in
      *header* must be entered in *lon2* and *lat2*.

      :param header:
         The header from which the spatial axis information must be
         copied into the current structure.
      :type header:
         PyFITS header object or Python dictionary
      :param axnum1:
         The axis numbers of the spatial axes in the input header
      :type axnum2:
         Tuple of two integers
      :param axnum2:
         The axis numbers of the spatial axes in the destination
         header
      :type axnum2:
         Tuple of two integers

      :Notes:

         For reprojections we use only the primary header. The alternate
         header is NOT removed from the header.

         If the input header has a CD matrix for its spatial axes, and
         the current data structure has not a CD matrix, then we 
         add missing elements in the current header. If the CDELT's
         are missing for the construction of CD elements, the
         value 1 is assumed for diagonal elements.
      """
      #---------------------------------------------------------------------
      #newheader = self.hdr.copy()
      comment = True; history = True
      newheader = fitsheader2dict(self.hdr, comment, history)
      repheader = fitsheader2dict(header, comment, history)
      lon1 = axnum1[0]   # Could also be a lat. from a lat-lon map
      lat1 = axnum1[1]
      lon2 = axnum2[0]   # Could also be a lat. from a lat-lon map
      lat2 = axnum2[1]
      naxis = len(self.proj.naxis)
      axnum_out = []
      for i in range( 1, naxis+1):
         if i not in [self.proj.lonaxnum, self.proj.lataxnum]:
            axnum_out.append(i)
      wcskeys = ['NAXIS', 'CRPIX', 'CRVAL', 'CTYPE', 'CUNIT', 'CDELT', 'CROTA']
      # Clean up first in new header (which was a copy)
      for k in wcskeys:
         for ax in (lon1, lat1):
            key1 = '%s%d'%(k,ax)
            if newheader.has_key(key1):
               del newheader[key1]
      for k in wcskeys:
         for ax in ((lon1,lon2), (lat1,lat2)):
            key1 = '%s%d'%(k,ax[0])
            key2 = '%s%d'%(k,ax[1])
            if repheader.has_key(key2):
               newheader[key1] = repheader[key2]

      # Process lonpole and latpole keywords
      wcskeys = ['LONPOLE', 'LATPOLE', 'EQUINOX', 'EPOCH', 'RADESYS', 'MJD-OBS', 'DATE-OBS']
      for key in wcskeys:
         if repheader.has_key(key):
            newheader[key] = repheader[key]
         else:
            if newheader.has_key(key):
               del newheader[key]

      # Process PV and other i_j elements in the input header
      # Clean up the new header first
      wcskeys = ['PC', 'CD', 'PV', 'PS']
      for key in newheader.keys():
         for wk in wcskeys:
            if key.startswith(wk) and key.find('_') != -1:
               try:
                  i,j = key[2:].split('_')
                  i = int(i); j = int(j)     # Cope with old style PC001_002 etc.
                  for ax in (lon2,lat2):
                     if i == ax:
                        del newheader[key]
               except:
                  pass

      # Insert ['PC', 'CD', 'PV', 'PS'] if they are in the 'insert header'
      cdmatrix = False
      pcmatrix = False
      for key in repheader.keys():
         for wk in wcskeys:
            if key.startswith(wk) and key.find('_'):
               try:
                  i,j = key[2:].split('_')
                  i = int(i); j = int(j)
                  ii = jj = None
                  if i == lon2: ii = lon1
                  if i == lat2: ii = lat1
                  if j == lon2: jj = lon1
                  if j == lat2: jj = lat1
                  if not ii is None and not jj is None:
                     newkey = "%s%d_%d" % (wk,ii, jj)
                     newheader[newkey] = repheader[key]
                     if wk == 'CD':
                        cdmatrix = True
                     if wk == 'PC':
                        pcmatrix = True
               except:
                  pass

      # Fill in missing CD elements if base FITS had none.
      if cdmatrix:
         for i in axnum_out:
            newkey = "CD%d_%d"%(i,i)
            if not newheader.has_key(newkey):
               # We changed to CD formalism, but the base header
               # does not have a CD element for the non spatial axes.
               # So we contruct one using the CDELT.
               cdeltkey = "CDELT%d"%i
               if newheader.has_key(cdeltkey):
                  cdelt = newheader[cdeltkey]
               else:
                  cdelt = 1.0
               newheader[newkey] = cdelt  # E.g. CD3_3 = CDELT3
      elif pcmatrix:
         for i in axnum_out:
            newkey = "PC%d_%d"%(i,i)
            if not newheader.has_key(newkey):
               # We changed to CD formalism, but the base header
               # does not have a CD element for the non spatial axes.
               # So we contruct one using the CDELT.
               newheader[newkey] = 1.0  # E.g. CD3_3 = 1

      # Clean up ALL old style PC elements in current header
      for i in [1, naxis]:
         for j in [1, naxis]:
            key = "PC%03d%03d" % (i, j)
            if newheader.has_key(key):
               del newheader[key]

      # If 'insert header' has pc elements then copy them. If a cd
      # matrix is available then do not copy these elements because
      # a mix of both is not allowed.
      # VOG: But PC elements were already copied, so next lines are not
      # necessary.
      """
      if not cdmatrix:
         for i2, i1 in zip([lon2,lat2], [lon1,lat1]):
            for j2, j1 in zip([lon2,lat2], [lon1,lat1]):
               key = "PC%03d%03d" % (i2, j2)
               newkey = "PC%03d%03d" % (i1, j1)
               if repheader.has_key(key):
                  newheader[newkey] = repheader[key]
      """
      return newheader



   def writetofits(self, filename=None, comment=True, history=True,
                   bitpix=None, bzero=None, bscale=None, blank=None,
                   clobber=False, append=False, extname=''):
   #---------------------------------------------------------------------
      """
      This method copies current data and current header to a FITS file
      on disk. This is useful if either header or data comes from an
      external source. If no file name is entered then a file name
      will be composed using current date and time of writing.
      The name then start with 'FITS'.

      :param filename:
         Name of new file on disk. If omitted the default name is
         'FITS' followed by a date and a time (in hours, minutes seconds).
      :type filename:
         String
      :param comment: 
         If you do not want to copy comments, set parameter to False
      :type comment:
         Boolean
      :param history: 
         If you do not want to copy history, set parameter to False
      :type history:
         Boolean
      :param bitpix:
         Write FITS data in another format (8, 16, 32, -32, -64).
         If no *bitpix* is entered then -32 is assumed. Parameters
         *bzero*, *bscale* and *blank* are ignored then.
      :type bitpix:
         Integer
      :param bzero:
         Offset in scaled data. If bitpix is not equal to -32 and the values
         for bscale and bzero are None, then the data is scaled between the 
         minimum and maximum data values. For this scaling the method scale() from
         PyFITS is used with ``option='minmax'``. However PyFITS 1.3 generates an
         error due to a bug. 
      :type bzero:
         Float
      :param bscale:
         Scale factor for scaled data. If bitpix is not equal to -32 and the values
         for bscale and bzero are None, then the data is scaled between the 
         minimum and maximum data values. For this scaling the method scale() from
         PyFITS is used with ``option='minmax'``. However PyFITS 1.3 generates an
         error due to a bug. 
      :type bscale:
         Float
      :param blank:
         Value that represents a blank. Usually only for scaled data.
      :type blank:
         Float/Integer
      :param clobber:
         If a file on disk already exists then an exception is raised.
         With *clobber=True* an existing file will be overwritten.
         We don't attempt to suppres PyFITS warnings because its warning
         mechanism depends on the Python version.
      :type clobber:
         Boolean
      :param append:
         Append image data in new HDU to existing FITS file
      :type append:
         Boolean
      :param extname:
         Name of image extension if append=True. Default is empty string.
      :type  extname:
         String

      :Raises:
         :exc:`ValueError`
             You will get an exception if the shape of your external data in parameter 'boxdat'
             is not equal to the current sliced data with limits.


      :Examples: Artificial header and data:

        ::
      
            # Example 1. From a Python dictionary header
            
            header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
                      'CTYPE1' : 'RA---TAN',
                      'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
                      'CTYPE2' : 'DEC--TAN',
                      'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
                     }
            x, y = numpy.mgrid[-sizex1:sizex2, -sizey1:sizey2]
            edata = numpy.exp(-(x**2/float(sizex1*10)+y**2/float(sizey1*10)))
            f = maputils.FITSimage(externalheader=header, externaldata=edata)
            f.writetofits()

            # Example 2. From an external header and dataset.
            # In this example we try to copy the data format from the input file.
            # PyFITS removes header items BZERO and BSCALE because it reads its
            # data in a NumPy array that is compatible with BITPIX=-32.
            # The original values for *bitpix*, *bzero*, *bscale* and *blank*
            # are retrieved from the object attributes with the same name.
            
            from kapteyn import maputils

            fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
            header = fitsobject.hdr 
            edata = fitsobject.dat
            f = maputils.FITSimage(externalheader=header, externaldata=edata)
            
            f.writetofits(history=True, comment=True, 
                          bitpix=fitsobject.bitpix,
                          bzero=fitsobject.bzero,
                          bscale=fitsobject.bscale,
                          blank=fitsobject.blank,
                          clobber=True)
                          
            # Example 3. Write a FITS file in the default format, BITPIX=-32
            # and don't bother about FITS history and comment cards.
            
            f.writetofits(history=False, comment=False)
      """
   #---------------------------------------------------------------------
      # Suppress user warnings for PyFITS actions
      warnings.resetwarnings()
      warnings.filterwarnings('ignore', category=UserWarning, append=True)
      if filename is None:
         filename = getfilename('mu', 'fits')
         append = False      # Cannot append if FITS file does not exists

      if append:
         try:
            f = pyfits.open(filename)
            f.close()
         except:
            append = False

      hdu = pyfits.PrimaryHDU(self.dat)
      pythondict = fitsheader2dict(self.hdr, comment, history)

      for key, val in pythondict.iteritems():
         if key=='HISTORY' or key=='COMMENT':
            if (history and key=='HISTORY'):
               for v in val:
                 hdu.header.add_history(v) 
            elif (comment and key=='COMMENT'):
               for v in val:
                  hdu.header.add_comment(v)
         else:
            # The Python dictionary is extended with comments that
            # correspond to keywords in a header. The comments are an
            # attribute and identified by the key.
            hdu.header.update(key, val, pythondict.comment[key])
      if bitpix != None:
         # User wants to scale
         code = hdu.NumCode[bitpix]   # Undocumented PyFITS function
         if bzero is None and bscale is None:
            # Option below does not work for (at least) int16
            # because PyFITS (1.3) has a bug in its scaling method. 
            hdu.scale(code, option='minmax')
         else:
            if bzero is None:
               bzero = 0.0
            if bscale is None:
               bscale = 1.0
            hdu.scale(code, bzero=bzero, bscale=bscale)
         if blank != None:
            hdu.header['BLANK'] = self.blank
      else:
         # The output format is copied from the Numpy array
         # Default we use a format (to read the data) which
         # corresponds to BITPIX=-32. Then there should be no
         # keyword BLANK in the header, because these numbers
         # are identified by NaN's.
         if hdu.header.has_key('BLANK'): 
            del hdu.header['BLANK']

      if append:
         hdu.header.update('EXTNAME', extname)
         pyfits.append(filename, self.dat, header=hdu.header)
      else:
         hdulist = pyfits.HDUList([hdu])
         # If there is a problem, try to fix it but suppress a warning
         hdulist.writeto(filename, clobber=clobber, output_verify='silentfix')
         hdulist.close()
      # Turn warnings on
      warnings.resetwarnings()
      warnings.filterwarnings('always', category=UserWarning, append=True)
         


   def Annotatedimage(self, frame=None, **kwargs):
   #---------------------------------------------------------------------
      """
      This method couples the data slice that represents an image to
      a Matplotlib Axes object (parameter *frame*). It returns an object
      from class :class:`Annotatedimage` which has only attributes relevant for
      Matplotlib.

      :param frame:
         Plot the current image in this Matplotlib Axes object.
         If omitted, a default frame will be set using
         Matplotlib's method *add_subplot()*
      :type frame:
         A Matplotlib Axes instance

      :param kwargs:
         These parameters are keyword arguments for the constructor of
         :class:`Annotatedimage`. All of them get a default value in this
         routine. The ones for which it can be useful to change are:

         * skyout: The sky definition for graticule and world coordinates
         * spectrans: The spectral translation for the spectral axis
         * aspect: The aspect ratio of the pixels
         * basename: A name for a file on disk e.g. to store a color lut
         * cmap: A color map
         * blankcolor: The color of bad pixels,
         * clipmin: Scale colors between image values clipmin and clipmax
         * clipmax: Scale colors between image values clipmin and clipmax
         * gridmode: Set modus of str2pos() to pixels or grids

      :type kwargs:
         Python keyword arguments

      :Attributes:
         See documentation at :class:`Annotatedimage`
      
      :Returns:
         An object from class :class:`Annotatedimage`

      :Examples:
         >>> f = maputils.FITSimage("ngc6946.fits")
         >>> f.set_imageaxes(1, 3, slicepos=51)
         >>> annim = f.Annotatedimage()

         or::

            from kapteyn import maputils
            from matplotlib import pyplot as plt
   
            f = maputils.FITSimage("m101.fits")
            fig = plt.figure()
            frame = fig.add_subplot(1,1,1)
            annim = f.Annotatedimage(frame)
            annim.Image()
            annim.Graticule()
            annim.plot()
            plt.show()

      """
   #---------------------------------------------------------------------
      ar = self.get_pixelaspectratio()
      if not kwargs.has_key('basename'):      
         basename = self.filename.rsplit('.')[0]
         kwargs['basename'] = basename      # Append because it must be available
      else:
        basename = kwargs['basename']
      # Note the use of self.boxdat  instead of self.dat !!
      if frame is None:
         fig = figure()
         frame = fig.add_subplot(1,1,1, frameon=False)
      # Give the axis types in the order of the axes in the Annotatedimage.
      wcstypes = [self.wcstypes[self.axperm[0]-1], self.wcstypes[self.axperm[1]-1]]
      mplimage = Annotatedimage(frame, self.hdr, self.pxlim, self.pylim, self.boxdat,
                                self.convproj, self.axperm, wcstypes,
                                skyout=self.skyout, spectrans=self.spectrans,
                                alter=self.alter,
                                mixpix=self.mixpix, aspect=ar, slicepos=self.slicepos,
                                sliceaxnames=self.sliceaxnames,
                                sourcename=self.filename, **kwargs)
      # The kwargs are for cmap, blankcolor, clipmin, clipmax for which
      # a FITSimage object does not need to set defaults because they
      # are used in another context (e.g. image display).
      # gridmode should also be set by a user
      return mplimage


class MovieContainer(object):
#-----------------------------------------------------------------
   """
This class is a container for objects from class :class:`maputils.Annotatedimage`.
For this container there are methods to alter the visibility
of the stored objects to get the effect of a movie loop.
The objects are appended to a list with method :meth:`maputils.MovieContainer.append`.
With method :meth:`MovieContainer.movie_events` the movie is started
and keys 'P', '<', '>', '+' and '-' are available to control the movie.

* 'P' : Pause/resume movie loop
* '<' : Step 1 image back in the sequence of images. Key ',' has the same effect.
* '>' : Step 1 image forward in the sequence of images. Key '.' has the same effect.
* '+' : Increase the speed of the loop. The speed is limited by the size of the image and
        the hardware in use.
* '-' : Decrease the speed of the movie loop

One can also control the movie with method :meth:`MovieContainer.controlpanel`

Usually one creates a movie container with class class:`Cubes`

:param helptext:
   Allow or disallow methods to set an informative text about the keys in use.
:type helptext:
   Boolean
:param imageinfo:
   Allow or disallow methods to set an informative text about which image
   is displayed and, if available, it prints information about the pixel
   coordinate(s) of the slice if the image was extracted from a data cube.
:type imageinfo:
   Boolean
:param slicemessages:
   A list with messages, associated with the frames in your movie.
   Usually one uses this option to gain some efficiency because there
   is no need to construct a message each time a frame is changed.
:type slicemessages:
   List with strings.

:Attributes:
   
    .. attribute:: annimagelist

       List with objects from class :class:`maputils.Annotatedimage`.
       
    .. attribute:: indx
    
       Index in list with objects of object which represents the current image.

    .. attribute:: framespersec

       A value in seconds, representing the interval of refreshing an image
       in the movie loop.

    .. attribute:: info

       An object of class infoObject which has an attribute for the index
       of the current frame ('indx') and an attribute with an informative message
       about the current displayed slice.


    
:Examples:
   Use of this class as a container for images in a movie loop:

   .. literalinclude:: EXAMPLES/mu_movie.py


   Skip informative text on the display:
   
   >>> movieimages = maputils.MovieContainer(helptext=False, imageinfo=False)


:Methods:

.. automethod:: append
.. automethod:: movie_events
.. automethod:: controlpanel
.. automethod:: setmessages
.. automethod:: setimage
.. automethod:: imageloop
.. automethod:: toggle_images
   """
#--------------------------------------------------------------------
   class infoObject(object):
      #---------------------------------------------------------------------
      """
      Object which is a parameter of a registered callback function
      It returns the index of the current frame in attribute 'indx' and
      a message with physical coordinates of the slice position in attribute
      'mes'
      """
      #---------------------------------------------------------------------
      def __init__(self):
         self.indx = 0
         self.mes = ""
         self.cubenr = None
         self.slicemessage = ''


   def __init__(self, fig, helptext=True, imageinfo=True, slicemessages=[],
                toolbarinfo=False, sliceinfoobj=None, callbackslist={}):
      self.annimagelist = []                     # The list with Annotatedimage objects
      self.indx = 0                              # Sets the current image in the list
      self.fig = fig                             # Current Matplotlib figure instance
      self.textid = None                         # Plot and erase text on canvas using this id.
      self.helptext = helptext
      self.imageinfo = imageinfo                 # Flag. When on, display image number
      self.slicemessages = slicemessages         # Pre defined messages 
      self.numimages = 0
      self.callbackinfo = self.infoObject()      # Object that returns info to calling environment
      self.pause = True
      self.forward = True
      self.toolbarinfo = toolbarinfo
      self.sliceinfoobj = None
      self.currentcubenr = None
      self.movieframelist = []
      self.movielistindx = 0       # First element in movieframelist
      self.oldim = None            # Administration for cleaning up images
      self.compareim = None        # Background image for transparency actions
      self.callbackslist = callbackslist
      self.framespersec = 20
      self.flushcounter = 0

      if self.helptext:
         delta = 0.01
         self.helptextbase = "Use keys 'p' to Pause/Resume. '+','-' to increase/decrease movie speed.  '<', '>' to step in Pause mode."
         speedtext = " Speed=%d im/s"% (self.framespersec)
         self.helptext_id = self.fig.text(0.5, delta, self.helptextbase+speedtext, color='g', fontsize=8, ha='center')
      else:
         self.helptextbase = ''
         speedtext = ''
         self.helptext_id = None

      self.canvas = self.fig.canvas

      # Setup of the movieloop timer
      #self.pause = False
      self.framespersec = 20
      self.movieloop = TimeCallback(self.imageloop, 1.0/self.framespersec)
      # Do not start automatically, but wait for start signal
      #self.pause = True
      self.movieloop.deschedule()



   def addcallbacks(self, cbs):
      #-----------------------------------------------------------------
      """
      Helper function for to add one or more callbacks to
      the dictionary with registered callbacks. Parameter cbs should
      also be a dictionary.
      """
      #-----------------------------------------------------------------
      self.callbackslist.update(cbs)


   def callback(self, cbid, *arg):
      #-----------------------------------------------------------------
      """
      Helper function for registered callbacks
      """
      #-----------------------------------------------------------------
      if self.callbackslist.has_key(cbid):
         self.callbackslist[cbid](*arg)


   def append(self, annimage, visible=True, cubenr=0, slicemessage=None):
      #---------------------------------------------------------------------
      """
      Append object from class :class:`Annotatedimage`.
      First there is a check for the class
      of the incoming object. If it is the first object that is appended then
      from this object the Matplotlib figure instance is copied.
      
      :param annimage:
         Add an image to the list.
      :type annimage:
         An object from class :class:`Annotatedimage`.

      :param visible:
         Set the data in this object to visible or invisible. Usually one sets
         the first image in a movie to visible and the others to invisible.

      :Raises:
         'Container object not of class maputils.Annotatedimage!'
         An object was not recognized as a valid object to append.

      """
      #---------------------------------------------------------------------
      if not isinstance(annimage, Annotatedimage):
         raise TypeError, "Container object not of class maputils.Annotatedimage!" 

      annimage.origdata = None
      annimage.image.im.set_visible(visible)
      self.annimagelist.append(annimage)
      self.numimages = len(self.annimagelist)

      if slicemessage is None:
         s = "im #%d=slice:%s"%(self.numimages-1, annimage.slicepos)
         annimage.slicemessage = s
      else:
         annimage.slicemessage = slicemessage
      annimage.cubenr = cubenr


   def setmessages(self, slicemessages):
      #---------------------------------------------------------------------
      """
      Set a list with messages that will be used to display information
      when a frame is changed. This message is either printed on the image
      or returned, using a callback function, to the calling environment.
      List may be smaller than the number of images in the movie.
      A message should correspond to the movie index number.
      """
      #---------------------------------------------------------------------
      self.slicemessages = slicemessages


   def set_movieframes(self, framelist):
      #---------------------------------------------------------------------
      """
      For INTERACTIVE use.
      Set the movie frame numbers that you want to display
      in a movie loop. If the list is None or empty, all images are
      part of the movieloop. The order of the numbers is arbitrary so
      many seetings are possible (e.g. reverse order (100:0:-1) or skip
      images (0:100:2).

      Note:
      We don't perfrom any checking here. The only check is in the toggle_image
      method where we check whether the frame number in this list is a valid
      index.
      The entered list is associated to the movie container not
      to specific cubes. If we add a new cube with images and their index
      is already in 
      """
      #---------------------------------------------------------------------
      self.movieframelist = framelist


   def movie_events(self, allow=True):
      #---------------------------------------------------------------------
      """
      Connect keys for movie control and start the movie.
      Note that we defined the callbacks to trigger on the entire canvas.
      We don't need to connect it to a frame (AxesCallback). We have to
      disconnect the callbacks explicitly.

      """
      #---------------------------------------------------------------------
      if len(self.annimagelist) == 0:
         return

      # For the interaction with the movie, we want key and mouse
      # to work on the entire canvas, not just a frame.
      if allow:
         # Then disconnect first if the callbacks still exists
         if self.cidkey:
            self.fig.canvas.mpl_disconnect(self.cidkey)
         if self.cidscroll:
            self.fig.canvas.mpl_disconnect(self.cidscroll)
         # and create new ones
         self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.controlpanel)
         self.cidscroll = self.fig.canvas.mpl_connect('scroll_event', self.controlpanel)
      else:
         if self.cidkey:
            self.fig.canvas.mpl_disconnect(self.cidkey)
         if self.cidscroll:
            self.fig.canvas.mpl_disconnect(self.cidscroll)
         self.cidkey = self.cidscroll = None
         

   def controlpanel(self, event, externalkey=''):
      #---------------------------------------------------------------------
      """
      Process the key and scroll events for the movie container.
      An external process (i.e. not the Matplotlib canvas) can access
      the control panel. It uses parameter 'externalkey'.
      In the calling environment one needs to set 'event' to None.
      """
      #---------------------------------------------------------------------
      if event is None and externalkey == '':
         return
      if event is not None:
         try:
            key = event.key.upper()      # Intercept keys like 'escape'
         except AttributeError:          # It could be a scroll event
            if event.button=='up':       # 'up' is previous image. Seems more suitable using scroll wheel
               key = '.'
            if event.button=='down':
               key = ','
         except:
            return                       # Do nothing for unknown events or keys
      else:
         key = externalkey.upper()

   
      # Pause button is toggle
      if key == 'P':
         if self.pause:
            self.movieloop.schedule()
            self.pause = False
         else:
            self.movieloop.deschedule()
            self.fig.canvas.draw()                # Restore also graticules etc.
            self.pause = True
      elif key == 'START':
         if not self.pause:
            self.movieloop.deschedule()
            if not self.forward:
               self.forward = True
               self.movieloop.schedule()
            else:
               self.pause = True
         else:
            self.forward = True
            self.movieloop.schedule()
            self.pause = False
      elif key == 'STARTBACK':
         if not self.pause:
            self.movieloop.deschedule()
            if self.forward:                     # Change direction without stopping
               self.forward = False
               self.movieloop.schedule()
            else:
               self.pause = True
         else:
            self.forward = False
            self.movieloop.schedule()            
            self.pause = False
      elif key == 'STOP' and not self.pause:
         self.movieloop.deschedule()
         self.fig.canvas.draw()                   # Restore also graticules etc.
         self.pause = True

      # Increase speed of movie
      elif key in ['+', '=']:
         self.framespersec = min(self.framespersec+1, 200)     # Just to be save
         self.movieloop.set_interval(1.0/self.framespersec)
         if self.helptext:
            speedtxt = " Speed=%d im/s"% (self.framespersec)
            self.helptext_id.set_text(self.helptextbase+speedtxt)

      elif key in ['-', '_']:
         self.framespersec = max(self.framespersec-1, 1)
         self.movieloop.set_interval(1.0/self.framespersec)
         if self.helptext:
            speedtxt = " Speed=%d im/s"% (self.framespersec)
            self.helptext_id.set_text(self.helptextbase+speedtxt)

      elif key in [',','<', 'PREV']:
         if not self.pause:
            self.movieloop.deschedule()
            self.pause = True
         self.toggle_images(next=False)

      elif key in ['.','>', 'NEXT']:
         if not self.pause:
            self.movieloop.deschedule()
            self.pause = True
         self.toggle_images(next=True)

      #self.fig.canvas.flush_events()

         
   def imageloop(self, cb):
      #---------------------------------------------------------------------
      """
      Helper method to get movie loop

      :param cb:
          Mouse event object with pixel position information.
      :type cb:
          Callback object based on matplotlib.backend_bases.MouseEvent instance
      """
      #---------------------------------------------------------------------
      self.toggle_images(self.forward)
      

   def setspeed(self, framespersec):
      #---------------------------------------------------------------------
      """
      Change movie speed by changing timer interval.
      Note that speed can be < 1.0
      """
      #---------------------------------------------------------------------
      self.framespersec = max(framespersec, 0)  #max(framespersec-1, 1)
      self.framespersec = min(self.framespersec, 200)
      self.movieloop.set_interval(1.0/self.framespersec)
      flushprint("interval in sec=%f"%(1.0/self.framespersec))
      if self.helptext:
         speedtxt = " Speed=%d im/s"% (self.framespersec)
         self.helptext_id.set_text(self.helptextbase+speedtxt)



   def setimage(self, i, force_redraw=False):
      #---------------------------------------------------------------------
      """
      Set a movie frame by index number. If the index is invalid, then do
      nothing.

      :param i:
          Index of movie frame which is any number between 0 and the
          number of loaded images.
      """
      #---------------------------------------------------------------------
      # If the index is invalid, then do nothing (also no warning)
      if i is None:
         return
      flushprint("MAPUTILS setimage imnr, numimages=%d %d"%(i,self.numimages))
      if 0 <= i < self.numimages:         
         self.toggle_images(indx=i, force_redraw=force_redraw)



   def toggle_images(self, next=True, indx=None, force_redraw=False):
      #---------------------------------------------------------------------
      """
      Toggle the visible state of images either by a timed callback
      function or by keys.
      This toggle works if one stacks multiple images in one frame
      with method :meth:`MovieContainer.append`.
      Only one image gets status visible=True. The others
      are set to visible=False. This toggle changes this visibility
      for images and the effect, is a movie.
      
      :param next:
         Step forward through list if next=True. Else step backwards.
      :type next:
         Boolean
      :param indx:
         Set image with this index
      :type indx:
         Integer
      :param force_redraw:
         Boolean flag to force everything to be repainted, which would
         otherwise only occur if there is a change of cubes
      """
      #---------------------------------------------------------------------
      xd = None                              # A position in the previous image
      yd = None
      if indx is not None:
         # User/programmer required an explicit image, not a next or
         # previous one
         newmovieframeindx = indx
         self.indx = newmovieframeindx
      else:
         # We get a request to show a next or a previous image
         fromlist = self.movieframelist      
         if fromlist:
            # This was not a requested image (with given indx) but a request
            # to show the next image, the next in a user defined 
            # sequence ('movieframelist'). Note that if we have a user defined
            # list with movie frames, then we have another index. This index
            # 'movielistindx' is related to the list with movie frames.
            movielistindx = self.movielistindx    # Retrieve
            if next:
               if movielistindx + 1 >= len(self.movieframelist):
                  movielistindx = 0
               else:
                  movielistindx += 1
            else:
               if movielistindx - 1 < 0:
                  movielistindx = len(self.movieframelist) - 1
               else:
                  movielistindx -= 1
            newmovieframeindx = self.movieframelist[movielistindx]
            # What if the index from the movie frame list is not valid?
            # Then there is nothing to do because there is no associated image.
            if newmovieframeindx  >= len(self.annimagelist) or newmovieframeindx < 0:
               return
            self.movielistindx = movielistindx    # Store for next
         else:
            currindx = self.indx                  # Retrieve
            if next:
               if currindx + 1 >= len(self.annimagelist):
                  currindx = 0
               else:
                  currindx += 1
            else:
               if currindx - 1 < 0:
                  currindx = len(self.annimagelist) - 1
               else:
                  currindx -= 1                  
            newmovieframeindx = currindx
            self.indx = newmovieframeindx         # Store for next

      oldim = self.oldim

      # TODO: Resetten van images na transparantie acties. Moet nog verder uitgewerkt
      # worden.
      
      # Make sure that the image used to compare to the current is reset to invisible
      if self.compareim:   
          self.compareim.image.im.set_alpha(1.0)
          self.compareim.image.im.set_visible(False)
          self.compareim = None
          if oldim:
             oldim.image.im.set_alpha(1.0)
      
      
      if oldim is not None:
        # If the old image was splitted, it does not contain the original data
        if oldim.origdata is not None:
           oldim.data[:] = oldim.origdata[:]
           oldim.origdata = None
           oldim.image.im.set_data(oldim.data)  # Necessary!
           # Note that we need to restore the blank color and transparancy
           # both in one time. We are sure that the attribute old_blankcolor
           # exists because we have splitted the image (it has a non empty
           # 'origdata' attribute.
           oldim.set_blankcolor(oldim.old_blankcolor[0], oldim.old_blankcolor[1])
           splitimage = self.annimagelist[oldim.splitmovieframe]
           splitimage.image.im.set_visible(False)           
        # Make the old image invisible and disconnect existing callbacks
        oldim.image.im.set_visible(False)
        flushprint("Toggle images: Made oldim (%d) invisible"%(id(oldim)))
        oldim.disconnectCallbacks()
        if self.toolbarinfo:
           # Deschedule the old toolbarinfo method, but first, store
           # the mouse position in display coordinates, so that we can
           # re-use them for another frame. Note that the movie can consist of
           # frames from different cubes and each cube is associated with
           # a separate frame (i.e. mpl axes object)
           x = oldim.X_lastvisited
           y = oldim.Y_lastvisited
           if not None in [x,y]:
              xd, yd = oldim.frame.transData.transform((x,y))           
           flushprint("Toolbarinfo voor oldim %s DISCONnected"%(str(id(oldim))))
           if None in [x,y]:
              flushprint("Toolbarinfo ziet None in x,y")
           else:
              flushprint("Toolbarinfo ziet valide x,y")

      #self.indx = currindx   #newindx
      newim = self.annimagelist[newmovieframeindx]      
      newim.image.im.set_visible(True)
      flushprint("Setimage movie index=%d"%(newmovieframeindx))      
      
      info = newim.infoobj
      if info:
         info.set_text(newim.slicemessage)

      # Handle lines and text if one changes cubes.
      #print "newim.cubenr", newim.cubenr, self.currentcubenr
      if (newim.cubenr != self.currentcubenr) or force_redraw:
          #print "Make onzichtbaar ", oldim
          flushprint("KUBUS WISSELING?? Current cube nr, newcubenr=%s %d"%(str(self.currentcubenr),newim.cubenr))
          # Depending on unknown factors, sometimes a remnant of the previous
          # image remains on the screen. This is only visible if the new image
          # is smaller than the old one. In any case, if cubes change, clear their
          # contents of the old axes object.
          if newim.cubenr != self.currentcubenr and oldim:
             #self.canvas.figure.clf()
             #newim.frame.draw_artist(newim.image.im)
             pass
             #
             #newim.image.im.set_visible(True)
             #oldim.frame.clear()
             #oldim.image.im.set_visible(False)
          self.currentcubenr = newim.cubenr  # currentcubenr is None for the first image
          if oldim is not None:
            for li in oldim.linepieces:
               li.set_visible(False)
            for tl in oldim.textlabels:
               tl.set_visible(False)
            if oldim.infoobj:
               oldim.infoobj.set_visible(False)
            if oldim.hascolbar and oldim.cbframe:
               oldim.cbframe.set_visible(False)
            oldim.splitcb.deschedule()

          for li in newim.linepieces:
             li.set_visible(True)
          for tl in newim.textlabels:
             tl.set_visible(True)
             flushprint("Ik zet in toggle_images textlabel %d %s op True"%(id(tl), str(tl)))
          if newim.infoobj:
             newim.infoobj.set_visible(True)
          newim.splitcb.schedule()
          flushprint("force_redraw=%s"%(str(force_redraw)))
          if newim.cbframe:
             flushprint("Nieuw image heeft colorbar frame")
             if newim.hascolbar:
                flushprint("Nieuw image heeft colorbar op YES")
                newim.cbframe.set_visible(True)
             else:
                flushprint("Nieuw image heeft colorbar op NO")
                newim.cbframe.set_visible(False)
          #newim.frame.draw_artist(newim.image.im)
          #print "Maak zichtbaar voor indx", newmovieframeindx
          self.canvas.draw()                                          # Draw all
          #self.canvas.blit(newim.frame.bbox)
          self.canvas.flush_events()
      else:
          flushprint("Current cube nr, newcubenr=%d %d"%(self.currentcubenr,newim.cubenr))
          flushprint("They are equal so blit now")
          newim.frame.draw_artist(newim.image.im)
          if info:
             newim.frame.draw_artist(info)
          # Restore the graticule grid! Note that zorder could not help us
          # because blitting the image always puts the image on top.
          for li in newim.linepieces:  
             flushprint("newim li=%d"%(id(li)))
             newim.grat.frame.draw_artist(li)
          for tl in newim.textlabels:
             flushprint("newim tl=%d %s"%(id(tl), str(tl.get_text())))
             newim.frame.draw_artist(tl)
          #for tl in newim.textlabels:  # Can be skipped if placed outside frame
          #   tl.set_visible(True)
          #   newim.frame.draw_artist(tl)
          #   flushprint("newim drawartist (hier teveel??) Text id=%d"%(id(tl)))
          self.canvas.blit(newim.frame.bbox)
          #self.canvas.draw()

      # We need to keep track of the last image viewed in a cube because
      # if we jump (in the calling environment) to another cube and back,
      # there should be a way to restore the last displayed image for this
      # cube.
      #C = self.cubelist[newim.cubenr]
      #C.lastimagenr = self.indx - C.movieframeoffset

      newim.interact_imagecolors()
      if self.toolbarinfo:
         newim.interact_toolbarinfo()
         newim.interact_writepos(pixfmt=newim.coord_pixfmt,
                                 dmsprec=newim.coord_dmsprec,
                                 wcsfmt=newim.coord_wcsfmt,
                                 zfmt=newim.coord_zfmt,
                                 gipsy=True,
                                 grids=newim.coord_grids,
                                 world=newim.coord_world,
                                 worlduf=newim.coord_worlduf,
                                 imval=newim.coord_imval,
                                 g_typecli=newim.coord_tocli,
                                 g_tolog=newim.coord_tolog,
                                 g_appendcr=newim.coord_appendcr)
         
         flushprint("Toolbarinfo voor newim %s CONnected"%(str(id(newim))))
         cb = newim.toolbarkey
         if not None in [xd, yd]:
            cb.xdata, cb.ydata = newim.frame.transData.inverted().transform((xd,yd))
            # Trigger the toolbarinfo method to get the right image data value (z)
            # for this new image. Note that events from the panels come always from
            # outside the image frame. In the eventpanel methods, the values
            # for the last position of the current image are reset to None. Then
            # skip the update of the toolbar message.
            newim.mouse_toolbarinfo(cb)

      self.oldim = newim
      self.callbackinfo.mes = newim.slicemessage
      self.callbackinfo.indx = newmovieframeindx
      self.callbackinfo.cubenr = newim.cubenr
      self.callbackinfo.slicemessage = newim.composed_slicemessage
      self.callback('movchanged', self.callbackinfo)

      # After setting a (new) image, we want to be sure that
      # all events are flushed, because otherwise, it seems
      # that Maplotlob skips redrawing the canvas and a change
      # of cubes will not result in a change of the colorbar
      # The flush method below, solves this problem.
      # It also seems necessary to put this method after setting
      # the new toolbar and image interactions.
      """if self.flushcounter > 10:
         self.canvas.flush_events()
         flushprint("********* FLUSH **********")
         self.flushcounter = 0
      else:
         self.flushcounter += 1
      """
      flushprint("------ Einde setimage----------")

#------------------------------------------------------------------------
"""Background

We developed some methods that enable a user to inspect slices of a data cube.
One of these methods displays an image which is part of a movie.
A position in such an image
sets the data in two slices which are shown as two panels. For each panel,
one axis is shared with the movie frame image.
The other is the third axis in the cube.
What we try to do is to copy the functionality of the display program
GIDS (Groningen Image Display Server). In that application one enters a number
of images. For example in a simple case: one has a RA-DEC-VELO cube. The
interesting parts of the data are in slices RA-VELO and DEC-VELO. These
slices need an extra position which is derived from the mouse position in a
movieframe.
In GIDS a series of movie frames create a cube. It doesn't matter if these
images have their origin in different files. It also doesn't matter if
there is an n-th axis (n>3). The second axis of a slice is always the third
axis of the movie cube. For the slices one enters movie frame numbers which
sets the values along the new slice axis.

Example: With GIDS and a big data cube with many spectral observations one can
enter images at each second velocity to speed up a movie. If the number of
spectral images are too many to display in an XV slice, one can enter a
selection of movie images which contribute to the spectral axis in your
XV slices.

Example: In GIDS one can upload 2-dim subsets of a 4 dimensional data set.
The order of the slices are set by grids on the repeat axes (e.g. FREQ, STOKES)
and this will also be the order in the movie cube. This will also be the default
order along the V axis in you slice, which is strictly speaking not a V axis
anymore.

In the XV slices there are also mouse interactions. The position in V
sets a movie frame. A position in X in one panel updates the other XV panel.

We try to achieve the same functionality as in GIDS but in the
context of module maputils. That is, we don't use X-memory to store the
movie data (the movie cube). With maputils we build a sequence of movieframes
and from this list with 2-dim data arrays we retrieve the slice data.
So one of the slice axes is the index axis of the movie frames and the other
is one of the movie frame axes.

For a sequence of data sources (data sets), we extend the movie loop with
new frames and add slice panels (instead of extending the previous panels).


TODO: Server functionality in the new viewer by adding communication via
wkey()'s
-Verbeteren positie informatie
"""
#------------------------------------------------------------------------


class Cubeatts(object):
   #----------------------------------------------------------------------------
   """
   Class that sets cube attributes
   """
   #----------------------------------------------------------------------------
   def __init__(self, frame, fitsobj, axnums, slicepos=[], pxlim=None, pylim=None,
                vmin=None, vmax=None, xpanel=[], ypanel=[], hasgraticule=False,
                gridmode=False, hascolbar=True, pixelaspectratio=None,
                clipmode=0, clipmn=(4,5), callbackslist={}):
      self.frame = frame
      self.pxlim = pxlim
      self.pylim = pylim
      # If there are no limits given, then calculate some defaults
      if pxlim is None:
         self.pxlim = (1, fitsobj.axisinfo[axnums[0]].axlen)
      if pylim is None:
         self.pylim = (1, fitsobj.axisinfo[axnums[1]].axlen)
      # 'slicepos' is a list with tuples. Each tuple sets positions
      # on repeat axes. For 2-dim images, the list is empty
      self.slicepos = slicepos
      self.fitsobj = fitsobj                    # The source from which the data is extracted
      self.pixelaspectratio = pixelaspectratio
      if slicepos:
         self.nummovieframes = len(slicepos)
      else:
         self.nummovieframes = 1
      self.hasgraticule = hasgraticule
      self.gridmode = gridmode                  # If True then positions are displayed in grids
      self.xold = None                          # Remember last visited position
      self.yold = None
      self.framep1 = None                       # Initialize the panel frames
      self.framep2 = None
      self.annimp1 = None                       # Annotated image object for panels 1 and 2
      self.annimp2 = None
      self.linepieces = set()
      self.textlabels = set()
      self.grat = None
      self.divider = None                       # To make axes locatable
      # Axis numbers for images and extra axis which
      # is one of the repeat axes (usually velocity is axnum 3)

      # A FITSimage object has two attributes that could have been useful
      # to find names and positions of the slice (repeat) axes.
      # However these are just default values if method set_imageaxes() has not
      # been called. This call is postponed until we create Annotatedimage
      # objects. So for now we need to build our own axis numbers array.
      # The first two numbers are the axis numbers of the image (one or a
      # sequence that belongs to this cube).
      anums = [axnums[0], axnums[1]]
      n = fitsobj.naxis
      if (n > 2):
         # Note that we have our slice axes always in the same order as in the header
         for i in range(n):
            axnr = i + 1
            if axnr not in anums:
               anums.append(axnr)         
      self.axnums = anums                       # To extract slices we need to know the axis order
      self.movieframeoffset = None              # Gets a value after loading a cube
      # What are the movieframes from which we want to extract the data?
      # Set these frames in a list. One list per data cube
      # Show all slices by default
      # TODO: Het is netter om set_panelframes() te gebruiken maar
      # dan crasht het programma omdat er nog geen cubelist is.
      #self.set_panelframes(xpanel, panel='X')
      #self.set_panelframes(xpanel, panel='Y')
      #self.set_panelXframes(xpanel)
      #self.set_panelYframes(ypanel)
      self.panelXframes = []    # Initialize the slice panels
      self.panelYframes = []
      self.cmap = None
      """
      if vmin is None or vmax is None:
         vmi, vma = fitsobj.get_dataminmax(box=False)
      if vmin is None:
         self.vmin = vmi
      if vmax is None:
         self.vmax = vma
      """
      self.panelscb = None
      self.splitcb = None
      self.origdata = None
      # Set the properties for the slice panel images
      self.set_xvprojections()
      self.preparefirst = True
      self.cnr = None                           # Cube number in container

      #self.lastimagenr = 0
      self.hascolbar = hascolbar
      #if hascolbar and self.cbframe is None:
      self.cbframe = None
      self.imagesinthiscube = []
      self.shortfilename = os_basename(self.fitsobj.filename)
      self.callbackslist = callbackslist
      # Crosshair lines
      lineprops = {'animated': True}
      self.lineh = self.frame.axhline(self.frame.get_ybound()[0], visible=False, **lineprops)
      self.linev = self.frame.axvline(self.frame.get_xbound()[0], visible=False, **lineprops)
      self.needclear = False
      self.background = None

      # Clip level related attributes
      self.vmin = vmin
      self.vmax = vmax
      self.datmin = None
      self.datmax = None
      self.mean = None
      self.rms = None      
      self.clipmode = clipmode
      self.clipmn = clipmn
      self.scalefac = None



   def callback(self, cbid, *arg):
      #-----------------------------------------------------------------
      """
      Helper function for registered callbacks
      """
      #-----------------------------------------------------------------
      if self.callbackslist.has_key(cbid):
         self.callbackslist[cbid](*arg)


   def cubestats(self):
      #-------------------------------------------------------------------------
      """
      Given the datmin, datmax, mean and rms attributes of all the images in
      this cube, calculate the same attributes for the entire cube

      Formulas are explained in article on:
      http://en.wikipedia.org/wiki/Standard_deviation
      Section: Population-based statistics
      """
      #-------------------------------------------------------------------------
      if not self.imagesinthiscube:      # To be save. Here is nothing to do
         return
      alldatmin = []
      alldatmax = []
      allmean = []
      allrms  = []
      for aim in self.imagesinthiscube:
         if aim.datmin and numpy.isfinite(aim.datmin):  # Not None, Nan or +-inf
            alldatmin.append(aim.datmin)
         if aim.datmax and numpy.isfinite(aim.datmax):
            alldatmax.append(aim.datmax)
         if aim.mean and numpy.isfinite(aim.mean):
            allmean.append(aim.mean)
         if aim.rms and numpy.isfinite(aim.rms):
            allrms.append(aim.rms)
      if len(alldatmin):
         self.datmin = min(list(alldatmin))  # Could be one value
      if len(alldatmax):
         self.datmax = max(list(alldatmax))
      # To calculate the mean and rms of all the data we use a formula
      # to get mean and rms as a result of a list with means and rms's
      # for data sets with an equal number of data points
      n = 0; mean_t = 0.0; rms_t = 0.0
      for mean, rms in zip(allmean, allrms):
         flushprint("mean, rms=%f %f"%(mean, rms))
         if rms and mean:
            flushprint("n=%d"% n)
            n += 1
            mean_t += mean
            rms_t += rms*rms + mean*mean
      if n:
         self.mean = mean_t/n
         self.rms = numpy.sqrt(rms_t/n - (self.mean*self.mean))
      else:
         self.mean = self.rms = None


   def imagestats(self):
      #-------------------------------------------------------------------------
      """
      For each image in this cube, calculate the values for the
      data min., data max., mean and rms.
      """
      #-------------------------------------------------------------------------
      if not self.imagesinthiscube:      # To be save. Here is nothing to do
         return
      for aim in self.imagesinthiscube:
         aim.datmin, aim.datmax, aim.mean, aim.rms = aim.get_stats()
      # Get same attributes applied to all cube images
      self.cubestats()


   def setcubeclips(self, clipmin=None, clipmax=None, clipmode=0, clipmn=(4,5)):
   #----------------------------------------------------------------------------
   # Purpose: Set the norm limits for image scaling to all images in this cube
   #
   # Use this method if you want to use the same clip levels for all the
   # images in this cube. Each image has attributes clipmin and clipmax that
   # are finite numbers (forced by the constructor of Annotatedimage.
   # Use the min and max of all these clip values to set the norm for all
   # images in the list.
   #----------------------------------------------------------------------------
      if not self.imagesinthiscube:      # To be save. Here is nothing to do
         return
      flushprint("setcubeclips: clipmin, max=%s %s"%(str(clipmin), str(clipmax)))
      if None in (clipmin, clipmax) or clipmode in [1,2]:
         # Set values for cube.datmin, cube.datmax etc.
         if None in [self.datmin, self.datmax, self.mean, self.rms]:
            self.cubestats()
      flushprint("setcubeclips: clipmode=%d"%clipmode)
      if clipmode == 0:
         if clipmin is None:
            clipmin = self.datmin
         if clipmax is None:
            clipmax = self.datmax

      if clipmode == 1:
         clipmin = self.datmin
         clipmax = self.datmax
         flushprint("setcubeclips: MODE=1  clipmin, max=%s %s"%(str(clipmin), str(clipmax)))

      if clipmode == 2:
         if self.mean and self.rms:         
            clipmin = self.mean - clipmn[0]*self.rms
            clipmax = self.mean + clipmn[1]*self.rms
            flushprint("setcubeclips: MODE=2  mean, rms=%s %s"%(str(self.mean), str(self.rms)))
            flushprint("setcubeclips: MODE=2  clipmin, max=%s %s"%(str(clipmin), str(clipmax)))

      if clipmin is None:
         if clipmax is not None:
            clipmin = clipmax - 1.0
         else:
            clipmin = 0.0
      if clipmax is None:
         if clipmin is not None:
            clipmax = clipmin + 1.0
         else:
            clipmax = 1.0

      self.vmin = clipmin
      self.vmax = clipmax
      for aim in self.imagesinthiscube:
         # Prevent exception if min > max
         aim.image.im.set_clim(min(clipmin,clipmax), max(clipmin,clipmax))

      canvas = aim.frame.figure.canvas
      canvas.draw()


   def set_slicemessages(self, spectrans):
      #-------------------------------------------------------------------------
      """
      Change slice messages to comply with this spectral translation.
      Apply message change to all Annotatedimage objects in this cube.

      :param spectrans:
         A coded spectral translation e.g. FREQ, WAVE or VRAD
         with translation code (e.g. V2F) or questionmarks as
         wildcard (e.g. VRAD-???)
      :type spectrans:
         String

      :Notes:

         After the slice messages are set, we have to refresh the image
         if a message was required on the image. After loading this is done
         by setting the image to the first loaded image (for this cube).
         If this method is called externally, the caller is responsible for
         refreshing the current image.
      """
      #-------------------------------------------------------------------------
      flushprint("set_sliceimages = %s"%(str(self.slicepos)))
      if not self.imagesinthiscube:      # To be save. Here is nothing to do
         return

      axnames = []
      crpix = []
      # Note: these are the axis numbers with sorted repeat axes!
      for an in self.axnums[2:]:
         axi = self.fitsobj.axisinfo[an]
         if axi.wcstype == 'spectral' and spectrans:
            axnames.append(spectrans.split('-')[0])
         else:
            axnames.append(axi.axname)
         # If we work in grid mode, we need to translate the slice positions
         # to grids. For this we need the CRPIX header values.
         crpix.append(axi.crpix)
         
      for i, aim in enumerate(self.imagesinthiscube):         
         j = i + self.movieframeoffset
         # This message should contain only the information about
         # the slice. It will be reused for the mouse position
         # in physical coordinates. The non image axes of the slice
         # panels, could be composed of multiple axes (e.g. FREQ, STOKES)
         # and cannot be represented by module wcsgrat.
         # We compose a more informative string (sc) to be used in the calling
         # environment using the 'slicemes' callback.
         s = ""
         sc = "im%3d: %s "%(j, os_basename(self.fitsobj.filename))
         if self.slicepos:
            # Note that each fitsobj has one 'slicepos'. This value
            # can be a tuple if there is more than one repeat axis
            self.fitsobj.slicepos = self.slicepos[i]
            vlist, ulist = self.fitsobj.slice2world(skyout=None,
                                                    spectra=spectrans,
                                                    userunits=None)
            #print "SLICEPOS=", self.slicepos[i], vlist, ulist,self.axnums            
            if not issequence(self.slicepos[i]):
               splist = [self.slicepos[i]]
            else:
               splist = self.slicepos[i]

            # If the slice position is required in grids, convert using CRPIX values
            if self.gridmode:
               #flushprint("len(splist=%d, %s %s"%(len(splist),str(splist), str(aim.projection.crpix)))
               splist = [splist[k]-crpix[k] for k in range(len(splist))]

            for v, u, aname, sp in zip(vlist, ulist, axnames, splist):
               #s += "%s(%*d)=%+-*g"%(aname, 4, sp, 10, v)
               s += "%s(%d)=%+-g"%(aname, sp, v)
               if u:
                  s += " (%s)"%u
               s += "  "
            s = s.strip()
         aim.slicemessage = s
         aim.composed_slicemessage = sc + ' '  + s
         aim.imnr = j
         aim.spectrans = spectrans



   def set_panelXframes(self, framelist):
      #-------------------------------------------------------------------------
      """
      For the current cube, build a list with indices which all represent a
      slice in the panel that is parallel to the X axis.
      Prepare a data array for this slice panel.
      Remember that you have set up a 'movie cube' (which can be composed out
      of more than one data source). In this movie cube one can imagine a third
      axis perpendicular to the image axes. Along this axis, one can take a
      slice through the data. The sliced data can be represented in an image
      (e.g. a position-velocity diagram). The positions on the third axis
      are indices of the movie frame array and need not to be contiguous or
      ordered.
      """
      #-------------------------------------------------------------------------
      self.panelXframes = framelist   # Should be filtered and corrected for cube offset in movie frames
      if self.panelXframes:
         lx = self.pxlim[1] - self.pxlim[0] + 1# Length of x axis of movie image
         ly = len(self.panelXframes) # Length of new axis is equal to the number
                                                      # of selected movie frames
         self.Mx = numpy.zeros((ly,lx))             # Image for horizontal panel



   def set_panelYframes(self, framelist):
      #-------------------------------------------------------------------------
      """
      For the current cube, build a list with indices which all represent a
      slice in the panel that is parallel to the Y axis.
      Prepare a data array for this slice panel.
      See also text at set_panelXframes().
      """
      #-------------------------------------------------------------------------
      self.panelYframes = framelist
      if self.panelYframes:
         ly = self.pylim[1] - self.pylim[0] + 1
         lx = len(self.panelYframes)
         self.My = numpy.zeros((ly,lx))               # Image for vertical panel



   def set_xvprojections(self):
      #----------------------------------------------------------------------------
      """
      Is there an axis in the set that is not part of the image?
      We may need this axis later if we need a 'missing' spatial axis
      """
      #----------------------------------------------------------------------------
      ap = set(range(1,self.fitsobj.naxis+1))
      apsub = set(self.axnums[:2])
      apdiff = list(ap.difference(apsub))
      if apdiff:
         dummyax = apdiff[0]
      else:
         dummyax = None

      # Find a suitable axis to represent the movieframe index axis
      # For the panel that shares the x axis, this will be the y axis.
      fo = self.fitsobj
      if dummyax:
         spatials = [fo.proj.lataxnum, fo.proj.lonaxnum]
         if self.axnums[0] in spatials and dummyax in spatials:
            convproj = fo.proj.sub((self.axnums[0], dummyax))
         else:
            if not (self.axnums[0] in spatials or dummyax in spatials):
               axperm = (self.axnums[0], dummyax)
               convproj = fo.proj.sub(axperm)
            else:
               if fo.proj.lonaxnum in [dummyax,self.axnums[0]]:
                  axperm = (self.axnums[0], dummyax, fo.proj.lataxnum)
                  convproj = fo.proj.sub(axperm)
               else:
                  axperm = (self.axnums[0], dummyax, fo.proj.lonaxnum)
                  convproj = fo.proj.sub(axperm)
         self.panelx_axperm = (self.axnums[0], dummyax)
      else:
         convproj = fo.convproj
         self.panelx_axperm = (self.axnums[0], self.axnums[1])
      self.panelx_proj = convproj
      
      # Now for the slice panel that shares the y axis
      if dummyax:

         spatials = [fo.proj.lataxnum, fo.proj.lonaxnum]
         if self.axnums[1] in spatials and dummyax in spatials:
            convproj = fo.proj.sub((dummyax, self.axnums[1]))
         else:
            if not (self.axnums[1] in spatials or dummyax in spatials):
               axperm = (self.axnums[1], dummyax)
               convproj = fo.proj.sub(axperm)
            else:
               if fo.proj.lonaxnum in [dummyax,self.axnums[1]] :
                  axperm = (dummyax, self.axnums[1],  fo.proj.lataxnum)
                  convproj = fo.proj.sub(axperm)
               else:
                  axperm = (dummyax, self.axnums[1], fo.proj.lonaxnum)
                  convproj = fo.proj.sub(axperm)
         self.panely_axperm = (dummyax, self.axnums[1])
      else:
         convproj = fo.convproj
         self.panely_axperm = (self.axnums[0], self.axnums[1])
      self.panely_proj = convproj
      
       

class Cubes(object):
   #----------------------------------------------------------------------------
   """
A container with Cubeatts objects. With this class we build the movie container
which can store images from different data cubes.

:param toolbarinfo:
   This flag sets an informative message in the toolbar. For the QT backend
   the message is printed on a separate line at the bottom of the plot.
   The message contains information about the position of the cursor and
   when possible, it will show the corresponding world coordinates. Also the
   pixel value is printed.
   #TODO: Afkappen voor andere backends verbeteren
   #TODO: Info voor panels is nog niet volledig. Hier nog image waarde aan toevoegen
          indien dit niet al te ingewikkeld is.
   #TODO: Colorbar toevoegen
   If you display a slice panel, then the information about the mouse position
   is different. For example the y direction of the horizontal slice panel is
   not a function of a world coordinate, but a function of movie frame number.
   This number can be used to find a world coordinate.
:type toolbarinfo:
   Boolean



:param callbacklist:
      Possible callbacks:

      progressbar:
      An (external) object that displays the progress. It should have at
      least four methods: setMinimum(), setMaximum(), setValue() and reset()
      For example in a QT gui we can define a progressbar with
      'QProgressBar()'. This routine provides the necessary methods.
      The QT progressbar displays percentages. We set its range between
      zero and the number of frames contained in the new cube.
      Usually, loading 1 image is to fast to use a progressbar, but
      with numerous images, this can be useful.


:type callbacklist:
      Python dictionary with function names and function pointers



:Example:

   myCubes = maputils.Cubes(toolbarinfo=True, printload=True)
   fitsobject = maputils.FITSimage('rense.fits')
   slicepos = []
   for i in range (1,100):
      slicepos.append(i)
   axnums = (1,2,3)
   fr1 = fig.add_axes(box, label=fitsobject.filename, frameon=False)
   myCubes.append(fr1, fitsobject, slicepos, axnums=axnums)
   myCubes.splitmovieframe = 30
   plt.show()
   TODO: make van attribute splitmovieframe een methode waarin je de validiteit test

   # Changelog:
   -18 aug 2011: Changed toolbar message for QT backends, i.e. do not cut off
   """
   #----------------------------------------------------------------------------
   # Class variables:
   # Define the required colorbar width. We take a fixed part of the
   # frame, so that it scales up when the screen becomes larger
   colbwidth = 0.06

   # The timerlist must be accessible by all objects of this class.
   timerlist = []    # One list with timer callbacks to load all cubes
   
   def __init__(self, fig, toolbarinfo=False, imageinfo=True,
                printload=False, helptext=True,
                callbackslist={}):


      self.numcubes = 0
      self.movieimages = MovieContainer(fig, helptext=helptext,
                                        toolbarinfo=toolbarinfo,
                                        imageinfo=imageinfo)
      # Initialize two callback id's for reacting to keys and scroll wheel
      # for movie actions in the figure. These are not of type AxexCallbacks
      # because the actions are not restricted to a frame.
      self.fig = fig
      self.movieimages.cidkey = None
      self.movieimages.cidscroll = None
      self.imageinfo = imageinfo
      self.imageloadnr = 0                   # A counter for images in all cubes
      self.movieframecounter = 0
      self.currentcube = 0
      self.cubelist = []
      self.maxframe = [None]*4               # Corner positions of biggest frame
      # Every cube represents an number of movie frames
      # If there is more than one cube, you need an offset to find the
      # correspondence between a position in a slice panel, the cube and the
      # index of the movie frame.
      self.movieframeoffset = 0
      self.cubeindx = 0
      self.loadcb = None                # The callback for the image load method
      self.cmap = None
      self.printload = printload
      self.splitmovieframe = 0
      self.panelmode = 0
      self.infoobj = None
      # Usually one loads images in Matplotlib, selects one image to
      # be visible and the call show() to display this image. By setting images
      # to visible/invisible, we create a movie effect.
      # In our load process, we want to show each loaded image immediately.
      # so not only the movie loop software, but also the load procedure
      # should know how to make an image visible or invisible. For the load
      # procedure we use attribute 'lastimage' to keep track of which image
      # was the last one that was on display.
      self.lastimage = None
      # We want to start to display the images as big as possible, given the
      # size of the mpl canvas. We do this only one time, because appended
      # cubes will follow this adjustment. So we need a flag to inspect
      # if this figure adjustment has ben done or not.
      self.figadjust = False
      #self.resizecb = CanvasCallback(self.reposition, fig.canvas, 'draw_event')
      self.resizecb = CanvasCallback(self.reposition, fig.canvas, 'resize_event')
      NavigationToolbar2.ext_callback = self.reposition
      NavigationToolbar2.ext_callback2 = self.set_graticules
      Figure.ext_callback = self.reposition    # From the subplot configurator
      
      # If we want to redraw after a draw_event, we must prevent that this draw
      # action also executes method 'reposition'. So we need a flag to prevent this.
      #self.resizecb.update = True                    
      flushprint("I CONNECT DRAW_EVENT to REPOSITION()")
      self.callbackslist = callbackslist
      if self.callbackslist.has_key('progressbar'):
         self.progressbar = self.callbackslist['progressbar']
      else:
         self.progressbar = None
      if self.callbackslist.has_key('memory'):
         self.memorystatus = self.callbackslist['memory']
      else:
         self.memorystatus = None
      # We need administration to keep track of the number of cubes that
      # define side panels. Then it is possible to use maximum space for these
      # panels to plot.
      self.numXpanels = 0
      self.numYpanels = 0
      # Needed to take re-position action of side panels when h/wspace changes
      self.previous_subpars = (fig.subplotpars.hspace, fig.subplotpars.wspace)
      self.crosshair = False
      

   def callback(self, cbid, *arg):
      #-----------------------------------------------------------------
      """
      Helper function for registered callbacks
      """
      #-----------------------------------------------------------------
      if self.callbackslist.has_key(cbid):
         self.callbackslist[cbid](*arg)

      
   def cleanupall(self):
      #-------------------------------------------------------------------------
      """
      Cleanup the contents of this object (destructor). We need a separate function
      to cleanup because our class contains circular references so it will not
      be touched by Python's garbage collector. Implementing a __del__ method
      also does not work because that decreases the reference counter with one,
      but it was bigger than 1 to begin with. See also:
      http://www.electricmonk.nl/log/2008/07/07/python-destructor-and-garbage-collection-notes/
      
      """
      #-------------------------------------------------------------------------
      cidkey= self.movieimages.cidkey
      if cidkey:
         self.canvas.mpl_disconnect(cidkey)  # Prevent problems while loading
      cidscroll = self.movieimages.cidscroll
      if cidscroll:
         self.canvas.mpl_disconnect(cidscroll)  # Prevent problems while loading




   def append(self, frame, fitsobj, axnums, slicepos=[],
              pxlim=None, pylim=None, vmin=None, vmax=None,
              xpanel=[], ypanel=[], hasgraticule=False,
              gridmode=False, hascolbar=True, pixelaspectratio=None,
              clipmode=0, clipmn=(4,5),
              callbackslist={}):
      #-------------------------------------------------------------------------
      """
      Add a new cube to the container. A cube can be a two dimensional data set.
      Then we imagine a third axis of length 1 without any meaning. But
      usually a cube is a set of two dimensional data structures. The axes of
      these 'images' are arbitrary. For instance, one can also display
      images with one spatial axis and one spectral axis (e.g. position
      velocity diagrams).

      :param frame:
          Each series of movie images that is appended needs its own frame.
      :type frame:
          Matplotlib Axes object.

      :param finished:
            An external method that is called after the images are loaded
      :type finished:
            A function that will be executed after the loading of a series of images
            has been completed (per cube)
      """
      #-------------------------------------------------------------------------
      C = Cubeatts(frame, fitsobj,  axnums, slicepos, pxlim, pylim, vmin, vmax,
                   xpanel, ypanel, hasgraticule, gridmode, hascolbar, pixelaspectratio,
                   clipmode, clipmn,
                   callbackslist)
      C.movieframeoffset = self.movieframeoffset       # First cube has offset 0
      C.cnr = self.cubeindx
      self.cubelist.append(C)
      self.numcubes += 1      
      # Prepare movie frame offset for the next cube.
      self.movieframeoffset += C.nummovieframes
      # We do not want the initial frame and default labeling on screen:
      frame.axis('off')
      # Before starting loading images in a timer callback, we disable
      # all registred callbacks in 'end_interactionMovieToPanels'.
      # In the loadimages() method, the callbacks are activated again
      if self.cubelist[0].panelscb is not None:
         self.end_interactionMovieToPanels()
      #if self.resizecb:
      #   self.resizecb.deschedule()
      self.cubeindx += 1                       # Prepare index for the next cube 
      # Start loading the images. We do this with a timer callback so that
      # we don't have to wait for the last image before a window appears.
      timer = TimeCallback(self.loadimages, 0.00001, False, count=0, cube=C)
      # The timer list keeps track of the timers. The first that is registered
      # is the first that will be processed entirely before the next callback is
      # scheduled.
      self.timerlist.append(timer)
      if len(self.timerlist) == 1:
         # There are no previous load sessions in progress. Start this one
         self.timerlist[0].schedule()
      C.previous_bbox = None
     
      return C
      

   def set_aspectratio(self, cube, aspectratio):
      #-------------------------------------------------------------------------
      """
      Change aspect ratio of frame of 'cube'
      """
      #-------------------------------------------------------------------------      
      if aspectratio is None or aspectratio == 0.0:
         aspectratio = cube.fitsobj.get_pixelaspectratio()  # Set default from header
      else:
         aspectratio = abs(aspectratio)
      cube.pixelaspectratio = abs(aspectratio)
      cube.frame.set_aspect(aspectratio)
      currentindx = self.movieimages.indx
      flushprint("MAPUTILS setimage in set_aspectratio")
      self.movieimages.setimage(currentindx, force_redraw=True)     
      
      

   def compose_movieframelist(self):
      #-------------------------------------------------------------------------
      """
      Each cube can have a series of images set as images for the movie loop.
      This method composes a list for all cubes.
      """
      #-------------------------------------------------------------------------
      newlist = []
      for C in self.cubelist:
         newlist += [j + C.movieframeoffset for j in C.movieframelist]
      self.movieimages.set_movieframes(newlist)


   def set_movieframes(self, framelist):
      #-------------------------------------------------------------------------
      """
      For INTERACTIVE use.
      Set the images that you want to include in a movieloop. The numbers are
      indices of the array with all the collected images in the moviecontainer.
      The index numbers are not related to a cube.
      """
      #-------------------------------------------------------------------------
      self.movieimages.set_movieframes(framelist)


   def set_panelframes(self, framelist, panel='X'):
      #-------------------------------------------------------------------------
      """
      Set a list with movie frame index numbers to show up in one of the
      side panels. The panels appear when the list is not empty.
      """
      #-------------------------------------------------------------------------
      self.numXpanels = self.numYpanels = 0
      for C in self.cubelist:
         frlo = C.movieframeoffset; frhi = C.movieframeoffset + C.nummovieframes
         panelframes = [d-frlo for d in framelist if frlo <= d < frhi]

         if panel.upper() == 'X':
            flushprint("SET_XPANELS.................")
            C.set_panelXframes(panelframes)
            #self.create_panel('x', C)
         else:
            flushprint("SET_YPANELS.................")
            C.set_panelYframes(panelframes)
            #self.create_panel('y', C)
         if C.panelXframes:     # Just set by set_panel(X)Yframes
            self.numXpanels += 1
         if C.panelYframes:     # Just set by set_panel(X)Yframes
            self.numYpanels += 1

      for C in self.cubelist:
         if panel.upper() == 'X':
            # Call method even when there are no panelXframes. Then we can clean up.
            self.create_panel('x', C)
         else:
            self.create_panel('y', C)
       
      
   def create_panel(self, panid, cube):
      #----------------------------------------------------------------------------
      """
      Plot (or clean up) one of the panels or both. This method is invoked either
      at the start of the program after loading all images, or when there are 
      triggers for the side panels.
      """
      #----------------------------------------------------------------------------
      panx = pany = False
      if panid == 'x':
         panx = True
      elif panid == 'y':
         pany = True
      # The request to update the 
      elif panid == 'r':   # A redo request # TODO: nog nodig??
         pany = True
         panx = True
      else:
         return
      flushprint("panx=%s pany=%s"%(str(panx), str(pany)))
      cubeindx = cube.cnr
      xold = cube.xold
      yold = cube.yold
      frame = cube.frame
      fig = frame.figure
      framep1 = cube.framep1
      framep2 = cube.framep2
      flushprint("Frames in create_panel: fr p1 p2=%d %d %d"%(id(frame), id(framep1), id(framep2)))
      cmap = cube.cmap
      mplimp1 = cube.annimp1
      mplimp2 = cube.annimp2
      slicepos = cube.slicepos
      fitsobject = cube.fitsobj
      pxlim = cube.pxlim
      pylim = cube.pylim
      vmin = cube.vmin
      vmax = cube.vmax
      offset = cube.movieframeoffset
      # The horizontal panel has an vertical range that should directly
      # convert mouse positions into array indices. I.e. if pylim is
      # 1,3, the range in the plot is -0.5, 2.5. Only the exact position
      # 2.5 will be rounded to 3, which is not a valid array index for this
      # range. Therefore we lower the upper value with a small number to
      # avoid this problem.
      delta = 0.000001

      #   !!!!!!!!!!!!!!!!!!!
      # Dit gaat alleen werken als van te voren bekend is wat de inhoud is van
      # panelXframes en panelYframes in de eventfrompanel routines.
      j = 0
      panelindx = j
      for C in self.cubelist:
         if C.cnr == cubeindx:
            panelindx = j
            flushprint("panelindx=%d, C.cnr, cubeindx=%d %d"%(panelindx,C.cnr, cubeindx ))
         if (C.panelXframes or C.panelYframes):
            j += 1

      cube.frame = frame         # Store it as attribute after it got new values

      # Important note: In this method we add slice panels. These get an initial
      # box with dummy sizes. That is because the actual drawing is postponed
      # and the reposition method (usually triggered after resize events)
      # calculated the correct values for all the panels.
      # Equal boxes in Matplotlib are considered to belong to the same
      # Axes object (frame). We force a unique frame by giving
      # the Axes contructor a unique label.
      panbox = [0,0,0.01,0.01]   # Dummy
      
      # If there is already a panel, then we need to remove its contents etc.
      if panx or pany:
         if panx and framep1:
            flushprint("Removing frame1")
         #if framep1:
            if mplimp1:
               mplimp1.disconnectCallbacks()
               mplimp1.regcb.deschedule()
               flushprint("Remove panel1 callback met id=%d"%(id(mplimp1.regcb)))
            framep1.clear()
            fig.delaxes(framep1)
            if mplimp1.grat:
               fig.delaxes(mplimp1.grat.frame)
               fig.delaxes(mplimp1.grat.frame2)
            framep1 = cube.framep1 = None
            mplimp1 = None
            #if framep2:
            #   pany = True
         if pany and framep2:
         #if framep2:
            flushprint("Removing frame2")
            if mplimp2:
               mplimp2.disconnectCallbacks()
               mplimp2.regcb.deschedule()
               flushprint("Remove panel2 callback met id=%d"%(id(mplimp2.regcb)))
            framep2.clear()
            fig.delaxes(framep2)
            if mplimp2.grat:
               fig.delaxes(mplimp2.grat.frame)
               fig.delaxes(mplimp2.grat.frame2)
            framep2 = cube.framep2 = None
            mplimp2 = None
            #if framep1:
            #   panx = True
      #if not (cube.panelXframes or cube.panelYframes):
      #   self.reposition(force=True)
      #   return              # Nothing to do
      

      xi = xold; yi = yold
      if None in [xi,yi]:  # If the first time is a resize event, then xold
         xi = pxlim[0]     # and yold are None.
         yi = pylim[0]


      # Create new frame. If a user did reset the panels with an empty list,
      # one must skip this part
      if panx and cube.panelXframes:
         label = 'x_'+str(id(cube))      # For a unique Axes object we need a unique label!!
         framep1 = fig.add_axes(panbox, sharex=frame, label=label)
         framep1.set_aspect('auto')
         framep1.set_adjustable('box-forced')
         framep1.axis('off')

         M = cube.Mx
         ly = M.shape[0]
         j = 0
         for indx in cube.panelXframes:
            # yi should be in range of pylim. To make it an array index,
            # subtract the limit pylim[0]
            I = self.movieimages.annimagelist[offset+indx]
            if I.origdata is not None:
               N = I.origdata
            else:
               N = I.data
            M[ly-j-1] = N[yi-pylim[0]]
            j += 1

         mixpix = pylim[0]
         fo = cube.fitsobj
         wcstypes = [fo.wcstypes[cube.panelx_axperm[0]-1],
                     fo.wcstypes[cube.panelx_axperm[1]-1]]
         #wcstypes = [fo.wcstypes[fo.axperm[0]-1], fo.wcstypes[fo.axperm[1]-1]]
         mplimp1 = Annotatedimage(framep1,
                                 header=fo.hdr,
                                 # Invert the limits in y
                                 # because the image is also build
                                 # with highest y below
                                 pxlim=pxlim, pylim=[ly-1-delta,0],   #[ly-1-delta,0],
                                 imdata=M,
                                 projection=cube.panelx_proj, #fo.convproj,
                                 axperm=cube.panelx_axperm, #fo.axperm
                                 wcstypes=wcstypes,
                                 skyout=fo.skyout,
                                 spectrans=fo.spectrans,
                                 alter=fo.alter,
                                 mixpix=fo.mixpix,
                                 aspect='auto',
                                 slicepos=mixpix, #fo.slicepos,
                                 sliceaxnames=fo.sliceaxnames,
                                 sourcename=fo.filename,
                                 cmap=cmap,
                                 adjustable='box-forced',
                                 clipmin=vmin, clipmax=vmax)

         #framep1.set_xlim(pxlim)
         #framep1.set_ylim([ly-1-delta,0])
         posobj1 = Positionmessage(cube.panelx_proj.skysys, fo.skyout, cube.panelx_proj.types)
         # Add a callback for this panel. A move with button 1 on, will change
         # the cube image to the one where the index corresponds to the (mouse) position
         mplimp1.regcb = AxesCallback(self.eventfrompanel1, framep1,
                                     'motion_notify_event',
                                      cubenr=cubeindx, posobj=posobj1)
         flushprint("Add motion notify for panel 1 callback %d for object %d"%(id(mplimp1.regcb), id(self)))
         mplimp1.AxesCallback_ids.append(mplimp1.regcb)
         #mplimp1.data = bo dat
         mplimp1.Image(animated=True)
         #mplimp1.Image()
         mplimp1.plot()
         mplimp1.grat = None
         cube.annimp1 = mplimp1
         cube.framep1 = framep1


      if pany and cube.panelYframes:
         label = 'y_'+str(id(cube))
         framep2 = fig.add_axes(panbox, sharey=frame, label=label)
         framep2.set_aspect('auto')
         framep2.set_adjustable('box-forced')
         framep2.axis('off')

         M = cube.My
         lx = M.shape[1]
         j = 0
         for indx in cube.panelYframes:
            I = self.movieimages.annimagelist[offset+indx]
            if I.origdata is not None:
               N = I.origdata
            else:
               N = I.data
            M[:,j] = N[:,xi-pxlim[0]]   # self.movieimages.annimagelist[offset+indx].data[:,xi-pxlim[0]]
            j += 1

         fo = cube.fitsobj
         mixpix = pxlim[0]
         wcstypes = [fo.wcstypes[cube.panely_axperm[0]-1],
                     fo.wcstypes[cube.panely_axperm[1]-1]]
         mplimp2 = Annotatedimage(framep2,
                                    header=fo.hdr,
                                    pxlim=[0,lx-1-delta], pylim=pylim,
                                    imdata=M,
                                    projection=cube.panely_proj,
                                    axperm=cube.panely_axperm, #fo.axperm,
                                    wcstypes=wcstypes,
                                    skyout=fo.skyout,
                                    spectrans=fo.spectrans,
                                    alter=fo.alter,
                                    mixpix=mixpix,
                                    aspect='auto',
                                    slicepos=fo.slicepos,
                                    sliceaxnames=fo.sliceaxnames,
                                    sourcename=fo.filename,
                                    cmap=cmap,
                                    adjustable='box-forced',
                                    clipmin=vmin, clipmax=vmax)

         #framep2.set_xlim([0,lx-1-delta])
         #framep2.set_ylim(pylim)
         posobj2 = Positionmessage(cube.panely_proj.skysys, fo.skyout, cube.panely_proj.types)
         mplimp2.regcb = AxesCallback(self.eventfrompanel2, framep2,
                                     'motion_notify_event',
                                      cubenr=cubeindx, posobj=posobj2)

         mplimp2.AxesCallback_ids.append(mplimp2.regcb)
         flushprint("Add motion notify for panel 1 callback %d for object %d"%(id(mplimp2.regcb), id(self)))
         #mplimp2.data = boxdat
         mplimp2.Image(animated=True)
         mplimp2.plot()
         """      grat = mplimp2.Graticule(offsety=True, skipx=True)
               grat.set_tickmode(mode="na")
               grat.setp_gratline(wcsaxis=1, visible=False)
               grat.setp_axislabel(plotaxis=("left"), visible=False)
               grat.setp_axislabel(plotaxis=("bottom"), visible=False)
               grat.setp_axislabel(plotaxis=("right"), visible=True)
               grat.setp_ticklabel(plotaxis=("right"), visible=True)
               grat.setp_ticklabel(plotaxis=("bottom"), visible=False)
               grat.setp_ticklabel(plotaxis=("left"), visible=False)
               grat.setp_tickmark(plotaxis=("bottom"), visible=False)
               grat.setp_tickmark(plotaxis=("left"), visible=False)
               grat.setp_tickmark(plotaxis=("right"), visible=True)
               mplimp2.plot()
               b = mplimp2.box
               xb = (b[0], b[1], b[1], b[0], b[0])
               yb = (b[2], b[2], b[3], b[3], b[2])
               grat.frame2.plot(xb, yb, lw=1, c='k', alpha=0.5)
               mplimp2.linepieces = set()
               mplimp2.linepieces.update(grat.frame2.findobj(Line2D))
               mplimp2.grat = grat
               # mplimp2.interact_toolbarinfo()
            else:
            """
         cube.annimp2 = mplimp2
         mplimp2.grat = None
         cube.framep2 = framep2

      flushprint("Frames net Na create_panel: fr p1 p2=%d %d %d"%(id(frame), id(cube.framep1), id(cube.framep2)))
      # Instead of a draw, we trigger a re-position so that all frames
      # will be adjusted to the new image frame. This saves an extra
      # call to canvas.draw().
      self.reposition(force=True)
         

   def updatemovieframe(self, cb, frompanel1=False, frompanel2=False):
      #-------------------------------------------------------------------------
      """
      This method changes an image in the movie container. It is called
      after a mouse move in one of the slice panels. That's why either
      'frompanel1' or 'frompanel2' must be true.
      But this mouse move can also imply a move in a (e.g. spatial) direction
      in a panel comparable to a move in the main window.
      This updates the other panel(s).
      """
      
      #-------------------------------------------------------------------------
      cbx = cb.xdata; cby = cb.ydata   # Temp. store current mouse position
      cubenr = cb.cubenr
      currentcube = self.cubelist[cubenr]
      if frompanel1:
         yi = int(numpy.round(cb.ydata))
         framenr = currentcube.panelXframes[yi] + currentcube.movieframeoffset
         if framenr >= 0 and framenr < self.movieimages.numimages and\
                             framenr != self.movieimages.indx:
            flushprint("MAPUTILS setimage in updatemovieframe")
            self.movieimages.setimage(framenr)

         # After we set the right movie image, we also have to update the
         # opposite panel, because, for example, a move to the right
         # in the x panel (along x axis) will change the slice along the y axis.
         self.updatepanels(None, frompanel1, frompanel2, cubenr=cubenr,
                           xpos=cb.xdata, ypos=cb.ydata)

      if frompanel2:
         xi = int(numpy.round(cb.xdata))
         framenr = currentcube.panelYframes[xi] + currentcube.movieframeoffset
         if framenr >= 0 and framenr < self.movieimages.numimages and\
                             framenr != self.movieimages.indx:
            flushprint("MAPUTILS setimage in updatemovieframe")
            self.movieimages.setimage(framenr)
         self.updatepanels(None, frompanel1, frompanel2, cubenr=cubenr,
                           xpos=cb.xdata, ypos=cb.ydata)         
         
      # Are there any other panels (from other cubes) to update?
      if frompanel1 or frompanel2:
         xd, yd = currentcube.frame.transData.transform((cbx,cby))
         for i in range(0, self.numcubes):
            if i != cubenr:
               # We need only an updated x or an updated y, not both
               # Method updatepanels() deals with this fact.
               xpos, ypos = self.cubelist[i].frame.transData.inverted().transform((xd,yd))
               cb.cubenr = i
               self.updatepanels(None, frompanel1, frompanel2, cubenr=i,
                                 xpos=xpos, ypos=ypos)
         cb.cubenr = cubenr



   def eventfrompanel1(self, cb):
      #------------------------------------------------------------------
      # Change the movie frame. The new frame number is
      # derived from the x position in the vertical panel (usually along
      # the latitude of the image). The function is registered as a
      # callback in the updatepanels() method.
      # The callback object 'cb' should have an attribute 'cubenr'
      # to identify the current cube to find the offset in the list
      # with movie frames.
      #------------------------------------------------------------------
      # Try to compose a message with position information.
      # We know the value of y. This is an index.
      yi = int(numpy.round(cb.ydata))

      # Each cube has a so called panelXframes and panelXframes list.
      # This is a list with all the
      # user supplied frame index numbers that are part of the movie. The default
      # is all movie frames in a cube, but a user could have changed this.
      # To find the index of the corresponding image we need to add the
      # offset of movieframes for the current cube.
      cubenr = cb.cubenr
      currentcube = self.cubelist[cubenr]
      if yi < 0 or yi >= len(currentcube.panelXframes):
         return
      framenr = currentcube.panelXframes[yi] + currentcube.movieframeoffset
      #print "yi, panelXframes[yi[, offset,framenr", yi, currentcube.panelXframes[yi], currentcube.movieframeoffset,framenr 
      currentim = self.movieimages.annimagelist[framenr]
      # The mouse is outside the current image. This is a trigger to
      # reset the last mouse position
      oim = self.movieimages.annimagelist[self.movieimages.indx]
      oim.X_lastvisited = oim.Y_lastvisited = None
      yinfo = currentim.slicemessage
      ypos = currentcube.pylim[0] # As a dummy
      xw, yw, missingspatial = currentim.toworld(cb.xdata, ypos, matchspatial=True)
      sl = cb.posobj.wcs2str(xw, yw, missingspatial, returnlist=True)
      s = sl[0] + ' ' + yinfo
      currentim.messenger(s)
      if cb.event.button == 1:      # Only action when the left m.button is pressed
         self.updatemovieframe(cb, frompanel1=True)


   def eventfrompanel2(self, cb):
      #------------------------------------------------------------------
      # Update the movie with a new frame. The new frame number is
      # derived from the y position in the horizontal panel (usually along
      # the longitude). The function is registered as a callback.
      # The callback object 'cb' should have an attribure 'cubenr'
      # to identify the current cube to find the offset in the list
      # with movie frames.
      #------------------------------------------------------------------
      xi = int(numpy.round(cb.xdata))
      # Each cube has a so called 'framesindxlist'. This is a list with all the
      # user supplied frame index numbers that are part of the movie. The default
      # is all movie frames in a cube, but a user could have changed this.
      # To find the index of the corresponding image we need to add the
      # offset of movieframes for the current cube.
      cubenr = cb.cubenr
      currentcube = self.cubelist[cubenr]
      # TODO: documenteer dit. Voorheen kon je in het witte gebied na pannen niet
      # navigeren want xi was buiten range.
      if xi < 0 or xi >= len(currentcube.panelYframes):
         return
      #framenr = currentcube.framesindxlist[xi] + currentcube.movieframeoffset
      framenr = currentcube.panelYframes[xi] + currentcube.movieframeoffset
      currentim = self.movieimages.annimagelist[framenr]
      # The mouse is outside the current image. This is a trigger to
      # reset the last mouse position
      oim = self.movieimages.annimagelist[self.movieimages.indx]
      oim.X_lastvisited = oim.Y_lastvisited = None
      xinfo = currentim.slicemessage
      xpos = currentcube.pxlim[0] # As a dummy
      xw, yw, missingspatial = currentim.toworld(xpos, cb.ydata, matchspatial=True)
      sl = cb.posobj.wcs2str(xw, yw, missingspatial, returnlist=True)
      s = sl[1] + ' ' + xinfo
      currentim.messenger(s)    
      if cb.event.button == 1:
         self.updatemovieframe(cb, frompanel2=True)



   def getpanelboxes(self, frame, numXpanels, numYpanels, Xpanelindx=None, Ypanelindx=None):
      #-----------------------------------------------------------------------------
      """
      This method calculates the positions of the side panels. It locates these
      panels to the bottom and right of an existing frame. The width of the gaps
      is fixed in pixels. The entire space between 
      """
      #-----------------------------------------------------------------------------
      
      # We want gaps between panels of a fixed number of pixels. So transform this
      # number to figure coordinates
      fig = frame.figure
      xf0, yf0 = fig.transFigure.inverted().transform((0, 0))
      xf1, yf1 = fig.transFigure.inverted().transform((4, 4))
      dx = xf1 - xf0
      dy = yf1 - yf0

      # Get the box in figure coordinates. It seems that we need to apply
      # the aspect ratio first.
      frame.apply_aspect()

      # Get the current box in figure coordinates
      bbox = frame.get_position()
      xlo, ylo, xhi, yhi = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

      # Get the boundaries of all frames and calculate the min, max
      # of those in figure coordinates           
      xmin, ymin, xmax, ymax = self.getbiggestbox()
      boxX = boxY = None
      
      if Xpanelindx is not None:
         # First box for x panel, i.e. connected to lower x axis
         H = ymin
         n = max(1.0,numXpanels)
         bxmin = xlo; bxmax = xhi
         h = (H-(n+1)*dy)/n
         if h <= 0.0:    # If there is no space, then create some. This gives overlap
            h = 0.1
         i = Xpanelindx
         bymin = ymin - (i+1)*(h+dy)
         bymax = bymin + h
         #flushprint("ymin=%f, bymin=%f, bymax=%f, h=%f, dy=%f n=%d"%(ymin,bymin, bymax, h, dy,n ))
         w = bxmax - bxmin
         boxX = (bxmin, bymin, w, h)

      if Ypanelindx is not None:
         # Second box for y panel, i.e. connected to right y axis
         W = 1.0 - xmax
         m = max(1.0,numYpanels)
         w = (W-(m+1)*dx)/m
         if w <= 0.0:
            w = 0.1
         i = Ypanelindx
         bxmin = xmax + dx + i*(w+dx)
         bxmax = bxmin + w
         #flushprint("xmax=%f, bxmin=%f, bxmax=%f, w=%f, dx=%f, HW=%f"%(xmax,bxmin, bxmax, w, dx, HW))
         bymin = ylo; bymax = yhi
         h = bymax - bymin
         boxY = (bxmin, bymin, w, h)

      flushprint("In getpanelboxes boxX, boxY=%s %s"%(str(boxX), str(boxY)))
      return boxX, boxY


   def set_crosshair(self, status):
   #----------------------------------------------------------------------------
      """
      This method
      """
   #----------------------------------------------------------------------------
      self.crosshair = status
      currentindx = self.movieimages.indx
      flushprint("MAPUTILS setimage in set_crosshair")
      self.movieimages.setimage(currentindx, force_redraw=True)


   def set_colbar_on(self, cube, mode):
    #----------------------------------------------------------------------------
      """
      This method
      """
   #----------------------------------------------------------------------------
      if not cube or not len(cube.imagesinthiscube):
         return                               # Nothing to do
      cube.hascolbar = mode
      currentindx = self.movieimages.indx
      for im in cube.imagesinthiscube:
         im.hascolbar = cube.hascolbar
      flushprint("MAPUTILS setimage in set_colbar_on")
      self.movieimages.setimage(currentindx, force_redraw=True)
       


   def new_colorbar(self, cube):
   #----------------------------------------------------------------------------
      """
      This method
      """
   #----------------------------------------------------------------------------
      label = 'CB_' + str(id(cube))    # We need a unique label
      cube.cbframe = self.fig.add_axes((0.0, 0.0,self.colbwidth,1.0),
                                        label=label,
                                        frameon=False,  # transparency needed
                                        autoscale_on=False)

      
      # We need one image from this cube as a dummy
      dummyim = cube.imagesinthiscube[0]      
      cube.colorbar = dummyim.Colorbar(frame=cube.cbframe, orientation='vertical')

      if cube.fitsobj.hdr.has_key('BUNIT'):
         dataunits = cube.fitsobj.hdr['BUNIT']
      else:
         dataunits = 'units unknown'

      cube.colorbar.plot(cube.cbframe, dummyim.image.im)  # Postponed draw
      #dummyim.cmap.add_frame(cube.cbframe)


      # We want the labels inside the colorbar. We need also a transparent background
      # for each label so that the labels are visible under all circumstances.
      bbox=dict(edgecolor='w', facecolor='w', alpha=0.4, boxstyle="square,pad=0.1")
      #cube.cbframe.yaxis.set_major_formatter(y_formatter)
      for tick in cube.cbframe.yaxis.get_major_ticks():
         tick.set_pad(-5.0) # Matplotlib 1.1 does it correctly. Matplotlib 1.01 not
         #tick.tick2line.set_markersize(6)
         tick.tick2line.set_color('w')
         tick.tick2line.set_markeredgewidth(2)         
         tick.label2.set_horizontalalignment('right')
         tick.label2.set_bbox(bbox)
         #flushprint("label=%s"%(tick.label2))
         

      for t in cube.cbframe.get_yticklabels():  # Must be 'plotted' before we can change properties
         t.set_fontsize(8)
         t.set_rotation(90)

      cube.cbframe.text(0.2, 0.98, dataunits,
                        fontsize=7, fontweight='bold', color='w', rotation=90,
                        horizontalalignment='left', verticalalignment='top',
                        bbox=dict(color='w', facecolor='red', alpha=0.5))

      # Copy pointer to all images in this cube
      for im in cube.imagesinthiscube:
         im.cbframe = cube.cbframe
         im.hascolbar = cube.hascolbar


   def set_skyout(self, cube, skyout):
   #----------------------------------------------------------------------------
      """ This method
      """
   #----------------------------------------------------------------------------    
      if not cube or not len(cube.imagesinthiscube):
         return
      for im in cube.imagesinthiscube:
         im.skyout = im.projection.skyout = skyout
         flushprint("Set skyout to %s"%(str(skyout)))
      if cube.hasgraticule:
         self.new_graticule(cube, visible=True)

      currentindx = self.movieimages.indx
      # Reset current image to change toolbarinfo and possibly graticule
      flushprint("MAPUTILS setimage in set_skyout")
      self.movieimages.setimage(currentindx, force_redraw=True) 



   def set_graticule_on(self, cube, mode):
    #----------------------------------------------------------------------------
      """
      This method 
      """
   #----------------------------------------------------------------------------   
      if not cube or not len(cube.imagesinthiscube):
         return
      cube.hasgraticule = mode
      currentindx = self.movieimages.indx
      if not mode:
         # No graticule should be visible
         if cube.grat:
            # There exists a graticule. Empty contents, leave frames unaltered
            for li in cube.linepieces:
               li.set_visible(False)
            for tl in cube.textlabels:
               tl.set_visible(False)
            cube.grat.frame.lines = []
            cube.grat.frame2.lines = []
            cube.grat.frame.texts = []
            cube.grat.frame2.texts = []                     
            cube.linepieces = []
            cube.textlabels  = []
            for im in cube.imagesinthiscube:
               im.grat = cube.grat
               im.linepieces = cube.linepieces
               im.textlabels = cube.textlabels            
      else:
         self.new_graticule(cube, visible=True)
      flushprint("MAPUTILS setimage in set_graticuleon")
      self.movieimages.setimage(currentindx, force_redraw=True)
         



   def set_graticules(self):
   #----------------------------------------------------------------------------
      """
      This method should be called after zoom/pan actions where we want to
      update the graticule on an image. Then one should update this for
      all cubes because a zoom is a zoom in all images in the movie
      container because each frame is zoomed to the same area on display.
      This is a feature of Matplotlib.
      """
   #----------------------------------------------------------------------------
      
      if not len(self.movieimages.annimagelist):
         return     # No images yet, nothing to do

      currentindx = self.movieimages.indx
      currentimage = self.movieimages.annimagelist[currentindx]
      cnr = currentimage.cubenr
      cube = self.cubelist[cnr]

      for cnr, cube in enumerate(self.cubelist):
         if cnr == currentimage.cubenr:
            visible = True
         else:
            visible = False
         self.new_graticule(cube, visible)
         
      #self.movieimages.setimage(currentindx)
      # This method is always triggered in combination with a reposition event
      # The reposition() method does the re-draw of the canvas.
      
      

   def new_graticule(self, cube, visible):
   #----------------------------------------------------------------------------
      """
      Note:
      For images with one spatial axis and one axis that is not spatial (
      for instance spectral or Stokes axis), the spatial
      axis is labeled with offsets. These offsets are a function of the missing
      spatial axis. This implies that in fact, each image in a movie of the same
      cube, could have a different labeling. We chose the alternative, which is
      that for each image the offsets are calculated at the same position on
      the missing axis. This position is the associated value of CRPIX.
      """
   #----------------------------------------------------------------------------
      """
      if not len(self.movieimages.annimagelist):
         return
      currentindx = self.movieimages.indx
      currentimage = self.movieimages.annimagelist[currentindx]
      cnr = currentimage.cubenr      
      cube = self.cubelist[cnr]
      """
      if not cube.hasgraticule:
         return

      if not len(cube.imagesinthiscube):
         return
      else:
         # We need just one image to attach the graticule to
         #currentindx = self.movieimages.indx
         #currentimage = self.movieimages.annimagelist[currentindx]
         currentimage = cube.imagesinthiscube[0]
      flushprint("ID cube, visible, id currentim= %d %s %d"%(id(cube), visible, id(currentimage)))
      # This seems the only way to clean up the line Line2D and Text objects
      # that belong to these frames (Axes objects).
      # However, the lines and text lists do not store all the relevant lines
      # and text objects (should it?)
      if cube.grat:      # Then reset what is necessary
         cube.grat.frame.lines = []
         cube.grat.frame2.lines = []
         cube.grat.frame.texts = []
         cube.grat.frame2.texts = []
         cube.frame.texts = []

         # Store the Axes objects associated with the graticule before
         # creating a new one. We need these frames to tell the Graticule's plot
         # method that it can reuse them.
         frame1 = cube.grat.frame
         frame2 = cube.grat.frame2
      else:
         frame1 = frame2 = None
      # If we arrive in this method, then our imagenew or is just interactively
      # zoomed in our out. This implies that its graticule system is changed.
      # We want the graticule also to be changed using the new limits of
      # the image in pixels. If we use the get_x/ylim methods then the
      # pixel boundaries are usually not integer. In module wcsgrat, we
      # Add 0.5 to the pixel positions to extend the boundary pixels so
      # that an image shows the entire boundary pixel.
      # But in this situation we don't want this, because the zoom action
      # determined exactly what the boundary is.
      pxlim = cube.frame.get_xlim(); pxlim = (pxlim[0]+0.5, pxlim[1]-0.5)
      pylim = cube.frame.get_ylim(); pylim = (pylim[0]+0.5, pylim[1]-0.5)
      flushprint("px/ylim na zoom: %s %s"%(str(pxlim), str(pylim)))

      # The setup for the graticule
      cube.grat = currentimage.Graticule(pxlim=pxlim, pylim=pylim)
      #cube.grat.setp_ticklabel(wcsaxis=0, rotation=0)
      cube.grat.setp_ticklabel(wcsaxis=1, rotation=90)  # Y axis labels along the axis not perpendicular
      
      #cube.grat.setp_ticklabel(wcsaxis=(0,1), color='r')
      #cube.grat.setp_ticklabel(wcsaxis=(0,1), visible=False)
      cube.grat.setp_tick(wcsaxis=(0,1), visible=False)
      cube.grat.set_tickmode(plotaxis=("left","bottom"), mode="NO_TICKS")
      #cube.grat.setp_axislabel(plotaxis=("left","bottom"), color='g', fontsize=10, fontweight='normal')
      cube.grat.setp_axislabel(plotaxis="left", visible=False)
      cube.grat.setp_axislabel(plotaxis="bottom",  visible=False)
      cube.grat.setp_gratline(wcsaxis=(0,1), lw=0.5, color='k')
      bbox=dict(edgecolor='w', facecolor='w', alpha=0.4, boxstyle="square,pad=0.")
      cube.insideX = cube.grat.Insidelabels(wcsaxis=0, fontsize=10, ha='right', va='bottom',
                             rotation_mode='anchor', bbox=bbox)
      bbox=dict(edgecolor='w', facecolor='w', alpha=0.4, boxstyle="square,pad=0.")
      cube.insideY = cube.grat.Insidelabels(wcsaxis=1, fontsize=10,
                             ha='right', va='bottom',
                             rotation_mode='anchor', bbox=bbox)
      cube.grat.plot(cube.frame, frame1, frame2)
      # In the documentation of wcsgrat we read that insidelabels are plotted on frame2
      #cube.insideX.plot(cube.frame)
      #cube.insideY.plot(cube.frame)
      
      flushprint("cube.frame position=%s"%(str(cube.frame.get_position())))
      flushprint("cube.grat.frame position=%s"%(str(cube.grat.frame.get_position())))
      #cube.grat.frame.set_position(cube.frame.get_position())
      #cube.grat.frame2.set_position(cube.frame.get_position())
      
      # Default, ticks are copied to the top and right axis. However we did
      # set those tick lines (Line2D objects) to invisible. We don't want
      # to include thse in the list of objects that need to be restored
      # after we change an image.
      cube.linepieces = set(l for l in cube.grat.frame.findobj(Line2D) if l.get_visible())
      cube.textlabels = set(t for t in cube.grat.frame2.findobj(Text) if (t.get_text() != '' and t.get_visible()))

      for tl in cube.textlabels:
         flushprint("         set_graticule: textlabel %d %s op True"%(id(tl), str(tl)))


      # Only the current image should show its graticule      
      if not visible:
         for l in cube.linepieces:
            l.set_visible(False)
         for t in cube.textlabels:
            t.set_visible(False)
      
      # A new graticule has been made, so the images should get some
      # knowledge of that. They are all updated with new line pieces and text
      # labels (i.e. they point to the same object). We need them if we
      # change images and want to keep the graticule overlay visible.
      for im in cube.imagesinthiscube:
         im.grat = cube.grat
         im.linepieces = cube.linepieces
         im.textlabels = cube.textlabels
         flushprint("cube.textlabels = %d"%(len(im.textlabels)))


   def set_coordinate_mode(self, grids=True, world=False, worlduf=False, imval=False,
                           pixfmt="%.1f", dmsprec=1, wcsfmt="%.7f", zfmt='%.3e',
                           appendcr=False, tocli=False, tolog=True, resetim=True):
   #----------------------------------------------------------------------------
      """
      Set modes for the formatting of coordinate information
      """
   #----------------------------------------------------------------------------
      flushprint("len container=%d"%(len(self.movieimages.annimagelist)))
      for im in self.movieimages.annimagelist:
         im.coord_grids = grids
         im.coord_world = world
         im.coord_worlduf = worlduf
         im.coord_imval = imval
         im.coord_appendcr = appendcr
         im.coord_tocli = tocli
         im.coord_tolog = tolog
         im.coord_pixfmt = pixfmt
         im.coord_wcsfmt = wcsfmt
         im.coord_zfmt = zfmt
         im.coord_dmsprec = dmsprec
      currentindx = self.movieimages.indx
      #currentimage = self.movieimages.annimagelist[currentindx]
      if resetim:
         flushprint("MAPUTILS setimage in set_coordinate_mode")
         self.movieimages.setimage(currentindx, force_redraw=True)
         


   def reposition(self, cb=None, force=False):
      #------------------------------------------------------------------
      # After zoom/pan/resize events a draw_event is generated.
      # We use this event to re-position the panels.
      #------------------------------------------------------------------
      action = True #(self.numXpanels or self.numYpanels) or force
      flushprint("\nThis is a RESIZE event. My action is %s !!!!!!!!!!!!!\n"%(str(action)))

      for cube in self.cubelist:
         if cube.scalefac:
            # If a scale factor is set, then re calculate the frame borders first
            self.scaleframe(cube, cube.scalefac, reposition=False)
         else:
            self.scaleframe(cube, cube.scalefac, tofit=True, reposition=False)
  
      if (self.numXpanels or self.numYpanels):
         pnrX = pnrY = 0
         for cube in self.cubelist:       
            frame = cube.frame
            framep1 = cube.framep1
            framep2 = cube.framep2
            if framep1:
               flushprint("Voor: cube=%s framep1=%s %s"%(str(cube), str(id(framep1)), str(framep1)))
               # The importance of this method is that it re-calculates all
               # slice panel frames after a resize (usually triggered by a
               # canvas.draw() call.)
               boxX, boxY = self.getpanelboxes(frame, self.numXpanels, self.numYpanels, pnrX, None)
               framep1.set_position(boxX)
               framep1.set_aspect('auto')
               pnrX += 1
               cube.framep1 = framep1
               flushprint("Na: framep1=%s"%(str(framep1)))
            if framep2:
               flushprint("VOOR: cube=%s framep2=%s %s"%(str(cube),  str(id(framep2)), str(framep2)))
               boxX, boxY = self.getpanelboxes(frame, self.numXpanels, self.numYpanels, None, pnrY)
               framep2.set_position(boxY)
               framep2.set_aspect('auto')
               pnrY += 1
               cube.framep2 = framep2
               flushprint("NA: framep2=%s"%(str(framep2)))

      
      for cube in self.cubelist:
         if cube.hasgraticule:
            if hasattr(cube, 'grat'):
               action = True
               #cube.frame.apply_aspect()
               gratframe = cube.grat.frame
               #gratframe.apply_aspect()
               gratframe.set_position(cube.frame.get_position())
               #gratframe.set_aspect(cube.frame.get_aspect())
               #flushprint("cube.frame position=%s"%(str(cube.frame.get_position())))
               #flushprint("cube.grat.frame position=%s"%(str(cube.grat.frame.get_position())))
               #gratframe.apply_aspect()
               gratframe = cube.grat.frame2
               #gratframe.set_aspect(cube.frame.get_aspect())
               #gratframe.apply_aspect()
               gratframe.set_position(cube.frame.get_position())
               
               #gratframe.set_aspect(cube.frame.get_aspect())

      """
      for cube in self.cubelist:
         if cube.cbframe:
            flushprint("Ireset the colorbar position")
            cube.cbframe.set_position([0.,0.,self.colbwidth,1.0])
            cube.cbframe.set_autoscale_on(False)
            #cube.colorbar.cb.set_mappable(cube.imagesinthiscube[0])
         # TODO: Er gaat wat mis als je voor het eerst de blankkleur wijzigt.
         # Dan wordt de image overschreven mat data van het nulde image.
         """

      #self.set_graticule() 
      if action:
         #currentindx = self.movieimages.indx
         #self.movieimages.setimage(currentindx, force_redraw=True)
         # TODO: Als je deze redraw weglaat, is interactie sneller maar je
         # komt (net) niet in de goede eindtoestand terecht met je graticulen
         # Ik denk dat het voor de panels nog erger is.
         self.fig.canvas.draw()

      """
      x0, x1 = cube.frame.get_xlim()
      y0, y1 = cube.frame.get_ylim()
      xd0, yd0 = cube.frame.transData.transform((x0,y0))
      xd1, yd1 = cube.frame.transData.transform((x1,y1))
      #X = cube.frame.get_position().get_points()
      #flushprint("Coords na resize: %s"%(str(X)))
      flushprint("Display Coords na resize: %f %f %f %f"%(xd0, yd0, xd1, yd1))
      sx = xd1 - xd0; sy = yd1 - yd0
      flushprint("new Frame = %f x %f"%(sx, sy))
      #xd, yd = self.frame.transData.transform((x,y))
      #x2, y2 = self.frame.transData.inverted().transform((xd,yd))
      #flushprint("Display coordinates data=%f %f display=%f %f inverse=%f %f"%(x, y, xd, yd, x2, y2))
      """

   def redopanels(self):
      #------------------------------------------------------------------
      """
      Draw the slice panels for the first time or update them
      (the create_panel() method
      checks whether something needs to be plotted). The check is done for
      every cube in the cube list.
      """
      #------------------------------------------------------------------
      return
      for cnr in range(0, self.numcubes):
         self.create_panel('r', self.cubelist[cnr])


   def set_splitimagenr(self, nr):
      #-------------------------------------------------------------------------
      """
      Set image (by its number) which is used to split the current
      """
      #-------------------------------------------------------------------------
      num = len(self.movieimages.annimagelist)
      if nr < 0 or nr >= len:
         raise ValueError, "This image does not exist"
      else:
         # First reset a possible earlier image that was used to compare
         # to the current.
         if self.movieimages.compareim:
            currentimage = self.movieimages.annimagelist[self.movieimages.indx]
            currentimage.image.im.set_alpha(1.0)
            self.movieimages.compareim.image.im.set_alpha(1.0)
            self.movieimages.compareim.image.im.set_visible(False)
            self.movieimages.compareim = None
         self.splitmovieframe = nr


   def show_transparent(self, cube, alpha):
      """-----------------------------------------------------------------------
      Make current image transparent with alpha factor
      -----------------------------------------------------------------------"""
      splitmovieframe = self.splitmovieframe                 # A Cubes attribute
      # If the current image is the same as the second image then do nothing.
      currentindx = self.movieimages.indx
      if splitmovieframe == currentindx:
         return

      currentimage = self.movieimages.annimagelist[currentindx]
      # The current image is already visible. But we need to make
      # the second image visible too.
      secondimage = self.movieimages.annimagelist[splitmovieframe]
      secondimage.image.im.set_visible(True)

      if splitmovieframe > currentindx:
         # The second image is on top (because it was loaded later and the
         # zorder is derived from the loading order). So make this image
         # transparent. But if the alpha is close to 1, we want the current
         # image to be visible, so the second image should be very transparent
         # then. This behaviour can be obtained by setting the transparency
         # to 1-alpha
         secondimage.image.im.set_alpha(1.0-alpha)
         currentimage.image.im.set_alpha(1.0)
      else:
         currentimage.image.im.set_alpha(alpha)
         secondimage.image.im.set_alpha(1.0)

      self.fig.canvas.draw()
      # Note that we need some reset actions. For example, we don't want
      # to keep the second image visible if we change images or select
      # another image as compare image.
      # But if we do this at this point, then it will not be possible to
      # save a plot to disk because it does a re-draw and sees different
      # settings then. So resetting must be done elsewhere e.g. in
      # toggle_images(). Then store the images used to compare.
      self.movieimages.compareim = secondimage


      

   def splitscreen(self, cb):
      """-----------------------------------------------------------------------
      With Shift and the mouse buttons 1, 2 or 3 while moving the mouse,
      one can split the screen to make another image visible.
      It makes the upper (=current image) transparent and shows the image
      which has index 'splitmovieframe'. 
      -----------------------------------------------------------------------"""
      if cb is None or cb.event is None:
         return
      if cb.event.button is None or cb.event.key is None:
         return
      if cb.event.key != 'control':
         return
      # Which mousebutton is pressed (together with key 'shift')?      
      mb = cb.event.button
      if not mb in [1,2,3]:       # Sometimes mb = 'up' which we don't want here
         return
      # Which movie are we in?
      try:
         currentindx = self.movieimages.indx
      except:
         return
      # What is the movie frame number with which we want to split the screen?
      splitmovieframe = self.splitmovieframe                 # A Cubes attribute
      # If the current image is the same as the splitimage (i.e. the
      # image below the current image) then do nothing.
      if splitmovieframe == currentindx:
         return
      cube = self.cubelist[cb.cubenr]
      # We need a copy of the original data and make part of it transparent
      currentimage = self.movieimages.annimagelist[currentindx]
      if currentimage.cubenr != cb.cubenr:
         # This came from an event in another frame than the frame
         # of the current image
         return
      splitimage = self.movieimages.annimagelist[splitmovieframe]
      if currentimage.origdata is None:
         # Make copy first and reset variables
         cube.origdata = currentimage.data.copy()
         currentimage.origdata = cube.origdata
         currentimage.splitmovieframe = splitmovieframe
         currentimage.split_xold = currentimage.split_yold = 0
         currentimage.split_mb = mb
      # Get the mouse position in integer pixel coordinates and
      # convert to array indices
      flushprint("px/pylim=%s %s"%(str(cube.pxlim), str(cube.pylim)))
      # If an image has a box that is different compared to the default,
      # we have to correct for the limits to have the array indices started at
      # 0. For a default box, the indices start at 1 (first pixel).
      # For another box, this number is the pixel number of the first
      # pixel in the box.  and is always > 1.
      # Note that we change index at the center of a pixel (which explains the
      # addition 0f 0.5)
      xi = nint(cb.xdata+0.5) - cube.pxlim[0]
      yi = nint(cb.ydata+0.5) - cube.pylim[0]
      xo = currentimage.split_xold; yo = currentimage.split_yold
      # If the mouse button has switched, then restore entire image first
      if mb != currentimage.split_mb:
         currentimage.data[:] = cube.origdata[:]
         currentimage.split_xold = currentimage.split_yold = 0
      if mb == 1:                        # Left mouse button & 's' is split in y
         if yi > currentimage.split_yold:
            currentimage.data[0:yi] = numpy.nan
         elif yi < currentimage.split_yold:
            currentimage.data[yi:yo] = cube.origdata[yi:yo]
      elif mb == 2:                    # Middle mouse button & 's' is split in x
         if xi > currentimage.split_xold:
            currentimage.data[:,0:xi] = numpy.nan
         elif xi < currentimage.split_xold:
            currentimage.data[:, xi:xo] = cube.origdata[:, xi:xo]
      elif mb == 3:                 # Right mouse button & 's' is split in x & y
         if xi > currentimage.split_xold:
            currentimage.data[0:yo:,0:xi] = numpy.nan
         elif xi < xo:
            currentimage.data[0:yo:, xi:xo] = cube.origdata[0:yo, xi:xo]
         if yi > yo:
            currentimage.data[yo:yi, 0:xi] = numpy.nan
         elif yi < yo:
            currentimage.data[yi:yo, 0:xi] = cube.origdata[yi:yo, 0:xi]
      currentimage.split_xold = xi
      currentimage.split_yold = yi
      currentimage.split_mb = mb
      splitimage.image.im.set_visible(True)
      #for v in splitimage.data:
      #   if numpy.isnan(v).any():
      #      print v, " is NaN"
      currentimage.image.im.set_visible(True)
      
      # This is the trick. We set all the new NaN numbers to transparent.
      # But first we need to store the current colormap settings
      currentimage.old_blankcolor = currentimage.cmap.bad_val
      currentimage.set_blankcolor('w', alpha=0.0)        # Transparent
      # The blank color is reset to its old value if we change images
      # in toggle_images()
      #currentimage.image.im.changed()
     
      currentimage.frame.draw_artist(currentimage.image.im)
      if self.imageinfo:
         currentimage.frame.draw_artist(currentimage.info)
      for li in currentimage.linepieces:
             #li.set_visible(True)
         currentimage.frame.draw_artist(li)
      for tl in currentimage.textlabels:
             #li.set_visible(True)
         currentimage.frame.draw_artist(tl)

      
      #currentimage.set_blankcolor('w', alpha=0.0)        # Transparent
      #currentimage.frame.figure.canvas.blit(currentimage.frame.bbox)



   def updatepanels(self, cb, frompanel1=False, frompanel2=False,
                    cubenr=None, xpos=None, ypos=None):
      #------------------------------------------------------------------
      # One moved the mouse while pressing button 1 in the main window.
      # Then one extracts slices at different positions in the data cube.
      # This is reflected in an update of the data in the corresponding
      # panels.
      # Note that if you stacked more than
      # one cube, you will have more than one Axescallback. Each one
      # will update its corresponding slice panels. This behaviour is
      # copied from GIPSY's GIDS program.
      #
      # This method expects a callback object 'cb' which should have an
      # attribute 'cubenr' so that we can know which cube gets updated
      # slice panels.
      # If the callback cb is None, then we expect the cube number and
      # the mouse position for the cube that corresponds to the cube number
      #------------------------------------------------------------------
      skipupdatep1 = skipupdatep2 = False
      if frompanel1:
         # Event came from panel 1 so do not update panel 1
         skipupdatep1 = True
      if frompanel2:
         skipupdatep2 = True

      if cb is None:
         x = xpos; y = ypos
         cubeindx = cubenr
      else:
         # Left mouse button must be pressed!
         if not hasattr(cb, 'cubenr') or cb.event.button != 1:
            return
         x = cb.xdata; y = cb.ydata
         cubeindx = cb.cubenr
      
      cube = self.cubelist[cubeindx]
      canvas = cube.frame.figure.canvas
      
      xold = cube.xold
      yold = cube.yold
      #frame = cube.frame
      #fig = frame.figure
      framep1 = cube.framep1
      framep2 = cube.framep2
      mplimp1 = cube.annimp1
      mplimp2 = cube.annimp2
      pxlim = cube.pxlim
      pylim = cube.pylim
      offset = cube.movieframeoffset
      
      # The limits of the movie frame axes are pxlim, pylim.
      # We are interested in the pixel position and this is stored by
      # the callback object 'cb'.
      xi = int(numpy.round(x))
      yi = int(numpy.round(y))
      if skipupdatep1 and xi == xold:
         # Skip this position
         return
      if skipupdatep2 and yi == yold:
         return
      # Check the limits
      if skipupdatep1 and (xi > pxlim[1] or xi < pxlim[0]):
         return
      elif skipupdatep2 and (yi > pylim[1] or yi < pylim[0]):
         return
      if not (skipupdatep1 or skipupdatep2):
         if xi > pxlim[1] or xi < pxlim[0] or yi > pylim[1] or yi < pylim[0]:
            return

      # EXPERIMENTEEL!  Implementatie van crosshair cursor
      """
      # Draw a crosshair
      if self.crosshair:
         if not cube.background:
            cube.background = self.fig.canvas.copy_from_bbox(cube.frame.bbox)
         #cube.linev.set_visible(False)
         #cube.lineh.set_visible(False)
         #if cube.needclear:
         #      self.fig.canvas.draw()
         #      cube.needclear = False
         #      flushprint("Crosshair clean  at %f %f"%(x, y))
         #else:
         if 1:
         #      cube.needclear = True
               flushprint("Crosshair set at %f %f"%(x, y))
               cube.linev.set_xdata((x,x))
               cube.lineh.set_ydata((y,y))
               cube.linev.set_visible(True)   #self.visible and self.vertOn)
               cube.lineh.set_visible(True)   #self.visible and self.horizOn)

               self.fig.canvas.restore_region(cube.background)
               cube.frame.draw_artist(cube.linev)
               cube.frame.draw_artist(cube.lineh)
               self.fig.canvas.blit(cube.frame.bbox)
      else:
         cube.linev.set_visible(False)
         cube.lineh.set_visible(False)
      """
            

      if cube.panelXframes and yi != yold and not skipupdatep1:
         M = cube.Mx
         ly = M.shape[0]
         j = 0
         for indx in cube.panelXframes:
            # yi should be in range of pylim. To make it an array index,
            # subtract the limit pylim[0]
            I = self.movieimages.annimagelist[offset+indx]
            if I.origdata is not None:
               N = I.origdata
            else:
               N = I.data
            M[ly-j-1] = N[yi-pylim[0]]
            j += 1

         mplimp1.image.im.set_data(M)
         #mplimp1.image.im.changed()         
         framep1.draw_artist(mplimp1.image.im)
         #for li in mplimp1.linepieces:
         #   framep1.draw_artist(li)
         framep1.figure.canvas.blit(framep1.bbox)


      # Second frame which shares an y axis
      if cube.panelYframes and xi != xold and not skipupdatep2:
         M = cube.My
         lx = M.shape[1]
         j = 0
         for indx in cube.panelYframes:
            I = self.movieimages.annimagelist[offset+indx]
            if I.origdata is not None:
               N = I.origdata
            else:
               N = I.data
            M[:,j] = N[:,xi-pxlim[0]]   # self.movieimages.annimagelist[offset+indx].data[:,xi-pxlim[0]]
            j += 1

         mplimp2.image.im.set_data(M)
         #mplimp2.image.im.changed()
         framep2.draw_artist(mplimp2.image.im)
         #for li in mplimp2.linepieces:
         #   framep2.draw_artist(li)
         framep2.figure.canvas.blit(framep2.bbox)
      #fig.canvas.draw()
      if skipupdatep1:   # Event from panel2: Only x has been changed
         cube.xold = xi
      elif skipupdatep2: # Event from panel1: Only y has been changed
         cube.yold = yi
      else:              # Event in main window. Both x and y are changed
         cube.xold = xi
         cube.yold = yi




   def getbiggestbox(self):
      #---------------------------------------------------------------
      """
      This method finds (e.g. after a resize action) the biggest frame
      in the list of cubes. That frame sets the positions of the
      slice panels (usually the position velocity slices).
      """
      #---------------------------------------------------------------
      self.biggestbox = None
      for c in self.cubelist:               # Get the position of all the frames
         # With this if statement we assume that the x&Y panelframes are known for each cube
         if c.panelXframes or c.panelYframes:
            c.frame.apply_aspect()
            # TODO: alweer een frame_aspect? Nodig of niet?
            bbox = c.frame.get_position()
            # The bounds attribute represents x0,y0,w,h, but we want
            # x0,y0,x1,y1:
            box = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            if self.biggestbox is None:
               self.biggestbox = box
            else:
               if box[0] < self.biggestbox[0]:
                  self.biggestbox[0] = box[0]
               if box[1] < self.biggestbox[1]:
                  self.biggestbox[1] = box[1]
               if box[2] > self.biggestbox[2]:
                  self.biggestbox[2] = box[2]
               if box[3] > self.biggestbox[3]:
                  self.biggestbox[3] = box[3]

      return self.biggestbox


   def clearimage(self, im):
      for li in im.linepieces:
            li.set_visible(False)
      for tl in im.textlabels:
            tl.set_visible(False)
      if im.infoobj:
         im.infoobj.set_visible(False)
      im.image.im.set_visible(False)
      if im.cbframe is not None:
         im.cbframe.set_visible(False)


   def scaleframe(self, cube, scalefac, tofit=False, reposition=True):
   #----------------------------------------------------------------------------
      """
      Scale the current frame
      """
   #----------------------------------------------------------------------------
      if tofit:
         if cube.hascolbar:
            left = self.colbwidth
         else:
            left = 0.0
         if self.numXpanels or self.numYpanels:
            fig = cube.frame.figure
            figaspect = fig.get_figwidth()/fig.get_figheight()
            H_vert = 0.2
            H_horz = H_vert/figaspect
            right = 1.0-0.2/figaspect
            bottom = 0.0+H_vert
            top=1.0
         else:
            right = 1.0
            bottom = 0.0
            top = 1.0
         cube.scalefac = None
      else:
         cube.scalefac = scalefac
         x0, x1 = cube.pxlim
         y0, y1 = cube.pylim
         lx = x1 - x0 + 1; ly = y1 - y0 + 1
         flushprint("pxylim x0, y0=%f %f"%(x0, y0))
         flushprint("pxylim x1, y1=%f %f"%(x1, y1))
         flushprint("scalefac = %s"%(str(scalefac)))
         flushprint("in data lx, ly=%f %f"%(lx, ly))

         fig = cube.frame.figure
         xf0, yf0 = fig.transFigure.inverted().transform((0, 0))
         xf1, yf1 = fig.transFigure.inverted().transform((lx*scalefac, ly*scalefac))
         xf = xf1 - xf0; yf = yf1 - yf0
         # Next coordinates are all in figure units
         if cube.hascolbar:
            xc = 0.5 + self.colbwidth/2.0
         else:
            xc = 0.5
         left = xc - xf/2.0; right = xc + xf/2.0
         bottom = 0.5 - yf/2.0; top = 0.5 + yf/2.0
         if cube.hascolbar:
            dx = self.colbwidth - left
            if dx > 0.0:
               left  += dx
               right += dx

      # Note that we cannot use subplots_adjust, because the zooming
      # factor applies to individual cubes and we don't want to adjust
      # all the cubes at the same time.
      cube.frame.set_position([left, bottom, right-left, top-bottom])
      # The call to this method could come from method reposition()
      # If not so, you need to request re-positioning explicitely.
      if reposition:
         self.reposition(force=True)

            

   def loadimages(self, cb):
      #----------------------------------------------------------------------------
      """
      Loading image(s) into a movie cube is done with a timer callback. This
      callback makes it possible to show the images that you load immediately,
      which is more user friendly then when you have to wait until all images
      are loaded to see something happening on screen.
      Also it makes the gui more responsive.
      """
      #----------------------------------------------------------------------------      
      cube = cb.cube
      cnr = cube.cnr
      fig = cube.frame.figure

      oldim = self.lastimage  # We need to remove the last displayed image
      
      self.movieimages.movie_events(allow=False)
      # Re-position frame to fill the available space. Do not use method
      # frame.set_position() because that will not update the subplot
      # configuration values like 'left', 'right', etc.
      
      i = cb.count
      if i == 0:
         if self.progressbar:
            self.progressbar.setMinimum(0)
            self.progressbar.setMaximum(cube.nummovieframes)
         cube.callback('waitcursor')
         # Remove image and other attributes
         if oldim:            
            self.clearimage(oldim)
         # We want image to fill as much space as possible. Set this only
         # once (i==0, cnr==0)
         if cnr == 0:
            # Re-position frame to fill the available space. Do not use method
            # frame.set_position() because that will not update the subplot
            # configuration values like 'left', 'right', etc.
            fig.subplots_adjust(left=self.colbwidth, right=1.0, bottom=0.0, top=1.0)
            
      
      # A message for the terminal
      if self.printload and cube.slicepos:
         s = "\nLoading image at slice position (in pixels)=%s"% str(cube.slicepos[i])
         flushprint(s)

      # We need different axes in this cube.
      if cube.slicepos:
         cube.fitsobj.set_imageaxes(cube.axnums[0], cube.axnums[1],
                                    slicepos=cube.slicepos[i])
      # Set the box
      cube.fitsobj.set_limits(cube.pxlim, cube.pylim)

      mplim = cube.fitsobj.Annotatedimage(cube.frame, cmap=cube.cmap,
                                          clipmin=cube.vmin, clipmax=cube.vmax,
                                          gridmode=cube.gridmode, adjustable='box-forced',
                                          newaspect=cube.pixelaspectratio,
                                          clipmode=cube.clipmode, clipmn=cube.clipmn,
                                          callbackslist=cube.callbackslist)
      # Prepare for blitting
      # 
      mplim.Image(animated=True, alpha=1.0)                
      mplim.plot()

      # We add two attributes that are necessary to restore graticules
      mplim.linepieces = []
      mplim.textlabels = []
      mplim.cubenr = cnr                        # Essential
      if cube.cmap is None:
         cube.cmap = mplim.cmap

      
      if not (i == 0 and cnr == 0):
         # Make the previous image (that is on the display now) invisible
         oldim.image.im.set_visible(False)

      # The text for the info object is first set to a dummy. This is the
      # text we see while loading the images. After the loading, the
      # text changes to a more informative message.
      s = "image number %d: "%self.movieframecounter
      mplim.info = None
      if self.imageinfo:
         mplim.info = mplim.frame.text(0.01, 0.99, s,
                                       horizontalalignment='left',
                                       verticalalignment='top',
                                       transform=mplim.frame.transAxes,
                                       fontsize=8, color='w', animated=True,
                                       family='monospace',
                                       bbox=dict(facecolor='red', alpha=0.5))

         if i == 0:
            mplim.infoobj = mplim.info
            mplim.infoobj.set_animated(False)  # Otherwise nothing will be displayed
         else:
            mplim.infoobj = oldim.infoobj
            mplim.infoobj.set_text(str(i))     # Dummy
      else:
         mplim.infoobj = None

      # Note that in the next method, each movie gets its own informative
      # text about its slice. The attribute is called 'slicemessage'
      self.lastimage = mplim
      self.movieimages.append(mplim, visible=True, cubenr=cnr, slicemessage=s)
      self.movieframecounter += 1
      cube.imagesinthiscube.append(mplim)

      if cb.count+1 == cube.nummovieframes:
         if cube.hasgraticule:
            self.new_graticule(cube, visible=False)
         # Alwas create a colorbar. Use set_visible to make it (in)visible.
         self.new_colorbar(cube)
         # Set a message for the current spectral translation
         cube.set_slicemessages(cube.fitsobj.spectrans)  # spectrans is the same for all mplim's
         # Set the color normalization to equal for all in this cube.
         # We gathered all the images and know for each image the minimum
         # and maximum value in the Annoteded image object (attributes clipmin, clipmax)
         # These attributes are always finite numbers. If a vmin and vmax
         # were given, then the clips for all the images are set to these values.
         flushprint("Voor 1e maal naar setcubeclips is clipmode: %d"%(cube.clipmode))
         cube.setcubeclips(cube.vmin, cube.vmax, cube.clipmode, cube.clipmn)
      
      # Note that if we use the Cubes class from a gui program, then the gui
      # will be in an endless loop and we don't need to give a show() command.
      # If however, the class is used from a simple script, then the script
      # should execute Matplotlib's show() command.
      fig.canvas.draw()  # Do not blit yet because movie container is not filled
      
      cb.count += 1      
      if self.progressbar:
         self.progressbar.setValue(cb.count)
      if cb.count == cube.nummovieframes:  # We already added 1 to cb.count
         # Deschedule this callback. Perhaps more loads are scheduled.
         cb.deschedule()
         # The use of the timerlist is to support loading multiple cubes
         # The loading process is appended to the previous and will not
         # interfere with the current loading process. Here we check whether
         # 
         self.timerlist.pop(0)    # Get next element until there is nothing more
         if self.timerlist:
            self.timerlist[0].schedule()
         else:
            # The last image must be set to invisible
            # This cannot be done by the setimage() method because this
            # method does not know yet what the previous image is.
            self.lastimage.image.im.set_visible(False)
            # Interesting point to show memory use to calling environment
            self.callback('memory', getmemory())
            self.set_coordinate_mode(resetim=False)           # Configure output of coordinates for all images
            # At this point there is nothing to load anymore, so we start
            # to define some callbacks for mouse interaction
            if self.cubelist[0].panelscb is None:             # First time
               self.set_interactionMovieToPanels()
               #self.drawpanels()                             # Draw the slice panels for the first time
            else:
               self.update_interactionMovieToPanels()
            # Panels can get new sizes e.g after adding a cube while the
            # input fields already had slices defined for the new cube
            #self.redopanels()
            flushprint("MAPUTILS SETS IMAGE nr %d"%(cube.movieframeoffset))
            self.movieimages.setimage(cube.movieframeoffset)
            self.movieimages.movie_events(allow=True)
            cube.callback('finished')            # Callback from calling environment to finish up
         if self.progressbar:
            self.progressbar.reset()                          # Reset bar
         
         cube.callback('resetcursor')

     
   def set_interactionMovieToPanels(self):
      #-------------------------------------------------------------------------
      """
      The movie frames for the current cube are all stored in the movie.
      Register a callback for interaction in the movie. Each cube is
      associated with a matplotlib frame (Axes object). So a change in
      mouse position will update the slice panels that belong to each cube
      """
      #-------------------------------------------------------------------------
      for cube in self.cubelist:
         if cube.nummovieframes > 1:
            cube.panelscb = AxesCallback(self.updatepanels,
                                         cube.frame,
                                        'motion_notify_event',
                                         cubenr=cube.cnr)
         # We need a separate callback for splitscreen actions.
         # These actions need the control key to be pressed
         # and the mouse to be moved.
         cube.splitcb = AxesCallback(self.splitscreen,
                                     cube.frame,
                                    'motion_notify_event',
                                     cubenr=cube.cnr)
         cube.splitcb.deschedule()
         for aim in cube.imagesinthiscube:
            aim.splitcb = cube.splitcb



   def update_interactionMovieToPanels(self):
      """-----------------------------------------------------------------------
      Update interaction in movie container. This implies that we add a new
      callback, because this method is called only when we added a new cube.
      -----------------------------------------------------------------------"""      
      for cube in self.cubelist:
         if cube.panelscb is not None:
            cube.panelscb.schedule()         
         else:
            if cube.nummovieframes > 1:
               cube.panelscb = AxesCallback(self.updatepanels,
                                            cube.frame,
                                           'motion_notify_event',
                                            cubenr=cube.cnr)

         if cube.splitcb is None:
            cube.splitcb = AxesCallback(self.splitscreen,
                                        cube.frame,
                                       'motion_notify_event',
                                        cubenr=cube.cnr)
            for aim in cube.imagesinthiscube:
               aim.splitcb = cube.splitcb

         cube.splitcb.deschedule()


   def end_interactionMovieToPanels(self):
      """-----------------------------------------------------------------------
      End interaction in side panels.
      Note that this method is called in the append() method where the
      timer is scheduled for the loadimages() method. There we append
      a cube which does not have any callbacks set, but in a loop
      over all the cubes we check whether attribute 'panelscb' is set.
      -----------------------------------------------------------------------"""
      for cube in self.cubelist:
         if cube.panelscb is not None:
            cube.panelscb.deschedule()
         if cube.splitcb is not None:
            cube.splitcb.deschedule()
# -- End of this source --
