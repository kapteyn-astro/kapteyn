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
# VERSION: 1.0
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#
# Todo:
# -positionmessage moet nog +inf, -inf onderscheiden
# -Blanksgedrag beschrijven
# -Alle voorbeelden herschrijven voor aspectratio
# -Apart blokje documenteren voor interactie
# -Iets doen met figuresize? Opruimen get_aspectratio in FITSimage
#----------------------------------------------------------------------

"""
Module maputils 
===============

Introduction
------------

One of the goals of the Kapteyn Package is to provide a user/programmer basic
tools to make plots of image data from FITS files.
These tools are based on the functionality of PyFITS and Matplotlib.
The methods from these packages are mofified in *maputils* for an optimal
support of inspection and presentation of astronomical image data with
easy to write and usually very short Python scripts. To illustrate
what can be done with this module, we list some steps you need
in the process to create a hard copy of an image from a FITS file.

* Open FITS file on disk or from a remote location (URL)
* Specify in which header data unit the image data is stored
* Specify the data slice for data sets with dimensions > 2
* Specify the order of the image axes
* Set the limits in pixels of both image axes

Then for the display:

* Plot the image or a mosaic of images in the correct aspect ratio
* Plot contours
* Plot world coordinate labels along the image axes  (:mod:`wcsgrat`)
* Plot coordinate graticules (:mod:`wcsgrat`)
* Interactively change color map and color limits
* Read the position of features in a map and write it to screen
* Resize your plot canvas to get the wanted layout
* Write the result to *png* or *pdf* (or another format from a list)

Of course there are many programs that can do this job. But most probably
no program does it exactly the way you want or it cannot write
a hard copy with sufficient quality to publish. Also you cannot change,
or add your own extensions as easy as with this module.

Module :mod:`maputils` is very useful as a tool to extract and plot
data slices from data sets with more than two axes. It can plot
so called *Position-Velocity* maps with correct WCS annotation using
the 'missing' spatial axis.

To facilitate the input of the correct data to open a FITS image,
to specify the right data slice or to set the pixel limits for the
image axes, we implemented also some helper functions.
These functions are primitive (terminal based) but effective. You are invited to
replace them by enhanced versions, perhaps with a graphical user interface.

Here is an example of what you can expect. We have a three dimensional dataset
on disk called *rense.fits* with axes RA, DEC and VELO. The image below
is a data slice in RA, DEC at VELO=50. Its data limits are set to [10,90, 10,90]
We changed interactively the color map (keys *pageup/pagedown*)
and the color limits (pressing right mouse button while moving mouse) and saved
a hard copy on disk.

.. literalinclude:: maputils.intro.2.py


   
   
.. image:: maputils_plot1.*
   :width: 700
   :align: center
   
   
.. centered:: Image from FITS file with graticules and WCS labels

Prompt functions
----------------

.. index:: Open a FITS file
.. autofunction:: getfitsfile
.. index:: Set axis numbers of FITS image
.. autofunction:: getimageaxes
.. index:: Set pixel limits in FITS image
.. autofunction:: getbox

Class FITSimage
---------------

.. index:: Extract image data from FITS file
.. autoclass:: FITSimage


Class MPLimage
--------------

.. index:: Plot image with Matplotlib
.. autoclass:: MPLimage

Class FITSaxis
--------------

.. autoclass:: FITSaxis

Class ImageContainer
--------------------

.. autoclass:: ImageContainer
"""
# In case we want to use the plot directive, we have an exampe here
# .. plot:: /Users/users/vogelaar/MAPMAKER/maputils.intro.1.py
#

#from matplotlib import use
#use('qt4agg')

from matplotlib.pyplot import setp as plt_setp,  get_current_fig_manager as plt_get_current_fig_manager
from matplotlib import cm
import matplotlib.nxutils as nxutils
import pyfits
import numpy
from kapteyn import wcs
from kapteyn import wcsgrat
#from kapteyn import mplutil
from mplutil import AxesCallback
import readline
from types import TupleType as types_TupleType
from types import  ListType as types_ListType
from string import upper as string_upper
from re import split as re_split

sequencelist = (types_TupleType, types_ListType)

__version__ = '1.0'



def getbox(pxlim, pylim, axnameX, axnameY):
#-----------------------------------------------------------
   """
External helper function which returns the
limits in pixels of the x- and y-axis.
The input syntax is: xlo,xhi, ylo,yhi. For *x* and *y*
the names of the image axes are subsituted.
Numbers can be separated by comma's and or
spaces. A number can also be specified
with an expression e.g. ``0, 10,  10/3, 100*numpy.pi``.
All these numbers are converted to integers.

:param pxlim:
   Sequence of two numbers representing limits
   in pixels along the x axis.
:type pxlim:
   tuple with two integers
:param pylim:
   Sequence of two numbers representing limits
   in pixels along the y axis.
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
      fitsobject.set_limits(promptfie=maputils.getbox)
   
   This prompt accepts e.g.::
   
       >>>  0, 10   10/3, 100*numpy.pi

   Note the mixed use of spaces and comma's to
   separate the numbers. Note also the use of
   NumPy for mathematical functions.
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



def getfitsfile(defaultfile=None, hnr=None, memmap=None):
#-----------------------------------------------------------------
   """
An external helper function for the FITSimage class to
prompt a user to open the right Header Data Unit (hdu)
of a FITS file.
A programmer can supply his/her own function as long
as the prameters that are returned are the hdu list
and the header unit number of the wanted header from that list.
   
:param defaultfile:
   Name of FITS file on disk or url of FITS file on the internet.
   The syntax follows the standard described in the PyFITS documentation.
   See also the examples.
:type defaultfile:
   String
:param hnr:
   The number of the FITS header that you want to use.
   This function lists the hdu information and when
   hnr is not given, you will be prompted.
:type hnr:
   Integer
:param memmap:
   Set PyFITS memory mapping on/off. Let PyFITS set the default.
:type memmap:
   Boolean

:Prompts:
   1. *Enter name of fits file ...... [a default]:*

      Enter name of file on disk of valid url.
   2. *Enter number of Header Data Unit ...... [0]:*

      If a FITS file has more than one header, one must decide
      which header contains the required image data.
      
:Returns:
   The HDU list and the user selected index of the wanted 
   hdu from that list. The HDU list is returned so that it
   can be closed in the calling environment.
   
:Notes:
   --
   
:Examples:  
   PyFITS allows url's to retreive FITS files e.g.::
   
      http://www.atnf.csiro.au/people/mcalabre/data/WCS/1904-66_ZPN.fits.gz
   """
#--------------------------------------------------------------------

   while True:
      try:
         if defaultfile == None:
            filename = ''
            s = "Enter name of FITS file: "
         else:
            filename = defaultfile
            s = "Enter name of FITS file ...... [%s]: " % filename   # PyFits syntax
         fn = raw_input(s)
         if fn != '':
            filename = fn
         # print "Filename memmap", filename, memmap
         hdulist = pyfits.open(filename, memmap=memmap)
         break
      except IOError, (errno, strerror):
         print "I/O error(%s): %s" % (errno, strerror)
      except:
         print "Cannot open file, unknown error."
         con = raw_input("Abort? ........... [Y]/N:")
         if con == '' or con.upper() == 'Y':
            raise Exception, "Loop aborted by user"

   hdulist.info()
   if hnr == None:
      n = len(hdulist)
      if  n > 1:
         while True:
            p = raw_input("Enter number of Header Data Unit ...... [0]:")
            if p == '':
               hnr = 0
            else:
               hnr = eval(p)
            if hnr < n:
               break 
      else:
         hnr = 0

   return hdulist, hnr, filename



def getimageaxes(fitsobj, axnum1=None, axnum2=None):
#-----------------------------------------------------------------------
   """
Helper function for FITSimage class. It is a function that requires
interaction with a user. Therefore we left it out of any class
definition. so that it can be replaced by any other function that
returns the position of the data slice in a FITS file.

It prompts the user
for the names of the axes of the wanted image. For a
2D FITS data set there is nothing to ask, but for
dimensions > 2, we should prompt the user to enter the
image axes. Then also a list with pixel positions should
be returned. These positions set the position of the data
slice on the axes that do not belong to the image.
Only with this information the right slice can be extracted.

User is prompted in a loop until a correct input is given.
If a spectral axis is part of the selected image then
a second prompt is prepared for the input of the required spectral
translation.

:param fitsobj:
   An object from class FITSimage. This prompt function must
   have knowledge of this object such as the allowed spectral
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
   1. Name of the image axes:
        *Enter 2 axes from (list with allowed axis names) .... [default]:*

        e.g.: ``Enter 2 axes from (RA,DEC,VELO) .... [RA,DEC]:``

        The axis names can be abbreviated. A minimal match is applied.

   2. The spectral translation if one of the image axes is a spectral axis.
   
         *Enter number between 0 and N of spectral translation .... [native]:*

         *N* is the number of allowed translations  minus 1.
         The default *Native* in this context implies that no translation is applied.
         All calculations are done in the spectral type given by FITS header
         item *CTYPEn* where *n* is the number of the spectral axis.

   
:Returns:
   * *axnum1*:
     Axis number of first image axis. Default or entered by a user.
   * *axnum2*:
     Axis number of second image axis. Default or entered by a user.
   * *slicepos*:
     A list with pixel positions. One pixel for each axis
     outside the image in the same order as the axes in the FITS
     header. These pixel positions are necessary
     to extract the right 2D data from FITS data with dimensions > 2.
   * *spectrans*:
     The selected spectral translation from a list with spectral
     translations that are allowed for the input object of class FITSimage.
     A spectral translation translates for example frequencies to velocities.


:Example:
   Interactively set the axes of an image using a prompt function::
   
      # Create a maputils FITSimage object from a FITS file on disk
      fitsobject = maputils.FITSimage('rense.fits')
      fitsobject.set_imageaxes(promptfie=maputils.getimageaxes)
   
   """
#-----------------------------------------------------------------------
   n = fitsobj.naxis
   if (axnum1 == None or axnum2 == None):
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
         if axnum1 == None:
            a1 = 1
            ax1 = -1
         if axnum2 == None:
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
            if (ax1 == ax2):
               print "axis 1 == axis 2"
         else:
            print "Number of images axes must be 2. You entered %d" % (len(axes),)
      print  "You selected: ", fitsobj.axisinfo[ax1].axname, fitsobj.axisinfo[ax2].axname
      axnum1 = ax1; axnum2 = ax2
   axperm = [axnum1, axnum2]


   # Retrieve pixel positions on axes outside image
   slicepos = []
   if n > 2:
      for i in range(n):
         axnr = i + 1
         maxn = fitsobj.axisinfo[axnr].axlen
         if (axnr not in [ax1,ax2]):
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

   # Ask user to enter spectral translation if one of the axes is spectral.
   asktrans = False
   for ax in axperm:
      if fitsobj.axisinfo[ax].wcstype == 'spectral':
         asktrans = True

   spectrans = None
   nt = len(fitsobj.allowedtrans)
   if (nt > 0 and asktrans):
      print "Allowed spectral translations:"
      for i, st in enumerate(fitsobj.allowedtrans):
         print "%d : %s" % (i, st)
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
                  spectrans = fitsobj.allowedtrans[st]
            else:
               unvalid = False
         except:
            unvalid = True


   return axnum1, axnum2, slicepos, spectrans



class MPLimage(object):
#-----------------------------------------------------------------
   """
This class creates an object that can be displayed with
methods from Matplotlib. Usually these objects are created
within the context of a :class:`FITSimage` object but it can
also be used as a stand alone class.

The purpose of the class is to facilitate a user in plotting
FITS image data. Of course one could use the Matplotlib functions
and methods as they are, but these methods set useful
defaults. It does focus on plotting an image and plotting contours.
For these contours we added methods to change the properties of
individual contours and contour labels.

:param frame:
   Couple image data to this frame in Matplotlib.
:type frame:
   Matplotlib Axes instance
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
   with methods *toworld()* and *topixel()* needed for conversions
   between pixels and world coordinates.
:type projection:
   Instance of Projection class from module :mod:`wcs`
:param mixpix:
   The axis number (FITS standard i.e. starts with 1) of the missing spatial axis for
   images with only one spatial axis (e.q. Position-Velocity plots).
:type mixpix:
   Value *None* or an integer

:Attributes:
   
    .. attribute:: frame
          
          Matplotlib Axes instance
    
    .. attribute:: fig

          Matplotlib Figure instance
          
    .. attribute:: map

          Extracted image date
          
    .. attribute:: mixpix

          The pixel of the missing spatial axis in a Position-Velocity
          image
          
    .. attribute:: projection

          An object from the Projection class as defined in module :mod:`wcs`
          
    .. attribute:: pxlim

          Pixel limits in x = (xlo, xhi)
          
    .. attribute:: pylim

          Pixel limits in y = (ylo, yhi)

    .. attribute:: pixelaspectratio

          The aspect ratio of the pixels in this image. It is a default value
          for attribute :attr:`aspectratio`. The pixel aspect ratio should have been supplied
          in the constructor of this class, because this class has
          no knowledge of pixel sizes.
    
    .. attribute:: aspectratio

          Aspect ratio of a pixel according to the FITS header.
          For spatial maps this value is used to set and keep an
          image in the correct aspect ratio.
          
    .. attribute:: box
     
          box = (self.pxlim[0]-0.5, self.pxlim[1]+0.5, self.pylim[0]-0.5, self.pylim[1]+0.5)

    .. attribute:: levels

          Entered or calculated contour levels
          
    .. attribute:: labs

          Matplotlib Text instances which are used to annotate contours.
          
    .. attribute:: datmin

          The minimum image value (in units of FITS keyword *BUNIT*)
          
    .. attribute:: datmax

          The maximum image value (in units of FITS keyword *BUNIT*)

    .. attribute:: clipmin

          The minimum value of the range of image values for which
          the colors are distributed.

    .. attribute:: clipmax

          The maximum value of the range of image values for which
          the colors are distributed.
          
    .. attribute:: im

          A *matplotlib.image.AxesImage* instance.

    .. attribute:: cb

          A *matplotlib.colorbar.Colorbar* instance.


:Methods:

.. index:: How to preserve the pixel aspect ratio
.. automethod:: set_aspectratio
.. automethod:: add_subplot
.. automethod:: add_axes
.. automethod:: imshow
.. automethod:: colorbar
.. automethod:: contour
.. automethod:: set_contattr
.. automethod:: clabel
.. automethod:: set_labelattr
.. automethod:: toworld
.. automethod:: topixel
.. automethod:: positionmessage
.. automethod:: on_move
.. automethod:: on_click
.. automethod:: key_pressed
.. automethod:: motion_events
.. automethod:: key_events
.. automethod:: click_events
.. automethod:: getflux

   """
#--------------------------------------------------------------------
   def __init__(self, fig, pxlim, pylim, imdata, projection, mixpix=None, slicepos=None, pixelaspectratio=None):
      self.frame = None
      self.fig = fig
      self.map = imdata
      self.mixpix = mixpix
      self.projection = projection
      self.pxlim = pxlim
      self.pylim = pylim
      self.slicepos = slicepos
      self.aspectratio = None
      self.pixelaspectratio = pixelaspectratio
      self.box = (self.pxlim[0]-0.5, self.pxlim[1]+0.5, self.pylim[0]-0.5, self.pylim[1]+0.5)
      self.levels = None
      self.figmanager = plt_get_current_fig_manager()
      self.polygons = []
      self.currentpolygon = 0
      self.filenameout = "polygon.dat"
      self.datmin = self.map[numpy.isfinite(self.map)].min()  # Take care of -inf, +inf & NaN
      self.datmax = self.map[numpy.isfinite(self.map)].max()
      self.clipmin = self.datmin
      self.clipmax = self.datmax
      self.epsilon = 15            # Neighbourhood of a marker
      self.world = False
      self.fluxfie = lambda s, a: s/a
      self.im = None
      self.cb = None               # The color bar instance
      self.cbfontsize = None       # Font size of colorbar labels
      self.cmindx = 0              # Index in array of color maps
      self.cidmove = None          # Identifier for the mouse move callback registration


   def set_aspectratio(self, aspectratio=None):
   #--------------------------------------------------------------------
      """
      This method sets the aspect ratio of the display pixels.
      The aspect ratio is defined as *pixel height / pixel width*.
      For many images this value is not 1 so we need to supply
      the display routines with this information. When parameter
      *aspectratio* is omitted in the call then a default is copied
      from attribute *pixelaspectratio*.
      A correct aspect ratio ensures correct ratios in world coordinates
      (which is essential for spatial maps).
       
      :param aspectratio:
         Defined as pixel *height / pixel width*. With this parameter you 
         can set the aspect ratio of your image. If the value is *None*
         then the default is the aspect ratio that preserves equal distances
         in world coordinates. If somehow this default is not set
         in the constructor of this class, then 1.0 is assumed.
      :type aspectratio:
         Floating point number
      """
   #--------------------------------------------------------------------
      if aspectratio == None:
         if self.pixelaspectratio == None:
            self.pixelaspectratio = 1.0
         self.aspectratio = self.pixelaspectratio
      else:
         self.aspectratio = abs(aspectratio)
      #nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      #ny = float(self.pylim[1] - self.pylim[0] + 1)
      #self.aspectratio *= nx/ny
      #print "apsectratio", self.aspectratio, nx,ny



   def add_subplot(self, *args, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *add_subplot()* but now as a method of
   the current :class:`MPLimage` object. If this was
   created with an aspect ratio (parameter *aspectratio) unequal to *None*
   then some extra keyword arguments are appended which preserve the
   aspect ratio when the plot window is resized.

   :param `*args`: Usually a rectangle l,b,w,h
   :type `*args`: Matplotlib parameter arguments
   :param `**kwargs`:
      Arguments to change the properties of the '*Axes* object.
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Attributes:
      
      .. attribute:: frame

         This is a Matplotlib Axes instance. We call it a frame.
         It is the rectangular area in which data is plotted.
         This value is used to set the frame for the graticules too,
         as in:

         ``gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)``

   :Examples:
      The example below is a plot without setting an aspect ratio:

      .. literalinclude:: maputils.add_subplot.1.py

      
      In the example below we set the plot frame (i.e. a Matplotlib Axes instance)
      with an aspect ratio:

      .. literalinclude:: maputils.add_subplot.2.py

      """
   #--------------------------------------------------------------------
      if self.aspectratio == None:
         self.frame = self.fig.add_subplot(*args, **kwargs)
      else:
         # An aspect ratio was set and should be preserved if one resizes the plot window.
         kwargs.update({'aspect':self.aspectratio, 'adjustable':'box', 'autoscale_on':False, 'anchor':'C'})
         self.frame = self.fig.add_subplot(*args, **kwargs)

      #self.frame.apply_aspect()



   def add_axes(self, *args, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *add_axes()* but now as a method of
   the current :class:`MPLimage` object. If this was
   created with an aspect ratio (parameter *aspectratio*) unequal to *None*
   then some extra keyword arguments are appended which preserve the
   aspect ratio when the plot window is resized.

   :param `*args`: Usually a rectangle l,b,w,h
   :type `*args`: Matplotlib parameter arguments
   :param `**kwargs`:
      Arguments to change the properties of the *Axes* object.
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Attributes:
      
      .. attribute:: frame

         This is a Matplotlib Axes instance. We call it a frame.
         It is the rectangular area in which data is plotted.
         This value is used to set the frame for the graticules too,
         as in:

         ``gratplot = wcsgrat.Plotversion('matplotlib', fig, image.frame)``


      """
   #--------------------------------------------------------------------
      if self.aspectratio == None:
         newkwargs = ({'aspect':'auto','autoscale_on':True}) # Allowed to overwrite
         newkwargs.update(kwargs)
         self.frame = self.fig.add_axes(*args, **newkwargs)
      else:
         # An aspect ratio was set and should be preserved if one resizes the plot window.
         kwargs.update({'aspect':self.aspectratio, 'adjustable':'box', 'autoscale_on':False, 'anchor':'C'})
         self.frame = self.fig.add_axes(*args, **kwargs)
      #self.frame.apply_aspect()



   def imshow(self, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *imshow()* but now as a method of
   the current :class:`MPLimage` object.
   Some keywords were added to process the curent image data.
   It sets the origin of the plot (lower left corner). It sets also
   the data limits for the plot axes, the aspect ratio and the minimum
   and maximum values of the image for the colors.
   Finally, this method sets the interpolation to 'nearest', i.e.
   no interpolation. Usually this is the default for viewing
   astronomical data.

   These defaults can be changed. When this method is called with keyword
   arguments which are the same as the defaults set here, then they will
   overwrite the defaults.
   Possible keyword arguments:
   *cmap, interpolation, norm, vmin, vmax, alpha, filternorm, filterrad*.

   :param `**kwargs`:
      Arguments to change the properties
      of a :class:`matplotlib.image.AxesImage` instance. Examples are:
      *colors, alpha, cmap, norm, interpolation* etc.
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Raises:
      :exc:`Exception`
          *The frame attribute is not yet set. Use add_subplot() first!* -- 
          One cannot plot an image if Matplotlib doesn't know where to plot it.
          An *Axes* object must be created. We call this a frame. Frames are
          created with :meth:`add_subplot` or with :meth:`add_axes`.
          
          
   :Attributes:

     .. attribute:: im

        *matplotlib.image.AxesImage* instance
        
   :Examples:
      A simple demonstration how to plot a channel map from
      a data cube:

      .. literalinclude:: maputils.imshow.1.py
   
      """
   #--------------------------------------------------------------------
      # Note that imshow's y origin defaulted to upper in Matplotlib's rc file.
      # Change default to 'lower'
      if self.frame == None:
         raise Exception("The frame attribute is not yet set. Use add_subplot() first!")
      aspect = self.frame.get_aspect()
      newkwargs = ({'aspect':aspect, 'origin':"lower", 'extent':self.box,
                     'vmin':self.datmin, 'vmax':self.datmax, 'interpolation':'nearest'})
      newkwargs.update(kwargs)
      self.im = self.frame.imshow(self.map, **newkwargs)
      self.frame.set_xlim((self.box[0], self.box[1]))
      self.frame.set_ylim((self.box[2], self.box[3]))
      # It can be that vmin and/or vamx are part of the keyword arguments
      # Make sure that if the are that the min and max clip values are updated.
      # Otherwise the keep the values of datmin and datmax.
      self.clipmin, self.clipmax = self.im.get_clim()



   def colorbar(self, fontsize=8, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *colorbar()* but now as a method of
   the current :class:`MPLimage` object.
   We added a parameter *fontsize* to change the size of the fonts
   in the labels along the colorbar.
   The default for the orientation is set to vertical. 
   These defaults can be changed. When this method is called with keyword
   arguments which are the same as the defaults set here, then they will
   overwrite the defaults.
   Possible keyword arguments:
   *orientation, fraction, pad, shrink, aspect*.

   :param fontsize:
      The font size of the colorbar labels.
   :type fontsize:
      Integer
   :param `**kwargs`:
      Arguments to change the properties
      of a :class:`matplotlib.colorbar.Colorbar` instance. Examples are:
      *colors, alpha, cmap, norm, interpolation* etc.
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Attributes:

     .. attribute:: cb

        *matplotlib.colorbar.Colorbar* instance
        
   :Examples:
      Add a horizontal color bar. Increase the font size to 10::

         image = fitsobject.createMPLimage(fig)
         image.add_subplot(1,1,1)
         image.imshow(interpolation='gaussian')
         image.colorbar(fontsize=10, orientation='horizontal')
   
      """
   #--------------------------------------------------------------------
      newkwargs = ({'orientation':'vertical'})
      newkwargs.update(kwargs)
      self.cb = self.fig.colorbar(self.im, ax=self.frame, **newkwargs)
      self.cbfontsize = fontsize
      for t in self.cb.ax.get_yticklabels():
         t.set_fontsize(self.cbfontsize)
      for t in self.cb.ax.get_xticklabels():
         t.set_fontsize(self.cbfontsize)



   def contour(self, levels=None, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *contour()* but now as a method of
   the current :class:`MPLimage` object.

   Note that the attributes for individual contours
   can be changed with method :meth:`set_contattr`.

   :param levels:
      One level or a sequence of levels corresponding to the image values for
      which you want to plot contours.
      to annotate the contour.
   :type levels:
      Value *None* or 1 floating point number or a sequence of floating point numbers
   :param `**kwargs`:
      Arguments to change the properties
      of a :class:`matplotlib.contour.ContourSet` instance. Examples are:
      *colors, alpha, cmap, norm, linewidths* and *linestyles.*
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Returns:

      *levels* - a numpy.ndarray object with the entered or calculated levels

   :Attributes:

      .. attribute:: CS
        
         A matplotlib.contour.ContourSet instance

      .. attribute:: levels

         The entered or calculated contour levels


   :Examples:
      Next example shows how to plot contours over an image.
      and how to change properties of individual contours and labels.
      
      .. literalinclude:: maputils.contour.2.py
     
      """
   #--------------------------------------------------------------------
      if self.frame == None:
         raise Exception("The frame attribute is not yet set. Use add_subplot() first!")
      if levels == None:
         self.CS = self.frame.contour(self.map, origin='lower', extent=self.box, **kwargs)
         self.levels = self.CS.levels
      else:
         if type(levels) not in sequencelist:
            levels = [levels]
         self.CS = self.frame.contour(self.map, levels, origin='lower', extent=self.box, **kwargs)
         self.levels = self.CS.levels
      return self.CS.levels



   def set_contattr(self, levels=None, **kwargs):
   #--------------------------------------------------------------------
      """
   Set contour attributes for one contour. The contour is given
   by its index in the levels list (attribute *levels*).
   When contours are set, an attribute *CS* is set. This attribute
   stores acollection of contours. We can isolate a contour by
   an array index. Then such a contour is an instance of class
   :class:`matplotlib.LineCollection`. Some of the properties of
   these objects are *linewidth*, *color*, and *linestyle*.
   A different linestyle is often applied to image data less than zero.

   :param levels:
      One level or a sequence of levels corresponding to the contour for
      which you want to want to change properties.
      If *levels=None* then all the properties for all labels are changed.
   :type levels:
      Value *None* or 1 floating point number or a sequence of floating point numbers
   :param `**kwargs`:
      Arguments to change the properties
      of a :class:`matplotlib.collections.LineCollection` instance. Examples are:
      *backgroundcolor, color, family, fontname, fontsize, fontstyle
      fontweight, rotation, text, ha, va.*
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Examples: Change properties of the first contour

      >>> image.set_contattr(0.02, linewidth=4, color='r', linestyle='dashed')
      
      """
   #--------------------------------------------------------------------
      l = len(self.levels)
      if l == 0:
         # There are no contours, do nothing
         return

      if levels == None:
         # Change properties of all contours
         for c in self.CS.collections:
            plt_setp(c, **kwargs)
      else:
         # Change only contour properties for levels in parameter 'levels'.
         if type(levels) not in sequencelist:
            levels = [levels]
         for lev in levels:
            try:
               i = list(self.levels).index(lev)
            except ValueError:
               i = -1 # no match
            if i != -1:
                plt_setp(self.CS.collections[i], **kwargs)



   def clabel(self, levels=None, **kwargs):
   #--------------------------------------------------------------------
      """
   Same as Matplotlib's method *clabel()* but now as a method of
   the current :class:`MPLimage` object.

   :param levels:
      One level or a sequence of levels corresponding to the contour for
      which you want to want to plot a label
      to annotate the contour.
   :type levels:
      One floating point number or a sequence of floating point numbers
   :param `**kwargs`:
      Arguments to change the properties
      of all :class:`matplotlib.Text` instance. Examples are:
      *fontsize, colors, inline, fmt*, etc.
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary- 

   :Attributes:

     .. attribute:: labs
        
        a list of n matplotlib.text.Text objects
        
   :Examples:
      Plot contours at default levels and label them

      >>> image.contour()
      >>> image.clabel(colors='r', fontsize=14, fmt="%.3f")

      or plot only a label for contour at image value 0.02:

      >>> image.clabel(0.02, colors='r', fontsize=14, fmt="%.3f")

      """
   #--------------------------------------------------------------------
      if levels == None:
         self.labs = self.frame.clabel(self.CS, **kwargs)
      else:
         if type(levels) not in sequencelist:
            levels = [levels]
         self.labs = self.frame.clabel(self.CS, levels, **kwargs)



   def set_labelattr(self, levels=None, **kwargs):
   #--------------------------------------------------------------------
      """
   Change the properties of a single contour label.

   :param levels:
      One level or a sequence of levels corresponding to the contour level for
      which you want to want to change the label properties.
      If *levels=None* then all the properties for all labels are changed.
   :type levels:
      Value *None* or 1 floating point number or a sequence of floating point numbers
   :param lnr:
      Index for one label in array of labels stored in
      attribute *labs*. If the integer is outside the range
      of valid indices, nothing will be done. Also there is
      no warning.
   :type lnr:
      Integer
   :param `**kwargs`:
      Arguments to change the properties
      of a :class:`matplotlib.Text` instance. Examples are:
      *backgroundcolor, color, family, fontname, fontsize, fontstyle
      fontweight, rotation, text, ha, va*
   :type `**kwargs`:
      Matplotlib keyword arguments -dictionary-

   :Example:
      Change label that corresponds to the contour level
      0.025. Make the text horizontal and give it a yellow background.
      Change also the text of the label.

      >>> image.set_labelattr(0.025, backgroundcolor="yellow",
                text="Contour 0.025", rotation=0, color="magenta", clip_on=True)

   :Notes:
      
      * fontsize seems to apply on all labels at once.
      * fontstyle doesn't seem to work
      * fontweight doesn't seem to work
      
      """
   #--------------------------------------------------------------------
      l = len(self.labs)
      if l == 0:
         # There are no contours, do nothing
         return

      if levels == None:
         # Change properties of all contours
         for c in self.labs:
            plt_setp(c, **kwargs)
      else:
         # Change only contour properties for levels in parameter 'levels'.
         if type(levels) not in sequencelist:
            levels = [levels]
         for lev in levels:
            try:
               i = list(self.levels).index(lev) # Find this value in the list and return its index or -1
            except ValueError:
               i = -1 # no match
            if i != -1:
                plt_setp(self.labs[i], **kwargs)



   def toworld(self, xp, yp):
   #--------------------------------------------------------------------
      """
   This is a helper method for method :meth:`wcs.Projection.toworld`.
   It knows about missing spatial axis if a data slice has only one
   spatial axis. It converts pixel positions from a map to world coordinates.
   Note that pixels in FITS run from 1 to *NAXISn*.

   :param xp:
      A pixel value corresponding to the x coordinate of a position.
   :type xp:
      Floating point number
   :param yp:
      A pixel value corresponding to the y coordinate of a position.
   :type yp:
      Floating point number

   :Raises:
      If an exception is raised then the return values of the world
      coordinates are all *None*.
      
   :Returns:
      Three world coordinates: *xw* which is the world coordinate for
      the x-axis, *yw* which is the world coordinate for
      the y-axis and *missingspatial* which is the world coordinate
      that belongs to the missing spatial axis (e.g. in position-velocity
      maps). If there is not a missing spatial axis, then the value of this
      output parameter is *None*.

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
      >>> from matplotlib.pyplot import figure
      >>> fig = figure()
      >>> fitsobject = maputils.FITSimage('rense.fits')
      >>> fitsobject.set_imageaxes(1,3, slicepos=51)
      >>> image = fitsobject.createMPLimage(fig)
      >>> image.toworld(51,-20)
          (-51.282084795899998, -243000.0, 60.1538880206)
      >>> image.topixel(-51.282084795899998, -243000.0)
          (51.0, -20.0)

      """
   #--------------------------------------------------------------------
      xw = yw = None
      missingspatial = None
      try:
         if (self.mixpix == None):
            xw, yw = self.projection.toworld((xp, yp))
         else:
            xw, yw, missingspatial = self.projection.toworld((xp, yp, self.mixpix))
      except:
         pass
      return xw, yw, missingspatial



   def topixel(self, xw, yw):
   #--------------------------------------------------------------------
      """
   This is a helper method for method :meth:`wcs.Projection.topixel`.
   It knows about missing spatial axis if a data slice has only one
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

   :Raises:
      If an exception is raised then the return values of the pixel
      coordinates are all *None*.
      
   :Returns:
      Two pixel coordinates: *x* which is the world coordinate for
      the x-axis and *y* which is the world coordinate for
      the y-axis.

   :Notes:
      This method knows about the pixel on the missing spatial axis
      (if there is one). This pixel is usually the slice position
      if the dimension of the data is > 2.

   :Examples:
      See example at :meth:`toworld`
      """
   #--------------------------------------------------------------------
      x = y = None
      try:
         if (self.mixpix == None):
            x, y = self.projection.topixel((xw, yw))
         else:
            # Note that we have a fixed grid for the missing spatial axis,
            # but this does not imply a constant world coordinate. Therefore
            # we have to use the mixed transformation method.
            unknown = numpy.nan
            wt = (xw, yw, unknown)
            pixel = (unknown, unknown, self.mixpix)
            (wt, pixel) = self.projection.mixed(wt, pixel)
            x = pixel[0]; y = pixel[1]
      except:
         pass
      return x, y



   def positionmessage(self, x, y):
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
            xw, yw, missingspatial = self.toworld(x, y)
            xi = numpy.round(x) - (self.pxlim[0]-1)
            yi = numpy.round(y) - (self.pylim[0]-1)
            #if not numpy.ma.is_masked(self.map[yi-1, xi-1]):
            if not numpy.isnan(self.map[yi-1, xi-1]):
               z = self.map[yi-1, xi-1]
               if missingspatial == None:
                  s = "x,y=%6.2f,%6.2f  wcs=%10f,%10f  Z=%+8.2e " % (x, y, xw, yw, z)
               else:
                  s = "x,y=%6.2f,%6.2f  wcs=%10f,%10f,%10f  Z=%+8.2e " % (x, y, xw, yw, missingspatial, z)
            else:
               if missingspatial == None:
                  s = "x,y=%6.2f,%6.2f  wcs=%10f,%10f  Z=NaN" % (x, y, xw, yw)
               else:
                  s = "x,y=%6.2f,%6.2f  wcs=%10f,%10f,%10f  Z=NaN" % (x, y, xw, yw, missingspatial)
         else: #except:
            s = "xp,yp: %.2f %.2f " % (x, y)
      return s



   def on_move(self, axesevent):
   #--------------------------------------------------------------------
      """
   *Display position information:*
      
   Moving the mouse to another position in your image will
   update the figure message in the toolbar. Use the projection
   information from the FITS header to create a message with pixel- and
   world coordinates and the image value of the pixel.
   You need to connect the mouse event with this function,
   see example. The string with position information is
   is returned by :meth:`positionmessage`.

   *Reset color limits:*

   If you move the mouse in this image and press the **right mouse button**
   at the same time, then the color limits for image and colorbar are
   set to a new value.
   
   :param axesevent:
      AxesCallback event object with pixel position information.
   :type axesevent:
      AxesCallback instance

   :Example:
      Given an object from class "class:`MPLimage`, 
      register this callback function for the position information with:

      >>> image.motion_events()

      In the next example we present a small program that displays a
      position-velocity plot. In the toolbar (if there is
      enough space, otherwise make you window wider) you
      will see the position of the missing spatial axis as
      the third coordinate. It is the coordinate that
      changes the least if you move the mouse.

      .. literalinclude:: maputils.interaction.2.py

      In the next example we demonstrate the interaction with
      the mouse to change colors in image and colorbar. The colors are
      only changed when you move the mouse in an image while pressing
      the right mouse button. We also registerd a function for keyboard
      keys. If you press *pageup* or *pagedown* you will loop through a
      list will color maps.
      
      .. literalinclude:: maputils.interaction.3.py

   :Notes:
      
      *  **Important**: If you want more than one image in your plot and
         want mouse interaction for each image separately, then you have
         to register this callback function for each instance of class
         :class:`MPLimage`. But you have to register them before you register
         any *press* related events (key_press_event, button_press_event).
         Otherwise, you will get only the position
         information from the first image in the toolbar.
      
      *  New color limits are calculated as follows: first the position
         of the mouse is transformed into normalized coordinates.
         These values are used to set a shift and a compression factor
         for the color limits. The shift changes with horizontal moves
         and the compression with vertical moves.
         If the data range of image values is [*datmin*, *datmax*] then
         :math:`\Delta_x = datmax - datmin`. If the mouse position in normalized
         coordinates is :math:`(\delta_x,\delta_y)` then a central clip
         value is :math:`c = datmin + \delta_x*\Delta_x`. Further,
         :math:`\Delta_y = \delta_y*\Delta_x` from which we calculate
         the new clip- or color limits: :math:`clipmin = c - \Delta_y`
         and :math:`clipmax = c + \Delta_y`.

      *  A 'standard' event object has an attribute which represents the frame
         (Axes object) in which it occurred. So we can check if this
         frame corresponds to the frame in which the current image is
         plotted. However, when we also plot graticules then more frames
         correspond to the current image and there is no rule from
         which we can derive to which frame the event belongs to.
         Therefore we use a modified event object. It is an object from class
         :class:`AxesCallback` and it is connected to a frame which was
         set in the constructor. Therefore we are sure that when the event
         is triggered, it is for the right frame.
      """
   #--------------------------------------------------------------------
      s = ''
      x, y = axesevent.xdata, axesevent.ydata
      if self.figmanager.toolbar.mode == '':
         s = self.positionmessage(x, y)
      if s != '':
         self.figmanager.toolbar.set_message(s)

      if axesevent.event.button == 3:
         if self.im == None:                     # There is no image to adjust
            return
         # 1. event.xdata and event.ydata are the coordinates of the mouse location in
         # data coordinates (i.e. in screen pixels)
         # 2. transData converts these coordinates to display coordinates
         # 3. The inverse of transformation transAxes converts display coordinates to
         # normalized coordinates for the current frame.
         xy = self.frame.transData.transform((x,y))
         xyn = self.frame.transAxes.inverted().transform(xy)
         Dx = self.datmax - self.datmin
         clipC = self.datmin + Dx * xyn[0]
         Dy = Dx * xyn[1]
         self.clipmin = clipC - Dy
         self.clipmax = clipC + Dy
         self.im.set_clim(self.clipmin, self.clipmax)

         if self.cb != None:
            # Next lines seem to be necessary to keep the right
            # font size for the colorbar labels. Otherwise each
            # call to clim() wil reset the size to 12.
            for t in self.cb.ax.get_xticklabels():  # Smaller font for color bar
               t.set_fontsize(self.cbfontsize)
            for t in self.cb.ax.get_yticklabels():  # Smaller font for color bar
               t.set_fontsize(self.cbfontsize)

         self.fig.canvas.draw()



   def key_pressed(self, axesevent):
   #--------------------------------------------------------------------
      """
   A function that can be registerd to catch key presses. Currently we
   catch keys *pageup* and *pagedown* and 'r'. These page keys move through a list
   with known color maps. You will see the results of a change immediately.
   Key 'r' (or 'R') reset the colors to the original colormap and scaling.
   The default color map is called 'jet'.
   
   :param axesevent:
      AxesCallback event object with pixel position information.
   :type axesevent:
      AxesCallback instance

   :Examples:
      If *image* is an object from :class:`MPLimage` then register
      this function with:
      
      >>> image1.key_events()
      """
   #--------------------------------------------------------------------
      redraw = False
      if axesevent.event.key in ['pageup', 'pagedown']:
         maps=[m for m in cm.datad.keys() if not m.endswith("_r")]
         lm = len(maps)
         if axesevent.event.key == 'pageup':
            self.cmindx += 1
            if self.cmindx >= lm:
               self.cmindx = 0
         if axesevent.event.key == 'pagedown':
            self.cmindx -= 1
            if self.cmindx < 0:
               self.cmindx = lm - 1
         
         m = cm.get_cmap(maps[self.cmindx])
         print "Color map %i: %s" % (self.cmindx, maps[self.cmindx])
         self.im.set_cmap(m)
         redraw = True
      elif axesevent.event.key.upper() == 'R':
         self.clipmin = self.datmin
         self.clipmax = self.datmax
         self.im.set_cmap(cm.get_cmap('jet'))
         self.im.set_clim(self.clipmin, self.clipmax)
         redraw = True

      if redraw:
         if self.cb != None:
            # Next lines seem to be necessary to keep the right
            # font size for the colorbar labels. Otherwise each
            # call to clim() wil reset the size to 12.
            for t in self.cb.ax.get_xticklabels():  # Smaller font for color bar
               t.set_fontsize(self.cbfontsize)
            for t in self.cb.ax.get_yticklabels():  # Smaller font for color bar
               t.set_fontsize(self.cbfontsize)

         self.fig.canvas.draw()



   def on_click(self, axesevent):
   #--------------------------------------------------------------------
      """
   Print position information of the position where
   you clicked with the mouse. Print the info in the toolbar.
   Register this function with an event handler,
   see example.

   :param axesevent:
      AxesCallback event object with pixel position information.
   :type axesevent:
      AxesCallback instance

   :Example:
      Register this callback function for object *image* with:

      >>> image.click_events()
      """
   #--------------------------------------------------------------------
      """
      if not event.inaxes:
         return                                  # Not in a frame
      # if event.inaxes is not self.frame:
      if not numpy.all(event.inaxes.get_position().get_points() == self.frame.get_position().get_points()):
         return
      """ 
      if axesevent.event.button != 1:
         return
      s = ''
      if self.figmanager.toolbar.mode == '':
         x, y = axesevent.xdata, axesevent.ydata
         s = self.positionmessage(x, y)
      if s != '':
         print s



   def motion_events(self):
   #--------------------------------------------------------------------
      """
      Allow this :class:`MPLimage` object to interact with the user.
      It reacts on mouse movements. During these movements, a message
      with information about the position of the cursor in pixels
      and world coordinates is displayed on the toolbar.
      """
   #--------------------------------------------------------------------
      #self.cidmove = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
      self.cidmove = AxesCallback(self.on_move, self.frame, 'motion_notify_event')




   def key_events(self):
   #--------------------------------------------------------------------
      """
      Allow this :class:`MPLimage` object to interact with the user.
      It reacts on keyboard key presses. With *pageup* and *pagedown*
      one scrolls to a list with color maps. Key 'r' (or 'R') resets
      the current image to its original color map and color limits.
      """
   #--------------------------------------------------------------------
      #self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.key_pressed)
      self.cidkey = AxesCallback(self.key_pressed, self.frame, 'key_press_event')



   def click_events(self):
   #--------------------------------------------------------------------
      """
      Allow this :class:`MPLimage` object to interact with the user.
      It reacts on pressing the left mouse button and prints a message
      to stdout with information about the position of the cursor in pixels
      and world coordinates.
      """
   #--------------------------------------------------------------------
      # self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
      self.cidclick = AxesCallback(self.on_click, self.frame, 'button_press_event')



   def getflux(self, xy, pixelstep=0.20):
   #--------------------------------------------------------------------
      """
   Return area in pixels and the sum of the image values in
   the polygon defined by *xy*.

   It is a demonstration how we can add methods to this class
   when we want to analyze the image. In this case one can
   extract the flux in a polygon. The positions of this
   polygon could be the result of interaction with the mouse.

   :param xy:
      Sequence of pixel positions (x,y)
   :type xy:
      Sequence of tuples/lists with two floating point numbers
   :param pixelstep:
      Sampling step size (usually smaller than one pixel to get
      the best results). The default is 0.2.
   :type pixelstep:
      Floating point number
   :Notes:
      There are some examples in the module :mod:`ellinteract`.
      
      """
   #--------------------------------------------------------------------
      poly = numpy.asarray(xy)
      mm = poly.min(0)
      xmin = mm[0]; ymin = mm[1]
      mm = poly.max(0)
      xmax = mm[0]; ymax = mm[1]
      xmin = numpy.floor(xmin); xmax = numpy.ceil(xmax)
      ymin = numpy.floor(ymin); ymax = numpy.ceil(ymax)
      # print xmin, xmax, ymin, ymax
      Y = numpy.arange(ymin,ymax+1, pixelstep)
      X = numpy.arange(xmin,xmax+1, pixelstep)
      l = int(xmax-xmin+1); b = int(ymax-ymin+1)
      numpoints = len(X)*len(Y)
      x, y = numpy.meshgrid(X, Y)
      pos = zip(x.flatten(), y.flatten())
      mask = nxutils.points_inside_poly(pos, poly)
      # Correction consists of three elements:
      # 1) The start position of the box must map on the start position of the array
      # 2) Positions all > 0.5 so add 0.5 and take int to avoid the need of a round function
      # 3) The start index of the array is not 
      xcor = self.pxlim[0] - 0.5
      ycor = self.pylim[0] - 0.5
      count = 0
      sum = 0.0
      for i, xy in enumerate(pos):
         if mask[i]:
            xp = int(xy[0] - xcor)
            yp = int(xy[1] - ycor)
            z = self.map[yp,xp]
            sum += z
            count += 1
      return count*(pixelstep*pixelstep), sum*(pixelstep*pixelstep)



class FITSaxis(object):
#-----------------------------------------------------------------
   """
This class defines objects which store WCS information from a FITS
header.
    
:param axisnr:
   FITS axis number. For this number the relevant keys in the header
   are read.
:type axisnr:
   Integer
:param hdr:
   FITS header
:type hdr:
   pyfits.NP_pyfits.Header instance

:Returns:
   --

:Notes:
   --
   
:Methods:

.. automethod:: printattr
.. automethod:: printinfo

   """  
#--------------------------------------------------------------------
   def __init__(self, axisnr, hdr):
      ax = "CTYPE%d" % (axisnr,)
      self.ctype = 'Unknown'
      self.axname = 'Unknown'
      self.axnamelong = 'Unknown'
      if hdr.has_key(ax):
         self.ctype = hdr[ax].upper()
         self.axnamelong = string_upper(hdr[ax])
         self.axname = string_upper(hdr[ax].split('-')[0])
      ai = "NAXIS%d" % (axisnr,)
      self.axlen = hdr[ai]
      self.axisnr = axisnr
      self.cdelt = 0.0
      ai = "CDELT%d" % (axisnr,)
      if hdr.has_key(ai):
         self.cdelt = hdr[ai]
      ai = "CRVAL%d" % (axisnr,)
      self.crval = hdr[ai]
      self.cunit = 'Unknown'
      ai = "CUNIT%d" % (axisnr,)    # Is sometimes omitted, so check first.
      if hdr.has_key(ai):
         self.cunit = hdr[ai]
      ai = "CRPIX%d" % (axisnr,)
      self.crpix = hdr[ai]
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
   axnamelong - Long axis name:  RA---NCP
   axname     - Short axis name:  RA
   cdelt      - Pixel size:  -0.007165998823
   crpix      - Reference pixel:  51.0
   crval      - World coordinate at reference pixel:  -51.2820847959
   cunit      - Unit of world coordinate:  DEGREE
   wcstype    - Axis type according to WCSLIB:  None
   wcsunits   - Axis units according to WCSLIB:  None
   outsidepix - A position on an axis that does not belong to an image:  None
   
   If we set the image axes in *fitsobject* then the wcs attributes
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
      print "axisnr     - Axis number: ", self.axisnr
      print "axlen      - Length of axis in pixels (NAXIS): ", self.axlen
      print "ctype      - Type of axis (CTYPE): ", self.ctype
      print "axnamelong - Long axis name: ", self.axnamelong
      print "axname     - Short axis name: ", self.axname
      print "cdelt      - Pixel size: ", self.cdelt
      print "crpix      - Reference pixel: ", self.crpix
      print "crval      - World coordinate at reference pixel: ", self.crval
      print "cunit      - Unit of world coordinate: ", self.cunit
      print "wcstype    - Axis type according to WCSLIB: ", self.wcstype
      print "wcsunits   - Axis units according to WCSLIB: ", self.wcsunits
      print "outsidepix - A position on an axis that does not belong to an image: ", self.outsidepix



   def printinfo(self):
   #----------------------------------------------------------
      """
   Print formatted information for this axis.
   
   :Examples:
   
   >>> from kapteyn import maputils
   >>> fitsobject = maputils.FITSimage('rense.fits')
   >>> ax1 = maputils.FITSaxis(1, fitsobject.hdr)
   >>> ax1.printinfo()
   (1) Axis RA has length 100 and units DEGREE.
   At pixel 51.000000 the world coordinate is -51.282085 (DEGREE).
   Step size is -0.007166 (DEGREE)
   WCS type of this axis: None, WCS (si) units: None

      """
   #----------------------------------------------------------
      print "(%d) Axis %s has length %d and units %s.\nAt pixel %f the world coordinate is %f (%s).\nStep size is %f (%s)" %(self.axisnr,
                  self.axname, self.axlen, self.cunit, self.crpix, self.crval, self.cunit, self.cdelt, self.cunit)
      print "WCS type of this axis: %s, WCS (si) units: %s " % (self.wcstype, self.wcsunits)
      if self.outsidepix != None:
         print "Pixel position on axis outside image: ", self.outsidepix



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
   :mod:`maputils` is function :func:`getfitsfile`
:type promptfie:
   Python function
:param hdunr:
   A preset of the index of the header from the hdu list.
   If this variable is set then it should not prompted for in the
   user supplied function *promptfie*.
:type hdunr:
   Integer
:param memmap:
   Set the memory mapping for PyFITS. The default is in the PYFITS
   version we used was memory mapping set to off (i.e. memmap=0)

:Returns:
   --
      
:Attributes:
   
    .. attribute:: filename

       Name of the FITS file (read-only).
       
    .. attribute:: hdr

       Header as read from the header (read-only).
       
    .. attribute:: naxis

       Number of axes (read-only).
       
    .. attribute:: dat

       Pointer to the raw image data (not sliced, swapped or limited in range).
       The required sliced image data is stored in attribute :attr:`map`.
       This is a read-only attribute. 
       
    .. attribute:: axperm

       Axis permutation array. These are the (FITS) axis numbers of your
       image x & y axis.
       
    .. attribute:: mixpix

       The missing pixel if the image has only one spatial axis. The other
       world coordinate could be calculated with a so called *mixed* method
       which allows for one world coordinate and one pixel.
       
    .. attribute:: axisinfo

       A list with :class:`FITSimage` objects. One for each axis. The index is
       an axis number (starting at 1), not an array index.
       
    .. attribute:: slicepos

       A list with position on axes in the FITS file which do not belong to
       the required image.
       
    .. attribute:: pxlim

       Axis limit in pixels. This is a tuple or list (xlo, xhi).
       
    .. attribute:: pylim

       Axis limit in pixels. This is a tuple or list (xlo, xhi).
       
    .. attribute:: map

       The image data. Possible sliced, axis swapped and limited in axis range.
       
    .. attribute:: imshape

       Sizes of the 2D array in :attr:`map`.
       
    .. attribute:: spectrans

       A string that sets the spectra translation. If one uses the prompt function
       for the image axes, then you will get a list of possible translations for the
       spectral axis in your image.
       
    .. attribute:: convproj

       An object from :class:`wcs.Projection`. This object is needed to
       be able to use methods *toworld()* and *topixel()* for the
       current image.
       
    .. attribute:: figsize

       A suggested figure size in X and Y directions.
       
    .. attribute:: aspectratio

       Plot a circle in world coordinates as a circle. That is, if the
       pixel size differs in X and Y, then correct the image so that the pixels
       are not plotted as squares.
      
:Notes:
   The constructor sets also a default position for a data slice if
   the dimension of the FITS data is > 2. This position is either the value
   of CRPIX from the header or 1 if CRPIX is outside the range [1, NAXIS].

:Examples:
   PyFITS allows url's to retreive FITS files. It can also read gzipped files e.g.:
   
      >>> f = 'http://www.atnf.csiro.au/people/mcalabre/data/WCS/1904-66_ZPN.fits.gz'
      >>> fitsobject = maputils.FITSimage(f)
      >>> fitsobject.printaxisinfo()
      Axis 1: RA---ZPN  from pixel 1 to   192  {crpix=-183 crval=0 cdelt=-0.0666667 (Unknown)}
      Axis 2: DEC--ZPN  from pixel 1 to   192  {crpix=22 crval=-90 cdelt=0.0666667 (Unknown)}

   Use Maputil's prompt function :func:`getfitsfile` to get
   user interaction for the FITS file specification.
   
      >>> fitsobject = maputils.FITSimage(promptfie=maputils.getfitsfile)

:Methods:

.. index:: Select image data from FITS file
.. automethod:: set_imageaxes
.. index:: Set pixel limits of image axes
.. automethod:: set_limits
.. Prepare FITS image for display
.. automethod:: createMPLimage
.. index:: Aspect ratio from FITS header data
.. automethod:: get_pixelaspectratio
.. index:: Print information from FITS header
.. automethod:: printheader
.. automethod:: printaxisinfo
.. automethod:: globalminmax

   """
#--------------------------------------------------------------------
   def __init__(self, filespec=None, promptfie=None, hdunr=None, memmap=0):
      """-------------------------------------------------------------
      See Class description
      -------------------------------------------------------------"""
      if promptfie:
         hdulist, hdunr, filename = promptfie(filespec, hdunr, memmap)
      else:
         if memmap == None:
            memmap = 0
         try:
            hdulist = pyfits.open(filespec, memmap=memmap)
            filename = filespec
         except IOError, (errno, strerror):
            print "I/O error(%s): %s" % (errno, strerror)
         except:
            print "Cannot open file, unknown error!"
            raise
         if hdunr == None:
            hdunr = 0
      hdu = hdulist[hdunr]
      self.filename = filename
      self.hdr = hdu.header
      self.naxis = self.hdr['NAXIS']
      if self.naxis < 2:
         print "You need at least two axes in your FITS file to extract a 2D image."
         print "Number of axes in your FITS file is %d" % (self.naxis,)
         hdulist.close()
         raise Exception, "Number of data axes must be >= 2."
      self.dat = hdu.data

      self.axperm = [1,2]
      self.mixpix = None 
      # self.convproj = wcs.Projection(self.hdr).sub(self.axperm)
      axinf = {}
      for i in range(self.naxis):
         # Note that here we define the relation between index (i) and
         # the axis number (i+1)
         # The axinf dictionary has this axis number as key values 
         axisnr = i + 1
         axinf[axisnr] = FITSaxis(axisnr, self.hdr)
      self.axisinfo = axinf

      hdulist.close()             # Close the FITS file

      slicepos = []                  # Set default positions (CRPIXn) on axes outside image for slicing data
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
      self.slicepos = slicepos

      n1 = self.axisinfo[self.axperm[0]].axlen
      n2 = self.axisinfo[self.axperm[1]].axlen
      self.pxlim = [1, n1]
      self.pylim = [1, n2]
      # Get sliced image etc.
      self.set_imageaxes(self.axperm[0], self.axperm[1], self.slicepos)
      self.aspectratio = None
      self.figsize = None



   def globalminmax(self):
      """
      Get minimum and maximum value of data in entire data structure
      defined by the current FITS header. These values can be important if
      you want to compare different images from the same source
      (e.g. channel maps in a radio data cube).

      :Returns:
         min, max, two floating point numbers representing the minimum
         and maximum data value in data units of the header (*BUNIT*).
      """
      return self.dat.min(), self.dat.max()



   def printheader(self):
      #------------------------------------------------------------
      """
      Print the meta information from the selected header.
      Omit items of type *HISTORY*.

      :Returns:
         --

      :Examples:
         If you think a user needs more information from the header than
         can be provided with method :meth:`printaxisinfo` it can be useful to
         display the contents of the selected FITS header.

         >>> from kapteyn import maputils
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.printheader()
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
      for s in self.hdr.ascardlist():
         if not str(s).startswith('HISTORY'):
            print s



   def printaxisinfo(self):
      #------------------------------------------------------------
      """
      For each axis in the FITS header, print the data related
      to the World Coordinate System (WCS).

      :Returns:
         --

      :Examples:
         Print useful header information after the input of the FITS file
         and just before the specification of the image axes:

         >>> from kapteyn import maputils
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.printaxisinfo()
         Axis 1: RA---NCP  from pixel 1 to   100  {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
         Axis 2: DEC--NCP  from pixel 1 to   100  {crpix=51 crval=60.1539 cdelt=0.007166 (DEGREE)}
         Axis 3: VELO-HEL  from pixel 1 to   101  {crpix=-20 crval=-243 cdelt=4200 (km/s)}

      """
      #------------------------------------------------------------      
      for i in range(self.naxis):  # Note that the dictionary is unsorted. We want axes 1,2,3,...
         ax = i + 1
         a = self.axisinfo[ax]
         print "Axis %d: %-9s from pixel 1 to %5d  {crpix=%d crval=%G cdelt=%g (%s)}" % (
               a.axisnr, 
               a.axnamelong,
               a.axlen,
               a.crpix,
               a.crval,
               a.cdelt,
               a.cunit)



   def set_imageaxes(self, axnr1=None, axnr2=None, slicepos=None, spectrans=None, promptfie=None):
      #--------------------------------------------------------------
      """
      A FITS file can contain a data set of dimension n.
      If n < 2 we cannot display the data without more information.
      If n == 2 the data axes are those in the FITS file, Their numbers are 1 and 2.
      If n > 2 then we have to know the numbers of those axes that
      are part of the image. For the other axes we need to know a
      pixel position so that we are able to extract a data slice.

      Atribute :attr:`dat` is then always a 2D array.

      :param axnr1:
         Axis number of first image axis (X-axis)
      :type axnr1:
         Integer
      :param axnr2:
         Axis number of second image axis (Y-axis)
      :type axnr2:
         Integer
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
         A function, supplied by the user, that can
         prompt a user to enter the values for axnr1, axnr2
         and slicepos. An example of a function supplied by
         :mod:`maputils` is function :func:`getimageaxes`

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

      :Returns:
         --

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
                At these position a slice with image data is extracted.

          .. attribute:: map

                Image data from the selected FITS file. It is always a
                2D data slice and its size can be found in attribute
                :attr:`imshape`.

          .. attribute:: imshape

                The shape of the array :attr:`map`.

          .. attribute:: mixpix

                Images with only one spatial axis, need another spatial axis
                to produces useful world coordinates. This attribute is
                extracted from the relevant axis in attribute ;attr:`slicepos`.

          .. attribute:: convproj

                An object from class Projection as defined in :mod:`wcs`.

          .. attribute:: axperm

                The axis numbers corresponding with the X-axis and Y-axis in
                the image.

      :Notes:
         The aspect ratio is reset (to *None*) after each call to this method.

      :Examples:
         Set the image axes explicitly:

         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(1,2, slicepos=30)

         Set the images axes in interaction with the user using
         a prompt function:

         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(promptfie=maputils.getimageaxes)

      """
      #-----------------------------------------------------------------
      n = self.naxis
      self.spectrans = spectrans      # Set the spectral translation
      if n >= 2:
         if (axnr1 == None or axnr2 == None) and promptfie == None:
            if (axnr1 == None and axnr2 == None):
               axnr1 = self.axperm[0]
               axnr2 = self.axperm[1]
            else:
               raise Exception, "One axis number is missing and no prompt function is given!"
         if slicepos == None and promptfie == None:
            slicepos = self.slicepos

      # If a spectral axis is found, make a list with allowed spectral transformations
      proj = wcs.Projection(self.hdr)
      allowedtrans = []
      for i in range(n):
         ax = i + 1
         self.axisinfo[ax].wcstype = proj.types[i]
         self.axisinfo[ax].wcsunits = proj.units[i]
         self.axisinfo[ax].cdelt = proj.cdelt[i]
         if proj.types[i] == 'spectral':
            stypes = ['FREQ', 'ENER', 'WAVN', 'VOPT', 'VRAD', 'VELO', 'WAVE', 'ZOPT', 'AWAVE', 'BETA']
            convs = ['F', 'W', 'V', 'A']
            for t in stypes:
               try:
                  proj.spectra(t)
                  allowedtrans.append(t)
               except:
                  pass
               for c1 in convs:
                  for c2 in convs:
                     if c1 != c2:
                        spectrans = '%s-%s2%s' % (t, c1, c2)
                        try:
                           proj.spectra(spectrans)
                           allowedtrans.append(spectrans)
                        except:
                           pass
      self.allowedtrans = allowedtrans

      if promptfie != None:
         axnr1, axnr2, self.slicepos, self.spectrans = promptfie(self, axnr1, axnr2)
      else:
         if slicepos != None and n > 2:
            if type(slicepos) in sequencelist:
               self.slicepos = slicepos
            else:
               self.slicepos = [slicepos]
         
      axperm = [axnr1, axnr2]
      wcsaxperm = [axnr1, axnr2]
      # User could have changed the order of which these axes appear 
      # in the FITS header. Sort them to assure the right order.
      axperm.sort()

      # print "LEN slicepos=",len(slicepos), slicepos
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
         # You can reshape with the shape attribute or with NumPy's squeeze method.
         # With shape: dat.shape = (n1,n2)
         # With squeeze: dat = numpy.squeeze( dat )
         self.map = self.dat[sl].squeeze()
      else:
         self.map = self.dat
      self.imshape = self.map.shape
      if axperm[0] != wcsaxperm[0]:
         # The x-axis should be the y-axis vv.
         self.map = numpy.swapaxes(self.map, 0,1)   # Swap the x- and y-axes
         axperm[0] = wcsaxperm[0]     # Return the original axis permutation array
         axperm[1] = wcsaxperm[1]
         self.imshape = self.map.shape

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
         if ax1 == proj.lonaxnum and ax2 != proj.lataxnum:
            matchingaxnum = proj.lataxnum
            mix = True
         elif ax1 == proj.lataxnum and ax2 != proj.lonaxnum:
            matchingaxnum = proj.lonaxnum
            mix = True
         if ax2 == proj.lonaxnum and ax1 != proj.lataxnum:
            matchingaxnum = proj.lataxnum
            mix = True
         elif ax2 == proj.lataxnum and ax1 != proj.lonaxnum:
            matchingaxnum = proj.lonaxnum
            mix = True
      if mix:
         if matchingaxnum != None:
            self.mixpix = self.axisinfo[matchingaxnum].outsidepix
            ap = (axperm[0], axperm[1], matchingaxnum)
         else:
            raise Exception, "Cannot find a matching axis for the spatial axis!"
      else:
          ap = (axperm[0], axperm[1])

      p1 = proj.sub(ap)    # To do straight conversions (x,y,mixpix) -> (xw,yw,dummy)
      if self.spectrans != None:
         self.convproj = p1.spectra(self.spectrans)
      else:
         self.convproj = p1

      self.axperm = wcsaxperm        # We need only the numbers of the first two axes
      self.aspectratio = None        # Reset the aspect ratio because we could have another image now



   def createMPLimage(self, fig):
      #---------------------------------------------------------------------
      """
      This method couples the data slice that represents an image to
      a Matplotlib Axes object (parameter *frame*). It returns an object
      from class :class:`MPLimage` which has only attributes relevant for
      Matplotlib.

      :param frame:
         Plot the current image in this Matplotlib Axes object.
      :type frame:
         A Matplotlib Axes instance

      :Returns:
         An object from class :class:`MPLimage`
      """
      #---------------------------------------------------------------------
      ar = self.get_pixelaspectratio()
      mplimage = MPLimage(fig, self.pxlim, self.pylim, self.map, self.convproj,
                          mixpix=self.mixpix, slicepos=self.slicepos, pixelaspectratio=ar)
      return mplimage



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
         :mod:`maputils` is function :func:`getbox`
      :type promptfie:
         Python function

      :Returns:
         --

      :Notes:
         --

      :Examples: Ask user to enter limits with prompt function :func:`getbox`
         
         >>> fitsobject = maputils.FITSimage('rense.fits')
         >>> fitsobject.set_imageaxes(1,2, slicepos=30) # Define image in cube
         >>> fitsobject.set_limits(promptfie=maputils.getbox)
      """
      #---------------------------------------------------------------------
      n1 = self.axisinfo[self.axperm[0]].axlen
      n2 = self.axisinfo[self.axperm[1]].axlen
      if pxlim == None:
         pxlim = [1, n1]
      if pylim == None:
         pylim = [1, n2]
      if promptfie != None:
         axname1 = self.axisinfo[self.axperm[0]].axname
         axname2 = self.axisinfo[self.axperm[1]].axname
         pxlim, pylim = promptfie(self.pxlim, self.pylim, axname1, axname2)
      # Check whether these values are within the array limits
      if pxlim[0] < 1:  pxlim[0] = 1
      if pxlim[1] > n1: pxlim[1] = n1
      if pylim[0] < 1:  pylim[0] = 1
      if pylim[1] > n2: pylim[1] = n2
      # Get the subset from the (already) 2-dim array 
      self.map = self.map[pylim[0]-1:pylim[1], pxlim[0]-1:pxlim[1]]       # map is a subset of the original (squeezed into 2d) image
      # self.map = numpy.ma.masked_where(numpy.isnan(z), z)
      self.imshape = self.map.shape
      self.pxlim = pxlim
      self.pylim = pylim



   def get_pixelaspectratio(self):
   #---------------------------------------------------------------------
      """
      Return the aspect ratio of the pixels in the current
      data structure defined by the two selected axes.
      The aspect ratio is defined as *pixel height / pixel width*.
      """
   #---------------------------------------------------------------------
      a1 = self.axperm[0]; a2 = self.axperm[1];
      cdeltx = self.axisinfo[a1].cdelt
      cdelty = self.axisinfo[a2].cdelt
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      aspectratio = abs(cdelty/cdeltx)
      if aspectratio > 10.0 or aspectratio < 0.1:
         aspectratio = nx/ny
      return aspectratio



   def set_aspectratio(self, aspectratio=None, xcm=None, ycm=None):
      #---------------------------------------------------------------------
      """
      This method knows about the sizes of a pixel.
      It can suggest a figure size for a Matplotlib canvas so that
      pixels are displayed in the ratio of the height and the width
      in world coordinates,
      i.e. pixels are displayed as rectangles if the width is
      unequal to the height.
       
      :param aspectratio:
         Defined as pixel height / pixel width. With this parameter you 
         can set the aspect ratio of your image. If the value is *None*
         the the default is the aspect ratio that preserves equal distances
         in world coordinates. 
      :type aspectratio:
         Floating point number
      :param xcm:
         Find a value for *ycm* so that the aspect ratio follows the size of
         a pixel, corrected for the number of pixels in x and y.
         If there are more pixels in the x- direction then we expect
         a smaller value for the height in *ycm* assuming that the
         aspect ratio was 1.
         If both parameters *xcm* and *ycm* have a value, then thse
         values are unaltered. 
      :type xcm:
        Floating point number
      :param ycm:
         Find a value for *xcm* so that the aspect ratio follows the size of
         a pixel, corrected for the number of pixels in x and y.
         If there are more pixels in the y- direction then we expect
         a smaller value for the height in *xcm* assuming that the
         aspect ratio was 1.
      :type ycm:
        Floating point number

      :Returns:
         The calculated aspect ratio = pixel height / pixel width

      :Attributes:
                        
         .. attribute:: aspectratio
   
               The ratio between pixel height and pixel width
   
         .. attribute:: figsize
   
               The suggested figure size to be used in Matplotlib.

      :Notes:
         The aspect ratio is reset after a call to :meth:`set_imageaxes`.
         
      :Example:
      
         In the next example we demonstrate the use of method :meth:`set_aspectratio`.
         We want the image pixel plotted with the correct sizes in world coordinates
         and use *set_aspectratio()* to set Matplotlib's Axes object in the correct
         aspect ratio.

         .. literalinclude:: maputils.aspectratio.1.py

      """
      #---------------------------------------------------------------------
      a1 = self.axperm[0]; a2 = self.axperm[1];
      cdeltx = self.axisinfo[a1].cdelt
      cdelty = self.axisinfo[a2].cdelt
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      if xcm == None and ycm == None:
         xcm = 25.0
      if aspectratio == None:
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
      fh = 0.8; fw = 0.6
      # self.axesrect = (0.2, 0.1, fw, fh)
      self.figsize = (xcm/2.54, ycm/2.54)
      self.aspectratio = aspectratio
      return aspectratio



class ImageContainer(object):
#-----------------------------------------------------------------
   """
This class is a container for objects from class :class:`MPLimage`.
The objects are stored in a list. The class has methods that 
act on all or single objects. A toggle which reacts on the keyboard keys
'*,*' and '*.*' sets the visibility of images. This property 
allows a programmer to implement a simple movie loop over
images stored in the container.
   
**Parameters**:
   --

:Returns:
   --

:Attributes:
   
    .. attribute:: mplim

       List with objects from class :class:`MPLimage`.
       
    .. attribute:: indx
    
       Index in list of objects which represents the current image.
       
:Notes:
   --
   

:Examples:
   Use of this class as a container for images in a movie loop:

   .. literalinclude:: maputils.movie.1.py


:Methods:

.. automethod:: append
.. automethod:: movie_events
.. automethod:: toggle_images

   """
#--------------------------------------------------------------------
   def __init__(self):
      self.mplim = []                            # The list with MPLimage objects
      self.indx = 0                              # Sets the current image in the list
      self.fig = None                            # Current Matplotlib figure instance
      self.textid = None                         # Plot and erase text on canvas using this id.



   def append(self, imobj, visible=True, schedule=True):
      """
      Append MPLimage object. First there is a check for the class
      of the incoming object. If it is the first object that is appended then
      from this object the Matplotlib figure instance is copied.
      
      :param imobj:
         Add an image object to the list.
      :type imobj:
         An object from class :class:`MPLimage`.
          
      :Returns:
          --
      :Notes:
          --
      """
      if not isinstance(imobj, MPLimage):
         raise TypeError, "Container object not of class maputils.MPLimage!" 
      if len(self.mplim) == 0:                   # This must be the first object in the container
         self.fig = imobj.fig
      imobj.im.set_visible(visible)
      imobj.motion_events()
      imobj.key_events()
      if not schedule:
         imobj.cidmove.deschedule()
         imobj.cidkey.deschedule()
      self.mplim.append(imobj)



   def movie_events(self):
      """
      Connect keys for movie control
      The keys to step through your images are ',' and '.'
      (these keys also show the characters '<' ans '>' for 
      backwards and forwards in list)
      
      :Returns:
          --
      :Notes:
          --
      """
      if self.fig == None:
         raise Exception("No matplotlib.figure instance available!")
      if len(self.mplim) == 0:
         raise Exception("No objects in container!")
      self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.toggle_images)
      self.textid = self.fig.text(0.01, 0.95, "Use keys ',' and '.' to loop through list with images", color='g', fontsize=8)


   def toggle_images(self, event):
       """
       Toggle the visible state of images.
       This toggle works if one stacks multiple image in one frame.
       Only one image gets status visible=True. The others
       are set to visible=False. This toggle changes this visibility
       for images and the effect, if you keep pressing the right keys,
       is a movie.
       
       The keys to step through your images are ',' and '.'
       (these keys also show the characters '<' ans '>' for 
       backwards and forwards in the list of images).

       :param event:
          Mouse event object with pixel position information.
       :type event:
          matplotlib.backend_bases.MouseEvent instance
          
       :Returns:
          --
       
       :Notes:
          Changes in color map or color limits are applied to other
          images e.g. in a movie loop, only if the image becomes the current image.
          This causes some delay when you change colors while the movie is running.
          
       """
       if event.key.upper() not in [',','.']: 
          return
       oldim = self.mplim[self.indx] 
       oldim.im.set_visible(False)
       oldim.cidmove.deschedule()
       oldim.cidkey.deschedule()
       if event.key.upper() == '.': 
          if self.indx + 1 >= len(self.mplim):
             self.indx = 0
          else:
             self.indx += 1
       elif event.key.upper() == ',': 
          if self.indx - 1 <= 0:
             self.indx = len(self.mplim) - 1
          else:
             self.indx -= 1
    
       newindx = self.indx
       newim = self.mplim[newindx]
       slicepos = str(newim.slicepos)
       newim.im.set_visible(True)
       if newim.clipmin != oldim.clipmin or newim.clipmax != oldim.clipmax: 
          newim.im.set_clim(oldim.clipmin, oldim.clipmax)
          newim.clipmin = oldim.clipmin
          newim.clipmax = oldim.clipmax
       if not (oldim.im.get_cmap() is newim.im.get_cmap()):
          newim.im.set_cmap(oldim.im.get_cmap())
          newim.cmindx = oldim.cmindx
       newim.cidmove.schedule()
       newim.cidkey.schedule()
       
       if self.textid != None:
          self.textid.set_text("im #%d slice:%s"%(newindx, slicepos))
       #newim.frame.set_xlabel("%d"%self.indx)
       self.fig.canvas.draw()

