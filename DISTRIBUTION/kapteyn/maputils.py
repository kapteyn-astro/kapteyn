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
tools to make plots (with wcs annotation) of image data from FITS files.
These tools are based on the functionality of PyFITS and Matplotlib.
The methods from these packages are mofified in *maputils* for an optimal
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

Of course there are many programs that can do this job. But most probably
no program does it exactly the way you want or it cannot write
a hard copy with sufficient quality to publish. Also you cannot change,
or add your own extensions as easy as with this module :mod:`maputils`.

Module :mod:`maputils` is also very useful as a tool to extract and plot
data slices from data sets with more than two axes. For example it can plot
so called *Position-Velocity* maps from a radio interferometer data cube
with channel maps. It can annotate these plots with the correct WCS annotation using
information about the 'missing' spatial axis.

To facilitate the input of the correct data to open a FITS image,
to specify the right data slice or to set the pixel limits for the
image axes, we implemented also some helper functions.
These functions are primitive (terminal based) but effective. You are invited to
replace them by enhanced versions, perhaps with a graphical user interface.

Here is an example of what you can expect. We have a three dimensional dataset
on disk called *ngc6946.fits* with axes RA, DEC and VELO. The image below
is a data slice in RA, DEC at VELO=50. Its data limits are set to [10,90, 10,90]
We changed interactively the color map (keys *pageup/pagedown*)
and the color limits (pressing right mouse button while moving mouse) and saved
a hard copy on disk.

.. literalinclude:: EXAMPLES/mu_introduction.py
   
.. image:: EXAMPLES/mu_introduction.png
   :width: 700
   :align: center
   
   
.. centered:: Image from FITS file with graticules and WCS labels

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


Class FITSaxis
--------------

.. autoclass:: FITSaxis

Class MovieContainer
--------------------

.. autoclass:: MovieContainer

"""
# In case we want to use the plot directive, we have an exampe here
# .. plot:: /Users/users/vogelaar/MAPMAKER/maputils.intro.1.py
#

#from matplotlib import use
#use('qt4agg')

from matplotlib.pyplot import setp as plt_setp,  get_current_fig_manager as plt_get_current_fig_manager
from matplotlib import cm
from matplotlib.colors import Colormap, Normalize          #, LogNorm, NoNorm
from matplotlib.colorbar import make_axes, Colorbar, ColorbarBase
import matplotlib.nxutils as nxutils
import pyfits
import numpy
from kapteyn import wcs, wcsgrat
from kapteyn.celestial import skyrefsystems, epochs, skyparser
from kapteyn.tabarray import tabarray
#from kapteyn import mplutil
from kapteyn.mplutil import AxesCallback, VariableColormap, TimeCallback, KeyPressFilter
import readline
from types import TupleType as types_TupleType
from types import ListType as types_ListType
from types import StringType as types_StringType 
from string import upper as string_upper
from string import letters
from re import split as re_split


KeyPressFilter.allowed = ['f', 'g']


sequencelist = (types_TupleType, types_ListType)

__version__ = '1.0'

(left,bottom,right,top) = (wcsgrat.left, wcsgrat.bottom, wcsgrat.right, wcsgrat.top)                 # Names of the four plot axes
(native, notnative, bothticks, noticks) = (wcsgrat.native, wcsgrat.notnative, wcsgrat.bothticks, wcsgrat.noticks) 


class Colmaplist(object):
   def __init__(self):
      # A list with available Matplotlib color maps
      # The '_r' entries are reversed versions
      self.colormaps = sorted([m for m in cm.datad.keys() if not m.endswith("_r")])
   def add(self, clist):
      if type(clist) not in sequencelist:
         clist = [clist]
      for c in clist[::-1]:
         self.colormaps.insert(0,c)

cmlist = Colmaplist()
colormaps = cmlist.colormaps


def prompt_box(pxlim, pylim, axnameX, axnameY):
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



def prompt_fitsfile(defaultfile=None, hnr=None, alter=None, memmap=None):
#-----------------------------------------------------------------
   """
An external helper function for the FITSimage class to
prompt a user to open the right Header Data Unit (hdu)
of a FITS file.
A programmer can supply his/her own function as long
as the parameters that are returned are 
the hdu list, the header unit number, the filename and a character for 
the alternate header.
   
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

      If a FITS file has more than one header, one must decide
      which header contains the required image data.
      
:Returns:

   * *hdulist* - The HDU list and the user selected index of the wanted 
     hdu from that list. The HDU list is returned so that it
     can be closed in the calling environment.
   * *hnr* - FITS header number. Usually the first header, i.e. *hnr=0*
   * *fitsname* - Name of the FITS file.
   * *alter* - A character that corresponds to an alternate header
     (with alternate wcs information e.g. a spectral translation).
     
:Notes:
   --
   
:Examples:  
   Besides file names of files on disk, PyFITS allows url's and gzipped 
   files to retreive FITS files e.g.::
   
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
      except KeyboardInterrupt:
         raise
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

   # If there is no character given for an alternate header
   # but an alternate header is detected, then the user is
   # prompted to enter a character from a list with allowed
   # characters. Currently an alternatre header is found if
   # there is a CRPIX1 followed by a character A..Z
   if alter == None:
      alternates = []
      hdr = hdulist[hnr].header
      for a in letters[:26]:
         k = "CRPIX1%c" % a.upper()  # To be sure that it is uppercase
         if hdr.has_key(k):
            print "Found alternate header:", a.upper()
            alternates.append(a)
   
      alter = ''
      if len(alternates):
         while True:
            p = raw_input("Enter char. for alternate header: ...... [No alt. header]:")
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



def prompt_imageaxes(fitsobj, axnum1=None, axnum2=None):
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

:Example:
   Interactively set the axes of an image using a prompt function::
   
      # Create a maputils FITSimage object from a FITS file on disk
      fitsobject = maputils.FITSimage('rense.fits')
      fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
   
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

   return axnum1, axnum2, slicepos



def prompt_spectrans(fitsobj):
#-----------------------------------------------------------------------
   """
Ask user to enter spectral translation if one of the axes is spectral.

:param fitsobj:
   An object from class FITSimage. This prompt function must
   have knowledge of this object such as the allowed spectral
   translations.
:type fitsobj:
   Instance of class FITSimage

:Returns: 
   * *spectrans* - The selected spectral translation from a list with spectral
     translations that are allowed for the input object of class FITSimage.
     A spectral translation translates for example frequencies to velocities.

   """
#-----------------------------------------------------------------------
   asktrans = False
   for ax in fitsobj.axperm:
      if fitsobj.axisinfo[ax].wcstype == 'spectral':
         asktrans = True

   spectrans = None
   nt = len(fitsobj.allowedtrans)
   if (nt > 0 and asktrans):
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
      An object from class FITSimage. This prompt function must
      have knowledge of this object such as the allowed spectral
      translations.
   :type fitsobj:
      Instance of class FITSimage

   :Returns:
      * *skyout* - The sky definition to which positions in the native system
        will be transformed.
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
      prompt = "Ref.sys 0=fk4, 1=fk4_no_e, 2=fk5, 3=icrs, 4=j2000 ... [native]: "
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

   if skysys == None:
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


def gauss_kern(size, sizey=None):
     """ Returns a normalized 2D gauss kernel array for convolutions """
     size = int(size)
     if not sizey:
         sizey = size
     else:
         sizey = int(sizey)
     x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
     g = numpy.exp(-(x**2/float(size)+y**2/float(sizey)))
     return g / g.sum()


def blur_image(self, n, ny=None) :
   """ blurs the image by convolving with a gaussian kernel of typical
         size n. The optional keyword argument ny allows for a different
         size in the y direction.
   """
   if self.data == None:
      raise Exception, "Cannot plot image because image data is not available!"
   g = gauss_kern(n, sizey=ny)
   self.data = numpy.convolve(self.data, g, mode='valid')
   self.datmin = self.data.min()
   self.datmax = self.data.max()
   u = {'vmin':self.datmin, 'vmax':self.datmax}
   self.kwargs.update(u)


class Image(object):
   #--------------------------------------------------------------------
   """
   Prepare the FITS- or external image data to be plotted in Matplotlib
   All parameters are set by method :meth:`Annotatedimage.Image`.
   The keyword arguments are those for Matplotlib's method *imshow()*.
   Two of them are useful in the context of this class. These parameters
   are *visible* a boolean to set the visibility of the image to on or off,
   and *alpha*, a number between 0 and 1 which sets the transparancy of
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
      newkwargs = ({'aspect':None, 'origin':'lower', 'extent':self.box, 'norm':norm,
                    'interpolation':'nearest'})
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
      :meth`Annotatedimage.plot` but it can also be used separately.

      :Example:

      >>> mplim = fitsobject.Annotatedimage(frame)
      >>> frame = fig.add_subplot(1,1,1)
      >>> grat = mplim.Graticule()
      >>> grat.plot(frame)

      """
      #--------------------------------------------------------------------
      if self.data == None:
         raise Exception, "Cannot plot image because image data is not available!"
      self.frame = frame
      self.im = self.frame.imshow(self.data, cmap=self.cmap, **self.kwargs)
      self.frame.set_xlim((self.box[0], self.box[1]))
      self.frame.set_ylim((self.box[2], self.box[3]))




class Contours(object):
   #--------------------------------------------------------------------
   """
   objects from this class calculate and plot contour lines.
   Most of the parameters are set by method
   :meth:`Annotatedimage.Contours`. The others are:

   
   :param filled:
      If True, then first create filled contours and draw
      the contours lines upon these filled contours
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
      :meth`Annotatedimage.plot` but it can also be used separately.
      """
      #--------------------------------------------------------------------
      if self.data == None:
         raise Exception, "Cannot plot image because image data is not available!"
      self.frame = frame
      if self.clevels == None:
         if self.filled:
            self.frame.contourf(self.data, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.CS = self.frame.contour(self.data, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.clevels = self.CS.levels
      else:
         if type(self.clevels) not in sequencelist:
            self.clevels = [self.clevels]
         if self.filled:
            self.frame.contourf(self.data, self.clevels, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.CS = self.frame.contour(self.data, self.clevels, cmap=self.cmap, norm=self.norm, **self.kwargs)
         self.clevels = self.CS.levels
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
               # print "lev=", lev
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
      
      >>> cont = mplim.Contours(levels=range(10000,16000,1000))
      >>> cont.setp_contour(linewidth=1)
      >>> cont.setp_contour(levels=11000, color='g', linewidth=3)
      """
      #--------------------------------------------------------------------
      if levels == None:
         self.commoncontourkwargs = kwargs
      else:
         if self.clevels == None:
            raise Exception, "Contour levels not set so I cannot identify contours"
         # Change only contour properties for levels in parameter 'levels'.
         if type(levels) not in sequencelist:
            levels = [levels]
         for lev in levels:
            try:
               i = list(self.clevels).index(lev)
            except ValueError:
               i = -1 # no match
            if i != -1:
               self.ckwargslist[i] = kwargs


   def setp_label(self, levels=None, **kwargs):
      #--------------------------------------------------------------------
      """
      Set properties for the labels along the contours.

      :Examples;

      >>> cont2 = mplim.Contours(levels=(8000,9000,10000,11000))
      >>> cont2.setp_label(11000, colors='b', fontsize=14, fmt="%.3f")
      >>> cont2.setp_label(fontsize=10, fmt="$%g \lambda$")
      """
      #--------------------------------------------------------------------
      if levels == None:
         self.commonlabelkwargs = kwargs
      else:
         if self.clevels == None:
            raise Exception, "Contour levels not set so I cannot identify contours"
         # Change only contour properties for levels in parameter 'levels'.
         if type(levels) not in sequencelist:
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
   Colorbar class. Usually the parameters will be provide by method
   :meth:`Annotatedimage.Colorbar`
   """
   #--------------------------------------------------------------------
   def __init__(self, cmap, norm=None, contourset=None, clines=False, fontsize=9,
                label=None, linewidths=None, **kwargs):
      #--------------------------------------------------------------------
      #--------------------------------------------------------------------
      self.ptype = "Colorbar"
      self.cmap = cmap
      self.norm = norm
      self.contourset = contourset
      self.plotcontourlines = clines
      self.cbfontsize = fontsize
      self.linewidths = linewidths
      self.label = label
      newkwargs = ({'orientation':'vertical'})
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


   def plot(self, frame, cbframe):
      #--------------------------------------------------------------------
      """
      Plot image object. Usually this is done by method
      :meth`Annotatedimage.plot` but it can also be used separately.
      """
      #--------------------------------------------------------------------
      # ColorbarBase needs a norm instance
      if self.plotcontourlines and self.contourset != None:
         CS = self.contourset.CS
         if not self.kwargs.has_key("ticks"):
            self.kwargs["ticks"] = CS.levels
      else:
          CS = None

      self.cb = ColorbarBase(cbframe, cmap=self.cmap, norm=self.norm, **self.kwargs)
      # User requires lines (corresponding to contours) in colorbar
      if CS != None:
         if self.linewidths != None:
            tlinewidths = [self.linewidths]*len(CS.tlinewidths)
         else:
            tlinewidths = [t[0] for t in CS.tlinewidths]
         tcolors = [c[0] for c in CS.tcolors]
         self.cb.add_lines(CS.levels, tcolors, tlinewidths)
      self.colorbarticks()    # Set font size given in kwargs or use default
      if self.label != None:
         self.cb.set_label(self.label)


class Annotatedimage(object):
#--------------------------------------------------------------------
   """
This is one of the core classes of this module. It sets the connection
between the Matplotlib independent FITS data and the routines that
do the actual plotting with Matplotlib.
The class is usually used in the context of class :class:`FITSimage` which
has a method that prepares the parameters for the constructor of
this class.

:param frame:
   This the frame where image and or contours will be plotted.
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
   with conversion methods from :mod"`wcs` like
   :meth:`wcs.Projection.toworld()` and :meth:`wcs.Projection.topixel()`
   needed for conversions between pixel- and world coordinates.
:type projection:
   Instance of Projection class from module :mod:`wcs`
:param axperm:
   Tuple or list with the FITS axis number of the two image axes,
   e.g. axperm=(1,2)
:type axperm:
   Tuple with integers
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
:param filename:
   Base name for new files on disk, for example to store a color map
   on disk. The default is supplied by method :meth:`FITSimage.Annotatedimage`.
:type filename:
   string
:param cmap:
   A colormap from class :class:`mplutil.VariableColormap` or a string
   that represents 
:type cmap:
   mplutil.VariableColormap instance
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

:Methods:

.. automethod:: set_norm
.. automethod:: write_colormap
.. automethod:: set_colormap
.. automethod:: set_aspectratio
.. automethod:: Image
.. automethod:: Contours
.. automethod:: Colorbar
.. automethod:: plot
.. automethod:: interact_toolbarinfo
.. automethod:: interact_imagecolors
.. automethod:: interact_writepos
.. automethod:: key_imagecolors
.. automethod:: mouse_imagecolors

:Attributes:

    .. attribute:: frame

          Matplotlib Axes instance where image and contours are plotted

    .. attribute:: data

          Image data

    .. attribute:: mixpix

          The pixel of the missing spatial axis in a Position-Velocity
          image

    .. attribute:: projection

          An object from the Projection class as defined in module :mod:`wcs`

    .. attribute:: skyout

          The sky definition for which graticule lines are plotted
          and axis annotation is made (e.g. "Equatorial FK4")

    .. attribute:: spectrans

          The translation code to transform native spectral coordinates
          to another system (e.g. frequencies to velocities)

    .. attribute:: pxlim

          Pixel limits in x = (xlo, xhi)

    .. attribute:: pylim

          Pixel limits in y = (ylo, yhi)

    .. attribute:: slicepos

          Single value or tuple with more than one values representing
          the pixel coordinates on axes in the original data structure
          that do not belong to the image. It defines how the data slice
          is ectracted from the original.
          The order of these 'outside' axes is copied from the (FITS) header.
          
    .. attribute:: aspect

          Aspect ratio of a pixel according to the FITS header.
          For spatial maps this value is used to set and keep an
          image in the correct aspect ratio.

    .. attribute:: cmap

           The color map. This is an object from class :class:`mplutil.VariableColormap`.
           which is inherited from the Matplotlib color map class.
           Its main methods are:
   
           * :class:`mplutil.VariableColormap.set_source`
           * :class:`mplutil.VariableColormap.set_bad`
           * :class:`mplutil.VariableColormap.add_frame`
           * :class:`mplutil.VariableColormap.modify`
           * :class:`mplutil.VariableColormap.set_scale`
           * :class:`mplutil.VariableColormap.set_inverse`
           * :class:`mplutil.VariableColormap.update`

    .. attribute:: objlist

          List with all plot objects (image, contours, colour bar, graticules)
          for this annotated image object.

   """
#--------------------------------------------------------------------
   def __init__(self, frame, header, pxlim, pylim, imdata, projection, axperm, skyout, spectrans,
                mixpix=None, aspect=1, slicepos=None, basename=None,
                cmap='jet', blankcolor='w', clipmin=None, clipmax=None):
      #-----------------------------------------------------------------
      """
      """
      #-----------------------------------------------------------------
      self.ptype = "Annotatedimage"
      self.hdr = header
      self.data = imdata
      self.mixpix = mixpix
      self.projection = projection
      self.axperm = axperm
      self.pxlim = pxlim
      self.pylim = pylim
      self.skyout = skyout
      self.spectrans = spectrans
      self.box = (self.pxlim[0]-0.5, self.pxlim[1]+0.5, self.pylim[0]-0.5, self.pylim[1]+0.5)
      self.image = None                          # A Matplotlib instance made with imshow()
      self.aspect = aspect
      self.slicepos = slicepos                   # Information about current slice 
      self.contours = None
      self.colorbar = None
      self.contourset = None
      self.objlist = []
      self.frame = self.adjustframe(frame)
      self.figmanager = plt_get_current_fig_manager()
      # Related to color maps:
      self.set_colormap(cmap)
      self.set_blankcolor(blankcolor)
      # Calculate defaults for clips if nothing is given
      self.clipmin = clipmin
      if self.data != None:
         if self.clipmin == None:                   # Take care of -inf, +inf & NaN
            self.clipmin = imdata[numpy.isfinite(imdata)].min()  
         self.clipmax = clipmax
         if self.clipmax == None:
            self.clipmax = imdata[numpy.isfinite(imdata)].max()
      else:
         self.clipmin = 0.0
         self.clipmax = 1.0
      self.norm = Normalize(vmin=self.clipmin, vmax=self.clipmax)
      self.histogram = False                     # Is current image equalized?
      self.data_hist = None                      # There is not yet a hist. eq. version of the data
      self.data_orig = self.data                 # So we can toggle between image versions
      if basename == None:
         self.basename = "Unknown"               # Default name for file with colormap lut data
      else:
         self.basename = basename


   def set_norm(self, clipmin, clipmax):
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
      """
      #-----------------------------------------------------------------
      # Here we need maintenance. The image is updated but the colorbar is not
      self.norm = Normalize(vmin=clipmin, vmax=clipmax)
      self.clipmin = clipmin; self.clipmax = clipmax
      print "In update norm:", self.clipmin, self.clipmax
      if self.cmap != None:
         self.cmap.update()
      if self.image.im != None:
         #self.image.im.set_norm(self.norm)
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

      If you add the event handler *interact_imagecolors()* it is
      possible to change colormaps with keyboard keys and mouse.
      """
      #-----------------------------------------------------------------
      if cmap == None:
         cmap = 'jet'
      if isinstance(cmap, Colormap):
         self.cmap = cmap                           # Either a string or a Colormap instance
         # What to do with the index. This is not a string from the list.
         self.cmindx = 0
      elif type(cmap) == types_StringType:
         try:
            # Is this colormap registered in our colormap list?
            self.cmindx = colormaps.index(cmap)
         except:
            # then register it now
            cmlist.add(cmap)
            self.cmindx = colormaps.index(cmap)
         self.cmap = VariableColormap(cmap)
      else:
         raise Exception, "Color map is not of type Colormap or string"
      if self.image != None:
         self.cmap.set_source(cmap)

      self.startcmap = self.cmap         # This could be one that is not in the list with color maps
      self.startcmindx = self.cmindx     # Use the start color map if a reset is requested
      self.cmapinverse = False
      if blankcolor != None:
         self.set_blankcolor(blankcolor)


   def write_colormap(self, filename):
      #-----------------------------------------------------------------
      """
      Method to write current colormap rgb values to file on disk.
      If you add the event handler *interact_imagecolors()*, this
      method is automatically invoked if you press key 'm'.
      """
      #-----------------------------------------------------------------
      print self.cmap.name
      print self.cmap._lut[0:-3,0:3]
      a = tabarray(self.cmap._lut[0:-3,0:3])
      a.writeto(filename)
      

   def set_blankcolor(self, blankcolor, alpha=1.0):
      #-----------------------------------------------------------------
      """
      Set the color of bad pixels.
      If you add the event handler *interact_imagecolors()*, this
      method steps throug a list of colors for the bad pixels in an
      image.
      """
      #-----------------------------------------------------------------
      self.cmap.set_bad(blankcolor, alpha)
      self.blankcol = blankcolor

      
   def set_aspectratio(self, aspect):
      #-----------------------------------------------------------------
      """
      Set the aspect ratio. Overrule the default aspect ratio which corrects
      pixels that are not square in world coordinates.
      """
      #-----------------------------------------------------------------
      self.aspect = abs(aspect)
      self.frame.set_aspect(aspect=self.aspect, adjustable='box', anchor='C')


   def adjustframe(self, frame):
      #-----------------------------------------------------------------
      """
      Method to change the frame for the right aspect ratio and
      how to react on an resize of the plot window.
      """
      #-----------------------------------------------------------------
      frame.set_aspect(aspect=self.aspect, adjustable='box', anchor='C')
      frame.set_autoscale_on(False)
      frame.xaxis.set_visible(False)
      frame.yaxis.set_visible(False)
      frame.set_xlim((self.box[0], self.box[1]))   # Initialize in case no objects are created
      frame.set_ylim((self.box[2], self.box[3]))   # then we still can navigate with the mouse
      return frame

      
   def histeq(self, nbr_bins=256):
      #-----------------------------------------------------------------
      """
      Create a histogram equalized version of the data.
      """
      #-----------------------------------------------------------------
      if self.data == None:
         raise Exception, "Cannot plot image because image data is not available!"
      # Algorithm by Jan Erik Solem
      im = self.data
      #im = numpy.ma.masked_where(numpy.isfinite(self.data), self.data)
      #get image histogram
      imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True, new=True,
                                    range=(self.clipmin, self.clipmax))
      cdf = imhist.cumsum() #cumulative distribution function
      #cdf = 255 * cdf / cdf[-1] #normalize
      cdf = cdf / cdf[-1] #normalize
      #use linear interpolation of cdf to find new pixel values
      im2 = numpy.interp(im.flatten(),bins[:-1],cdf)
      # Rescale between clip levels original data
      m = self.clipmax - self.clipmin
      im2 = im2 * m + self.clipmin
      self.data_hist = im2.reshape(im.shape)

      
   def Image(self, **kwargs):
      #-----------------------------------------------------------------
      """
      Setup of an image. This object has an attribute *im* which is
      an instance of Matplotlib's *imshow()* method.
      This object has a plot method. This method is used by the
      more general :meth:`Annotatedimage.plot` method.

      :Attributes:

         .. attribute:: im

            The object generated after a call to Matplotlib's *imshow()*.
      """
      #-----------------------------------------------------------------
      if self.image != None:
         raise Exception, "Only 1 image allowed per Annotatedimage object"
      image = Image(self.data, self.box, self.cmap, self.norm, **kwargs)
      self.objlist.append(image)
      self.image = image
      return image


   def Contours(self, levels=None, **kwargs):
      #-----------------------------------------------------------------
      """
      Setup of contours. Either as single contour lines or as a combination
      of contour lines and filled regions between contours.
      """
      #-----------------------------------------------------------------
      contourset = Contours(self.data, self.box, levels, cmap=self.cmap, norm=self.norm, **kwargs)
      self.objlist.append(contourset)
      self.contourset = contourset
      return contourset


   def Colorbar(self, clines=False, **kwargs):
      #-----------------------------------------------------------------
      """
      A color bar is an image which represents the current color scheme.
      It annotates the colors with image values so that it is possible
      to get an idea of the distribution of the values in your image.

      :param clines:
          If set to true AND a contour set (a :meth:`Contours` object)
          is available, then lines will be plotted in the color bar
          at positions that correspond to the contour levels
      :type clines:
          Boolean
      :param kwargs:
          Keyword arguments for Matplotlib's method *ColorbarBase()*

      :Notes:
          The labels
      
      """
      #------------------------------------------------------------------
      if self.colorbar != None:
         raise Exception, "Only 1 colorbar allowed per Annotatedimage object"
      colorbar = Colorbar(self.cmap, norm=self.norm, contourset=self.contourset, clines=clines, **kwargs)
      self.objlist.append(colorbar)
      self.colorbar = colorbar
      return colorbar


   def Graticule(self, visible=True, **kwargs):
      #-----------------------------------------------------------------
      """
      Calculate and plot graticule lines of constan longitude or constant
      latitude. The description of the parameters is found in
      :class:`wcsgrat.Graticule`. An extra parameter is *visible*.
      If visible is set to False than we can plot objects derived from this
      class such as 'Rulers' and 'Insidelabels' without plotting
      graticule lines and labels.

      Other parameters such as *hdr*, *axperm*, *pxlim*, *pylim*, *mixpix*,
      *skyout* and *spectrans* are set to defaults in the context of this method
      and should not to be supplied.
      """
      #-----------------------------------------------------------------
      class Gratdata(object):
         def __init__(self, hdr, axperm, pxlim, pylim, mixpix, skyout, spectrans):
            self.hdr = hdr
            self.axperm = axperm
            self.pxlim = pxlim
            self.pylim = pylim
            self.mixpix = mixpix
            self.skyout = skyout
            self.spectrans = spectrans

      gratdata = Gratdata(self.hdr, self.axperm, self.pxlim, self.pylim, self.mixpix, self.skyout, self.spectrans)
      graticule = wcsgrat.Graticule(graticuledata=gratdata, **kwargs)
      graticule.visible = visible     # A new attribute only for this context
      self.objlist.append(graticule)
      return graticule


   def Pixellabels(self, **kwargs):
      #-----------------------------------------------------------------
      """
      Set annotation along a plot axis to pixel coordinates.
      See also description at :class:`wcsgrat.Pixellabels`.
      """
      #-----------------------------------------------------------------
      pixlabels = wcsgrat.Pixellabels(self.pxlim, self.pylim, **kwargs)
      self.objlist.append(pixlabels)
      return pixlabels


   def plot(self):
      #-----------------------------------------------------------------
      """
      Plot all objects stored in the objects list for this Annotated image.
      """
      #-----------------------------------------------------------------
      needresize = False
      for obj in self.objlist:
         try:
            pt = obj.ptype
            if pt == "Colorbar":
               needresize = True
               orientation = obj.kwargs['orientation']
         except:
            raise Exception, "Unknown object. Cannot plot this!"

      if needresize:                             # because a colorbar must be included
         self.cbframe = make_axes(self.frame, orientation=orientation)[0]
         self.frame = self.adjustframe(self.frame)
      self.cmap.add_frame(self.frame)
      
      for obj in self.objlist:
         try:
            pt = obj.ptype
         except:
            raise Exception, "Unknown object. Cannot plot this!"
         if pt in ["Image", "Contour", "Graticule", "Pixellabels"]:
            obj.plot(self.frame)
            # If we want to plot derived objects (e.g. ruler) and not the graticule
            # then set visible to False in the constructor.
         elif pt == "Colorbar":
            obj.plot(self.frame, self.cbframe)



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
         >>> frame = fig.add_subplot(1,1,1)
         >>> mplim = fitsobject.Annotatedimage(frame)
         >>> mplim.toworld(51,-20)
            (-51.282084795899998, -243000.0, 60.1538880206)
         >>> mplim.topixel(-51.282084795899998, -243000.0)
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
            #if not numpy.ma.is_masked(self.boxdat[yi-1, xi-1]):
            if not numpy.isnan(self.data[yi-1, xi-1]):
               z = self.data[yi-1, xi-1]
               if missingspatial == None:
                  s = "x,y=%6.1f,%6.1f  wcs=%10f,%10f  Z=%+8.2e " % (x, y, xw, yw, z)
               else:
                  s = "x,y=%6.1f,%6.1f  wcs=%10f,%10f,%10f  Z=%+8.2e " % (x, y, xw, yw, missingspatial, z)
            else:
               if missingspatial == None:
                  s = "x,y=%6.1f,%6.1f  wcs=%10f,%10f  Z=NaN" % (x, y, xw, yw)
               else:
                  s = "x,y=%6.1f,%6.1f  wcs=%10f,%10f,%10f  Z=NaN" % (x, y, xw, yw, missingspatial)
         else: #except:
            s = "xp,yp: %.2f %.2f " % (x, y)
      return s


   def mouse_toolbarinfo(self, axesevent):
      #--------------------------------------------------------------------
      """
      *Display position information:*

      """
      #--------------------------------------------------------------------
      s = ''
      x, y = axesevent.xdata, axesevent.ydata
      if self.figmanager.toolbar.mode == '':
         s = self.positionmessage(x, y)
      if s != '':
         self.figmanager.toolbar.set_message(s)


   def interact_toolbarinfo(self):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
      It reacts on mouse movements. A message is prepared with position information
      in both pixel coordinates and world coordinates. The world coordinates
      are in the units given by the (FITS) header.

      :Notes:

         If a message does not fit in the toolbar then nothing is
         displayed. We don't have control over the maximum size of that message
         because it depends on the backend that is used (GTK, QT,...).
         If nothing appears, then a manual resize of the window will suffice.

      :Example: 

         Attach to an object from class :class:`Annotatedimage`:
      
         >>> mplim = f.Annotatedimage(frame)
         >>> mplim.interact_toolbarinfo()

         A more complete example::

            from kapteyn import maputils
            from matplotlib import pyplot as plt

            f = maputils.FITSimage("m101.fits")
            
            fig = plt.figure(figsize=(9,7))
            frame = fig.add_subplot(1,1,1)
            
            mplim = f.Annotatedimage(frame)
            ima = mplim.Image()
            mplim.Pixellabels()
            mplim.plot()
            
            mplim.interact_toolbarinfo()
            
            plt.show()

      """
      #--------------------------------------------------------------------
      self.toolbarkey = AxesCallback(self.mouse_toolbarinfo, self.frame, 'motion_notify_event')



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
         Given an object from class :class:`Annotatedimage` called *annimage*,
         register this callback function for the position information with:
   
         >>> annimage.interact_imagecolors()
   
      :Notes:
         
         *  New color limits are calculated as follows: first the position
            of the mouse is transformed into normalized coordinates.
            These values are used to set a shift and a compression factor
            for the color limits. The shift changes with horizontal moves
            and the compression with vertical moves.
            If the data range of image values is [*datmin*, *datmax*] then...

      """
      #--------------------------------------------------------------------      
      if axesevent.event.button == 3:
         x, y = axesevent.xdata, axesevent.ydata
         if self.image.im == None:                     # There is no image to adjust
            return
         # 1. event.xdata and event.ydata are the coordinates of the mouse location in
         # data coordinates (i.e. in screen pixels)
         # 2. transData converts these coordinates to display coordinates
         # 3. The inverse of transformation transAxes converts display coordinates to
         # normalized coordinates for the current frame.
         xy = self.frame.transData.transform((x,y))
         self.image.xyn_mouse = self.frame.transAxes.inverted().transform(xy)
         x , y = self.image.xyn_mouse
         slope = 3.0 * x; offset = y - 0.5
         self.cmap.modify(slope, offset)
         #if self.colorbar != None:
            # Next line seem to be necessary to keep the right
            # font size for the colorbar labels.
         #   self.colorbar.colorbarticks()
         

   def key_imagecolors(self, axesevent):
   #--------------------------------------------------------------------
      """
   This method catches keys which change the color setting of an image.
   These keys are *pageup* and *pagedown*, 'I', and 'R' and numbers 1 to 5.
   The page up/down keys move through a list
   with known color maps. You will see the results of a change immediately.

   Keys:
      
   * 'r' (or 'R') reset the colors to the original colormap and scaling.
     The default color map is called 'jet'.
   * 'i' (or 'I') toggles between inverse and normal scaling.
   * '1' sets the colormap scaling to linear
   * '2' sets the colormap scaling to logarithmic
   * '3' sets the colormap scaling to exponential
   * '4' sets the colormap scaling to square root
   * '5' sets the colormap scaling to square
   * 'b' (or 'B') changes color of bad pixels.
   * 'm' (or 'M') saves current colormap lut data to a file.
     The default name of the file is the name of file from which the data
     was extracted or the name given in the contructor. The name is
     appended with '.lut'. This data is written in the
     right format so that it can be be (re)used as input colormap.
     This way you can fix a color setting and reproduce the same setting
     in another run of
     a program that allows one to enter a colormap from file.
   * 'h' (or 'H') replaces the current data by a histogram equalized
     version of this data.
   
   :param axesevent:
      AxesCallback event object with pixel position information.
   :type axesevent:
      AxesCallback instance

   :Examples:
      If *image* is an object from :class:`Annotatedimage` then register
      this function with:

      >>> mplim = f.Annotatedimage(frame)
      >>> mplim.interact_imagecolors()
      """
   #--------------------------------------------------------------------
      scales = {'1': 'linear', '2': 'log', '3': 'exp', '4': 'sqrt', '5': 'square'}
 
      # Request for another color map with page up/down keys
      if axesevent.event.key in ['pageup', 'pagedown']:
         lm = len(colormaps)
         if axesevent.event.key == 'pagedown':
            self.cmindx += 1
            if self.cmindx >= lm:
               self.cmindx = 0
         if axesevent.event.key == 'pageup':
            self.cmindx -= 1
            if self.cmindx < 0:
               self.cmindx = lm - 1
         newcolormapstr = colormaps[self.cmindx]
         self.figmanager.toolbar.set_message(newcolormapstr)
         self.cmap.set_source(newcolormapstr)     # Keep original object, just change the lut
         #self.cmap.update()
      # Request for another scale, linear, logarithmic etc.
      elif axesevent.event.key in scales:
         key = axesevent.event.key 
         self.figmanager.toolbar.set_message(scales[key])
         self.cmap.set_scale(scales[key])
         mes = "Color map scale set to '%s'" % scales[key]
         self.figmanager.toolbar.set_message(mes)
      # Invert the color map colors
      elif axesevent.event.key.upper() == 'I':
         if self.cmapinverse:
            self.cmap.set_inverse(False)
            self.cmapinverse = False
            mes = "Color map not inverted"
         else:
            self.cmap.set_inverse(True)
            self.cmapinverse = True
            mes = "Color map inverted!"
         self.figmanager.toolbar.set_message(mes)
      # Reset all color map parameters
      elif axesevent.event.key.upper() == 'R':
         self.cmap.auto = False      # Postpone updates of the canvas.
         self.figmanager.toolbar.set_message('Reset color map to default')
         self.cmap.modify(1.0, 0.0)
         if self.cmapinverse:
            self.cmap.set_inverse(False)
            self.cmapinverse = False
         #colmap_start = colormaps[self.image.startcmap]
         self.cmap.set_source(self.startcmap)
         self.cmindx = self.startcmindx
         self.cmap.set_scale(scales['1'])
         self.cmap.auto = True
         self.cmap.update()          # Update all
      elif axesevent.event.key.upper() == 'C':
         cmin = self.clipmin + 200.0
         cmax = self.clipmax - 200.0
         self.set_norm(cmin, cmax)
      elif axesevent.event.key.upper() == 'B':
         # Toggle colors for bad pixels (blanks) 
         blankcols = ['w', 'k', 'y', 'm', 'c', 'r', 'g', 'b']
         try:
            indx = blankcols.index(self.blankcol)
         except:
            indx = 0
         if indx + 1 == len(blankcols):
            indx = 0
         else:
            indx += 1
         self.blankcol = blankcols[indx]
         self.cmap.set_bad(self.blankcol)
         mes = "Color of bad pixels changed to '%s'" % self.blankcol
         self.figmanager.toolbar.set_message(mes)
      elif axesevent.event.key.upper() == 'M':
         self.write_colormap(self.colormapfilename)
         mes = "Save color map to file [%s]" % self.colormapfilename
         self.figmanager.toolbar.set_message(mes)
      elif axesevent.event.key.upper() == 'H':
         # Set data to histogram equalized version
         if self.histogram:
            # Back to normal
            self.set_histogrameq(False)
            self.figmanager.toolbar.set_message('Original image displayed')
         else:
            if self.data_hist == None:
               self.figmanager.toolbar.set_message('Calculating histogram')
            self.set_histogrameq()
            self.figmanager.toolbar.set_message('Histogram eq. image displayed')


   def set_histogrameq(self, on=True):
      if not on:
         # Back to normal
         self.data = self.data_orig
         self.histogram = False
      else:
         if self.data_hist == None:
            self.datahist = self.histeq()
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


   def interact_imagecolors(self):
      #--------------------------------------------------------------------
      """
      Add mouse interaction (right mouse button) to change the colors
      in an image.
      
      Add key interaction to change or reset the colormap.
      See also description of the used keys at
      :meth:`Annotatedimage.key_imagecolors`

      For mouse interaction see documentation at:
      :meth:`Annotatedimage.mouse_imagecolors`

      If *mplim* is an object from class :class:`Annotatedimage` then activate
      color editing with:
      
      >>> mplim.interact_imagecolors()
      """
      #--------------------------------------------------------------------
      self.imagecolorskey = AxesCallback(self.key_imagecolors, self.frame, 'key_press_event')
      self.imagecolorsmouse = AxesCallback(self.mouse_imagecolors, self.frame, 'motion_notify_event')


   def mouse_writepos(self, axesevent):
      #--------------------------------------------------------------------
      """
      Print position information of the position where
      you clicked with the mouse. Print the info on the command line.
      Register this function with an event handler,
      see example.
   
      :param axesevent:
         AxesCallback event object with pixel position information.
      :type axesevent:
         AxesCallback instance
   
      :Example:
         Register this callback function for object *mplim* with:
   
         >>> image.interact_writepos()
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


   def interact_writepos(self):
      #--------------------------------------------------------------------
      """
      Add mouse interaction (left mouse button) to write the position
      of the mouse to screen. The position is written both in pixel
      coordinates and world coordinates.
      
      :Example:
      
      >>> mplim.interact_writepos()
      """
      #--------------------------------------------------------------------
      self.writeposmouse = AxesCallback(self.mouse_writepos, self.frame, 'button_press_event')


   def motion_events(self):
      #--------------------------------------------------------------------
      """
      Allow this :class:`Annotatedimage` object to interact with the user.
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
      Allow this :class:`Annotatedimage` object to interact with the user.
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
      Allow this :class:`Annotatedimage` object to interact with the user.
      It reacts on pressing the left mouse button and prints a message
      to stdout with information about the position of the cursor in pixels
      and world coordinates.
      """
      #--------------------------------------------------------------------
      # self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
      self.cidclick = AxesCallback(self.on_click, self.frame, 'button_press_event')



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
      if hdr.has_key(ax):
         self.ctype = hdr[ax].upper()
         self.axname = string_upper(hdr[ax].split('-')[0])
      ai = "NAXIS%d" % (axisnr,)
      self.axlen = hdr[ai]
      self.axisnr = axisnr
      self.axstart = 1
      self.axend = self.axlen
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
      ai = "CROTA%d" % (axisnr,)    # Not for all axes
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
      s = "axisnr     - Axis number: %s\n" % self.axisnr \
        + "axlen      - Length of axis in pixels (NAXIS): %s\n" % self.axlen \
        + "axstart    - First pixel coordinate: %s\n" % self.axstart \
        + "axend      - Last pixel coordinate: %s\n" % self.axend \
        + "ctype      - Type of axis (CTYPE): %s\n" % self.ctype \
        + "axname     - Short axis name: %s\n" % self.axname \
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
   Set the memory mapping for PyFITS. The default is in the PYFITS
   version we used was memory mapping set to off (i.e. memmap=0)
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
   should check is the shape of the numpy array fits the sizes given
   in FITS keywords *NAXISn*.
:type externaldata:
   Numpy array

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
       
    .. attribute:: boxdat

       The image data. Possibly sliced, axis swapped and limited in axis range.
       
    .. attribute:: imshape

       Sizes of the 2D array in :attr:`boxdat`.
       
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

       Plot a circle in **world coordinates** as a circle. That is, if the
       pixel size in the FITS header differs in X and Y, then correct
       the (plot) size of the pixels with value *aspectratio*
       so that features in an image have the correct sizes in longitude and
       latitude in degrees.
      
:Notes:
   The constructor sets also a default position for a data slice if
   the dimension of the FITS data is > 2. This position is either the value
   of CRPIX from the header or 1 if CRPIX is outside the range [1, NAXIS].

:Examples:
   PyFITS allows url's to retreive FITS files. It can also read gzipped files e.g.:
   
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
.. automethod:: globalminmax
.. automethod:: writetofits

   """

#--------------------------------------------------------------------
   def __init__(self, filespec=None, promptfie=None, hdunr=None, alter='', memmap=0,
                externalheader=None, externaldata=None):
      #----------------------------------------------------
      # Usually the required header and data are extracted
      # from a FITS file. But it is also possible to provide
      # header and data from an external source. Then these
      # must be processed instead of other keyword parameters
      #-----------------------------------------------------
      if externalheader != None:
         self.hdr = externalheader
         self.filename = "Header dictionary"
         self.dat = externaldata
      else:
         # Not an external header, so a file is given or user wants to be prompted.
         if promptfie:
            hdulist, hdunr, filename, alter = promptfie(filespec, hdunr, alter, memmap)
         else:
            try:
               hdulist = pyfits.open(filespec, memmap=memmap)
               filename = filespec
            except IOError, (errno, strerror):
               print "Cannot open FITS file: I/O error(%s): %s" % (errno, strerror)
               raise
            except:
               print "Cannot open file, unknown error!"
               raise
            if hdunr == None:
               hdunr = 0
         hdu = hdulist[hdunr]
         self.filename = filename
         self.hdr = hdu.header
         if externaldata == None:
            self.dat = hdu.data
         else:
            self.dat = externaldata
         hdulist.close()             # Close the FITS file

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
         axinf[axisnr] = FITSaxis(axisnr, self.hdr)
      self.axisinfo = axinf

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

      self.proj = wcs.Projection(self.hdr, alter=self.alter)
      for i in range(n):
         ax = i + 1
         self.axisinfo[ax].wcstype = self.proj.types[i]
         self.axisinfo[ax].wcsunits = self.proj.units[i]
         #self.axisinfo[ax].cdelt = self.proj.cdelt[i]
         #if self.alter != '':
         self.axisinfo[ax].cdelt = self.proj.cdelt[i]
         self.axisinfo[ax].crval = self.proj.crval[i]
         self.axisinfo[ax].ctype = self.proj.ctype[i]
         self.axisinfo[ax].cunit = self.proj.cunit[i]
         self.axisinfo[ax].crpix = self.proj.crpix[i]

      self.spectrans = None   # Set the spectral translation
      self.skyout = None      # Must be set before call to set_imageaxes
      self.set_imageaxes(self.axperm[0], self.axperm[1], self.slicepos)
      self.aspectratio = None
      self.pixelaspectratio = self.get_pixelaspectratio()
      self.figsize = None      # TODO is dit nog belangrijk??
      self.boxdat = self.dat



   def globalminmax(self):
   #------------------------------------------------------------
      """
      Get minimum and maximum value of data in entire data structure
      defined by the current FITS header. These values can be important if
      you want to compare different images from the same source
      (e.g. channel maps in a radio data cube).

      :Returns:
         min, max, two floating point numbers representing the minimum
         and maximum data value in data units of the header (*BUNIT*).
      """
   #------------------------------------------------------------
      if self.boxdat == None:
         return 0, 1
      filtr = self.boxdat[numpy.isfinite(self.boxdat)]
      mi = filtr.min()
      ma = filtr.max()
      #av = filtr.mean(); print "AV=", av
      #rms = filtr.std(); print "std=", rms
      return mi, ma


   def str_header(self):
   #------------------------------------------------------------
      """
      Print the meta information from the selected header.
      Omit items of type *HISTORY*.

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
         To display information about the two image axes one shoulf use
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
      if axnum == None:
         axnum = range(1, self.naxis+1)
      if type(axnum) not in sequencelist:
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
      Print the data related to the World Coordinate System (WCS)
      such as the current sky system and which axes are
      longitude, latitude or spectral.

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
      Print the spectral translations for this data.

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


   def set_imageaxes(self, axnr1=None, axnr2=None, slicepos=None, promptfie=None):
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
                extracted from the relevant axis in attribute :attr:`slicepos`.

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
         >>> fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)

      """
      #-----------------------------------------------------------------
      n = self.naxis
      if n >= 2:
         if (axnr1 == None or axnr2 == None) and promptfie == None:
            if (axnr1 == None and axnr2 == None):
               axnr1 = self.axperm[0]
               axnr2 = self.axperm[1]
            else:
               raise Exception, "One axis number is missing and no prompt function is given!"
         if slicepos == None and promptfie == None:
            slicepos = self.slicepos

      # If there is a spectral axis in the FITS file, then get allowed
      # spectral translations
      self.allowedtrans = self.proj.altspec

      if promptfie != None:
         axnr1, axnr2, self.slicepos = promptfie(self, axnr1, axnr2)
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
         if self.dat != None:
            self.boxdat = self.dat[sl].squeeze()
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

      self.convproj = self.proj.sub(ap)  # Projection object for selected image only
      if self.spectrans != None:
         self.convproj = self.convproj.spectra(self.spectrans)
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
      for ax in self.axperm:
         if self.axisinfo[ax].wcstype == 'spectral':
            isspectral = True
      if not isspectral:
         return         # Silently
   
      if promptfie == None:
         if spectrans == None:
            raise Exception, "No spectral translation given!"
         else:
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
      
      if promptfie == None:
         if skyout == None:
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

      :Returns:
         --

      :Notes:
         --

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
      if pxlim == None:
         npxlim = [1, n1]
      else:
         if type(pxlim) not in sequencelist:
            raise Exception, "pxlim must be tuple or list!"
         npxlim[0] = pxlim[0]
         npxlim[1] = pxlim[1]
      if pylim == None:
         npylim = [1, n2]
      else:
         if type(pylim) not in sequencelist:
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
         # self.boxdat = numpy.ma.masked_where(numpy.isnan(z), z)
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
      self.pixelaspectratio = aspectratio
      return aspectratio



   def get_figsize(self, xsize=None, ysize=None, cm=False):
   #---------------------------------------------------------------------
      """
      Usually a user will set the figure size manually
      with Matplotlib's figure(figsize=...) construction.
      For many plots this is a waste of whithe space around the plot.
      This can be improves by taken the aspectratio into account
      and adding some extra space for labels and titles.
      For aspect ratios far from 1.0 the number of pixels in x and y
      are taken into account.

      A handy feature is that you can enter two value in centimeter
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
         return (xcm/2.54, ycm/2.54)

      a1 = self.axperm[0]; a2 = self.axperm[1];
      cdeltx = self.axisinfo[a1].cdelt
      cdelty = self.axisinfo[a2].cdelt
      nx = float(self.pxlim[1] - self.pxlim[0] + 1)
      ny = float(self.pylim[1] - self.pylim[0] + 1)
      aspectratio = abs(cdelty/cdeltx)
      if aspectratio > 10.0 or aspectratio < 0.1:
         aspectratio = nx/ny
      extraspace = 3.0  # cm

      if xsize == None and ysize == None:
         if abs(nx*cdeltx) >= abs(ny*cdelty):
            xsize = 21.0        # A4 width
         else:
            ysize = 21.0
      if xsize != None:                       # abs(nx*cdeltx) >= abs(ny*cdelty):
         xcm = xsize
         # The extra space is to accomodate labels and titles
         ycm = xcm * (ny/nx) * aspectratio + extraspace
      else:
         ycm = ysize
         xcm = ycm * (nx/ny) / aspectratio + extraspace
      return (xcm/2.54, ycm/2.54)


   def writetofits(self, filename=None):
   #---------------------------------------------------------------------
      """
      This method copies current data and current header to a FITS file
      on disk. This is useful if either header or data comes from an
      external source. If no file name is entered then a file name
      will be composed using current date and time of writing.
      The names start with 'FITS'.

      :param filename:
         Name of new file on disk. If omitted the default name is
         'FITS' followed by a date and a time (in hours, minutes seconds).
      :type filename:
         String
      
      :Returns:
         ---
      
      :Notes:
         ---

      :Examples: Artificial header and data:

        ::
      
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


      """
   #---------------------------------------------------------------------
      if filename == None:
         # Create filename unique to the second. If more resolution
         # is needed use %f to get microseconds.
         from datetime import datetime
         d = datetime.today()
         filename = d.strftime("FITS%y%m%d_%Hh%Mm%Ss.fits")
      hdu = pyfits.PrimaryHDU(self.dat)
      # There is no simple method to copy a dict to a hdu.header object
      # which is not a dict.
      for k, v in self.hdr.iteritems():
         hdu.header.update(k, v)
      hdulist = pyfits.HDUList([hdu])
      hdulist.writeto(filename)
      hdulist.close()
      

   def Annotatedimage(self, frame, **kwargs):
   #---------------------------------------------------------------------
      """
      This method couples the data slice that represents an image to
      a Matplotlib Axes object (parameter *frame*). It returns an object
      from class :class:`Annotatedimage` which has only attributes relevant for
      Matplotlib.

      :param frame:
         Plot the current image in this Matplotlib Axes object.
      :type frame:
         A Matplotlib Axes instance

      :Returns:
         An object from class :class:`Annotatedimage`
      """
   #---------------------------------------------------------------------
      ar = self.get_pixelaspectratio()
      basename = self.filename.rsplit('.')[0]
      # Note the use of self.boxdat  instead of self.dat !!
      mplimage = Annotatedimage(frame, self.hdr, self.pxlim, self.pylim, self.boxdat,
                                self.convproj, self.axperm,
                                skyout=self.skyout, spectrans=self.spectrans,
                                mixpix=self.mixpix, aspect=ar, slicepos=self.slicepos,
                                basename=basename, **kwargs)
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
   
:param helptext:
   Allow or disallow methods to set an informative text about the keys in use.
:type helptext:
   Boolean
:param imagenumbers:
   Allow or disallow methods to set an informative text about which image
   is displayed and, if available, it prints information about the pixel
   coordinate(s) of the slice if the image was extracted from a data cube.
:type imagenumbers:
   Boolean
   

:Returns:
   --

:Attributes:
   
    .. attribute:: annimagelist

       List with objects from class :class:`maputils.Annotatedimage`.
       
    .. attribute:: indx
    
       Index in list with objects of object which represents the current image.

    .. attribute:: framespersec

       A value in seconds, representing the interval of refreshing an image
       in the movie loop.
       
:Notes:
   --
   

:Examples:
   Use of this class as a container for images in a movie loop:

   .. literalinclude:: EXAMPLES/mu_movie.py


   Skip informative text:
   
   >>> movieimages = maputils.MovieContainer(helptext=False, imagenumbers=False)


:Methods:

.. automethod:: append
.. automethod:: movie_events
.. automethod:: controlpanel
.. automethod:: imageloop
.. automethod:: toggle_images
.. 
   """
#--------------------------------------------------------------------
   def __init__(self, helptext=True, imagenumbers=True):
      self.annimagelist = []                     # The list with Annotatedimage objects
      self.indx = 0                              # Sets the current image in the list
      self.fig = None                            # Current Matplotlib figure instance
      self.textid = None                         # Plot and erase text on canvas using this id.
      self.helptext = helptext
      self.imagenumbers = imagenumbers


   def append(self, annimage, visible=True):
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
      
      :Returns:
          --

      :Raises:
         'Container object not of class maputils.Annotatedimage!'
         An object was not recognized as a valid object to append.
      
      :Notes:
          --
      """
   #---------------------------------------------------------------------
      if not isinstance(annimage, Annotatedimage):
         raise TypeError, "Container object not of class maputils.Annotatedimage!" 
      if len(self.annimagelist) == 0:            # This must be the first object in the container
         self.fig = annimage.frame.figure
      annimage.image.im.set_visible(visible)
      self.annimagelist.append(annimage)


   def movie_events(self):
   #---------------------------------------------------------------------
      """
      Connect keys for movie control and start the movie.
      
      :Returns:
          --

      :Raises:
         'No objects in container!'
         The movie container is empty. Use method :meth:`MovieContainer.append`
         to fill it.
         
      :Notes:
          --
      """
   #---------------------------------------------------------------------
      if self.fig == None:
         raise Exception, "No matplotlib.figure instance available!"
      if len(self.annimagelist) == 0:
         raise Exception, "No objects in container!"
      self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.controlpanel)
      self.pause = False
      self.framespersec = 30
      self.movieloop = TimeCallback(self.imageloop, 1.0/self.framespersec)
      if self.helptext:
         self.helptextbase = "Use keys 'p' to Pause/Resume. '+','-' to increase/decrease movie speed.  '<', '>' to step in Pause mode."
         speedtext = " Speed=%d im/s"% (self.framespersec)
         self.helptext_id = self.fig.text(0.5, 0.005, self.helptextbase+speedtext, color='g', fontsize=8, ha='center')
      else:
         self.helptextbase = ''
         speedtext = ''
         self.helptext_id = None
      if self.imagenumbers:
         # Initialize info text (which shows where we are in the movie loop).
         self.imagenumberstext_id = self.fig.text(0.01, 0.95, '', color='g', fontsize=8)


   def controlpanel(self, event):
   #---------------------------------------------------------------------
      """
      Process the key events.
      """
   #---------------------------------------------------------------------
      delta = 0.005
      key = event.key.upper()
      # Pause button is toggle
      if key == 'P':
         if self.pause:
            self.movieloop.schedule()
            self.pause = False
         else:
            self.movieloop.deschedule()
            self.pause = True

      # Increase speed of movie
      elif key in ['+', '=']:
         self.framespersec = min(self.framespersec+1, 100)   # Just to be save
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

      elif key in [',','<']:
         self.toggle_images(next=False)

      elif key in ['.','>']:
         self.toggle_images()


         
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
      self.toggle_images()
      

   def toggle_images(self, next=True):
   #---------------------------------------------------------------------
       """
       Toggle the visible state of images either by a timed callback
       function or by keys.
       This toggle works if one stacks multiple image in one frame
       with method :meth:`MovieContainer.append`.
       Only one image gets status visible=True. The others
       are set to visible=False. This toggle changes this visibility
       for images and the effect, is a movie.
       
       :param next:
          Step forward through list if next=True. Else step backwards.
       :type next:
          Boolean
          
       :Returns:
          --
       
       :Notes:
          
       """
    #---------------------------------------------------------------------
       oldim = self.annimagelist[self.indx]
       oldim.image.im.set_visible(False)

       numimages = len(self.annimagelist) 
       if next:
          if self.indx + 1 >= numimages:
             self.indx = 0
          else:
             self.indx += 1
       else:
          if self.indx - 1 < 0:
             self.indx = numimages - 1
          else:
             self.indx -= 1
    
       newindx = self.indx
       newim = self.annimagelist[newindx]
       slicepos = str(newim.slicepos)
       newim.image.im.set_visible(True)

       if self.imagenumbers:
          self.imagenumberstext_id.set_text("im #%d slice:%s"%(newindx, slicepos))

       self.fig.canvas.draw()
