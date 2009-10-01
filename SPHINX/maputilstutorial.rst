Tutorial maputils module
========================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------

This module is a top level utility. I combines the functionality in
:mod:`wcs`, :mod:`celestial`, and :mod:`wcsgrat`, together with Matplotlib,
into a powerful module for the extraction and display of FITS image data.

FITS files
-----------

A simple utility to analyze a FITS file
.......................................

The most important class in module :mod:`maputils` is class
:class:`maputils.FITSimage`. It extracts a two dimensional data set
from a FITS file. There are a number of methods that print information
from the FITS header and the :class:`wcs.Projection` object.

.. literalinclude:: EXAMPLES/mu_fitsutils.py

Output::
   
   HEADER:
   
   SIMPLE  =                    T / SIMPLE FITS FORMAT
   BITPIX  =                  -32 / NUMBER OF BITS PER PIXEL
   NAXIS   =                    3 / NUMBER OF AXES
   NAXIS1  =                  100 / LENGTH OF AXIS
   etc.
   
   AXES INFO:
   
   Axis 1: RA---NCP  from pixel 1 to   100
   {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
   {wcs type=longitude, wcs unit=deg}
   etc.
   
   EXTENDED AXES INFO:
   
   axisnr     - Axis number:  1
   axlen      - Length of axis in pixels (NAXIS):  100
   ctype      - Type of axis (CTYPE):  RA---NCP
   axnamelong - Long axis name:  RA---NCP
   axname     - Short axis name:  RA
   etc.
   
   WCS INFO:
   
   Current sky system:                 Equatorial
   reference system:                   ICRS
   Output sky system:                  Equatorial
   Output reference system:            ICRS
   etc.
   
   SPECTRAL INFO:

   0   FREQ-V2F (Hz)
   1   ENER-V2F (J)
   2   WAVN-V2F (1/m)
   3   VOPT-V2W (m/s)
   etc.

This example extacts data from a FITS file given in the code.
To make the script a utility one should allow the user to enter a file
name. This can be done with Python's `raw-input` function but to make
it useful one should check on the existence of a file, and if a FITS
file has more than one header, one should prompt a user to
specify the header. We also have to deal with alternate headers etc. etc.
To facilitate parameter settings we implemented so called *prompt functions*.
These are external functions which read context and then set reasonable
defaults for the required parameters.


Specification of a map
......................

Class :class:`maputils.FITSimage` extracts data from a FITS file so that a map can be plotted
with its associated world coordinate system.
So we have to specify a number of parameters to get the required image data.
This is done with the following methods:

   * **Header** - The constructor :class:`maputils.FITSimage` needs name and path
     of the FITS file. It can be a file on disk or an URL. The file
     can be zipped. A FITS file can contain more than one header data unit.
     If this is an issue you need to enter the number of the unit that you want to use.
     A FITS header can also contain one or more *alternate* headers. Usually these
     describe another sky or spectral system. We list three examples. The first
     is a complete description of the FITS header. The second get its parameters
     from an external 'prompt' function (see next section)
     and the third uses a prompt function with a pre specfication
     of parameter *alter* which sets the alternate header.
     
     >>> fitsobject = maputils.FITSimage('alt2.fits', hdunr=0, alter='A', memmap=1)
     >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
     >>> fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile, alter='A')

   * **Axis numbers** - Method :meth:`maputils.FITSimage.set_imageaxes` sets the axis numbers (FITS
     standard, i.e. starts with 1) for the two axes in you FITS data. If the data
     has only two axes then it is possible to swap the axes. This method can be
     used in combination with an external prompt function. Examples:

     >>> fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
     >>> fitsobject.set_imageaxes(axnr1=2, axnr2=1)

     For an artificial set called 'manyaxes.fits', we want to extrac one spatial map.
     The axes order is frequency, declination, right ascension, stokes.
     We extract a data slice at FREQ=2 and STOKES=1.
     This spatial map is obtained with the following lines:
        
     >>> fitsobject = maputils.FITSimage('manyaxes.fits') # FREQ-DEC-RA-STOKES
     >>> fitsobject.set_imageaxes(axnr1=3, axnr2=2, slicepos=(2,1))

   * **Coordinate limits** - If you want to extract only a part of the image then
     you need to set limits for the pixel coordinates. This is set with
     :meth:`maputils.FITSimage.set_limits`. The limits can be set manually or with a prompt
     function. Here are examples of both:

     >>> fitsobject.set_limits(pxlim=(20,40), pylim=(22,38))
         or:
     >>> fitsobject.set_limits(promptfie=maputils.prompt_box)

   * **Output sky definition** - For conversions between pixel- and world coordinates
     one can define to which output sky definition the world coordinates are related.
     The sky parameters are set with :meth:`maputils.FITSimage.set_skyout`.
     The syntax for a sky definition (sky system, reference system, equinox, epoch of observation)
     is documented in :meth:`celestial.skymatrix`.
     
     >>> fitsobject = maputils.FITSimage('m101.fits')
     >>> fitsobject.set_skyout((wcs.equatorial,"J1952",wcs.fk4_no_e,"J1980"))
        or:
     >>> fitsobject.set_skyout(promptfie=maputils.prompt_skyout)
     
     
Prompt functions
..................

Usually one doesn't know exactly what's in the header of a FITS file
or one has limited knowledge about the  input parameters :meth:`maputils.FITSimage.set_imageaxes`
Then a helper function is available. It is called :func:`maputils.prompt_imageaxes`.

But a complete description of the world coordinate system implies also that we
set limits for the pixel coordinates (e.g. to extract part of the entire image)
and specify the sky system in which we present a spatial map or the
spectral translation (e.g. from frequency to velocity) for an image with a
spectral axis. Then we turn our basic script into an interactive application
that sets all necessary parameters to extract the required image data
from a FITS file. The next script is an example how we use prompt functions
to ask a user to enter relevant information. These prompt functions are
external functions. They are aware of the context and set reasonable
defaults for the required parameters.

.. literalinclude:: EXAMPLES/mu_getfitsimage.py


Simple examples
----------------

Basic
......

Example: mu_simple.py - Simple plot using defaults

.. plot:: EXAMPLES/mu_simple.py
   :include-source:
   :align: center

**Explanation:**

This is a simple script that displays an image using the default color map.
>From a :class:`FITSimage` object an :class:`Annotatedimage` object is derived.
This object has methods to create other objects that can be plotted
with Matplotlib. To plot these objects we need to call method
:meth:`Annotatedimage.plot`

If you are only interested in displaying the image and don't care for any annotation
then you could replace the *add_subplot()* method by a frame that covers
(depending on the aspectratio of the image) the entire width or height of the window.
One creates this with:

>>> frame = fig.add_axes([0,0,1,1])


Adding contours
...............

Example: mu_simplecontours.py - Simple plot with contour lines only

.. plot:: EXAMPLES/mu_simplecontours.py
   :include-source:
   :align: center

This example shows how to plot contours without plotting an image.
It also shows how one can retrieve the contour levels that are
calculated as a default because no levels were specified.

Example: mu_annotatedcontours.py - Add annotation to contours

.. plot:: EXAMPLES/mu_annotatedcontours.py
   :include-source:
   :align: center


The plot shows two sets of contours. The first set is to plot all contours
in a straightforward way. The second is to plot annotated contours.
For this second set we don't see any contours if a label could not be fitted
that's why we first plot all the contours. Note that now we can use the properties methods
for single contours because we can identify these contours by their corresponding level.


Adding pixel coordinate labels
...............................

Example: mu_pixellabels.py - Add annotation for pixel coordinates

.. plot:: EXAMPLES/mu_pixellabels.py
   :include-source:
   :align: center



Example: mu_graticules.py - Simple plot using defaults

.. plot:: EXAMPLES/mu_graticule.py
   :include-source:
   :align: center


Interaction with the display
----------------------------

Matplotlib (v 0.99) provides a number of built-in keyboard shortcuts.
These are available on any Matplotlib window.

**Navigation Keyboard Shortcuts**

================================ =============================
Command                          Keyboard Shortcut(s)
================================ =============================
Home/Reset                       h or r or home
Back                             c or left arrow or backspace
Forward                          v or right arrow
Pan/Zoom                         p
Zoom-to-rect                     o
Save                             s
Toggle fullscreen                f
Constrain pan/zoom to x axis     hold x
Constrain pan/zoom to y axis     hold y
Preserve aspect ratio            hold CONTROL
Toggle grid                      g
Toggle y axis scale (log/linear) l
================================ =============================

One can add keyboard- and mouse events with methods from :class:`Annotatedimage`.

Adding messages with position information
.........................................

Information about the pixel coordinates, the world coordinates
and the image values at the position of the mouse is displayed 
in the toolbar in a Matplotlib window. There is a minimum width
for the window to be able to display the message.
Here is a minimalistic example how to add this functionality:

.. literalinclude:: EXAMPLES/mu_interactive2.py