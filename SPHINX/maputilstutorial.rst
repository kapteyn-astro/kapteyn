.. _maputils_tutorial:
   
Tutorial maputils module
========================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------


This module is a base module for small applications which all start with
reading data from a FITS file. There are methods to inspect the header and methods
to display the image data. It's relation to world coordinates is the simple interface
for drawing a :term:`graticule` in a plot. Related to graticules are spatial rulers
which show offsets of constant distance whatever the projection of the map is.
 
The module combines the functionality in
:mod:`wcs`, :mod:`celestial`, and :mod:`wcsgrat`, together with Matplotlib,
into a powerful module for the extraction and display of FITS image data.
We show examples of:
   
   * overlays of different graticules (each representing
     a different sky system),
   * plots of data slices from a data set with more than two axes
     (e.g. a FITS file with channel maps from a radio interferometer observation)
   * plots with a spectral axis with a 'spectral translation' (e.g. Frequency to Radio velocity)
   * rulers
   * plots that cover the entire sky (allsky plot)
   * mosaics of multiple images (e.g. HI channel maps)

We describe simple methods to add interaction to the Matplotlib canvas e.g. for
changing color maps or color ranges.

In this tutorial we assume a basic knowledge of FITS files.
Also a basic knowledge of Matplotlib is handy but not
necessary to be able to modify the examples in this tutorial.
For useful references see information below.

.. seealso::

   `FITS standard <http://fits.gsfc.nasa.gov/standard30/fits_standard30.pdf>`_
      A pdf document that describes the current FITS standard.

   `Matplotlib <http://matplotlib.sourceforge.net/index.html>`_
      Starting point for documentation about plotting with Matplotlib.
      
   Module :mod:`celestial`
      Documentation of sky- and reference systems. Useful if you need
      to define a celestial system.

   Module :mod:`wcsgrat`
      Documentation about graticules. Useful if you want to fine tune the
      wcs coordinate grid.


FITS files
-----------

A simple utility to analyze a FITS file
.......................................

The most important class in module :mod:`maputils` is class
:class:`maputils.FITSimage`. It extracts a two dimensional data set
from a FITS file. There are a number of methods that print information
from the FITS header and the derived :class:`wcs.Projection` object.

.. literalinclude:: EXAMPLES/mu_fitsutils.py

This code generates the following output::
   
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

This example extacts data from a FITS file on disk as given in the example code.
To make the script a real utility one should allow the user to enter a file
name. This can be done with Python's `raw-input` function but to make
it robust one should check the existence of a file, and if a FITS
file has more than one header, one should prompt a user to
specify the header. We also have to deal with alternate headers for world coordinate
systems etc. etc.
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
     >>> fitsobject.set_imageaxes(2,1)

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
     >>> fitsobject.set_skyout("Equatorial, J1952, FK4_no_e, J1980")
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

Example: mu_getfitsimage.py - Use prompt functions to set attributes of the
FITSimage object and print information about the world coordinate system.

.. literalinclude:: EXAMPLES/mu_getfitsimage.py


Image objects
-------------

Basic image
...........

If one is interested in displaying image data only (i.e. without any wcs information)
then we need very few lines of code as we show in the next example.

.. plot:: EXAMPLES/mu_simple.py
   :include-source:
   :align: center

.. centered:: Fig.: mu_simple.py - Simple plot using defaults
 

**Explanation:**

This is a simple script that displays an image using the default color map.
>From a :class:`maputils.FITSimage` object an :class:`maputils.FITSimage.Annotatedimage` object is derived.
This object has methods to create other objects that can be plotted
with Matplotlib. To plot these objects we need to call method
:meth:`maputils.FITSimage.Annotatedimage.plot`

If you are only interested in displaying the image and don't want any white space
around the plot
then you should replace the *add_subplot()* method by a frame that covers
(depending on the aspectratio of the image) the entire width or height of the window.

We show this by the next example where we also use keyword parameters *cmap*, *vmin* and *vmax*
to set a color map and the clip levels between which the color mapping is applied.


.. plot:: EXAMPLES/mu_withimage.py
   :include-source:
   :align: center

.. centered:: Fig.: mu_withimage.py - Image with parameters for color map and clip levels.


Graticules
----------

Introduction
............

Module :mod:`maputils` can create graticule objects with method :meth:`maputils.Graticule`.
But in fact the method that does all the work is defined in module :mod:`wcsgrat` so
in the next sections we often refer to module :mod:`wcsgrat`.

Module :mod:`wcsgrat` creates a :term:`graticule` for a given header with WCS information.
That implies that it finds positions on a curve in 2 dimensions in image data
for which one of the world coordinates is a constant value.
These positions are stored in a graticule object :class:`wcsgrat.Graticule`.
The positions at which these lines cross one of the sides
of the rectangle (made up by the limits in pixels in both x- and y-direction),
are stored in an object from class :class:`wcsgrat.WCStick`, together with a
text label showing the world coordinate of the crossing. In principle this is
all what this module does. Currently we support *Matplotlib*
as plot software, but we developed the module also with other plot packages
in mind so it should not be impossible to add another (e.g. ppgplot).


Simple example
..............

Example: mu_axnumdemosimple.py - Simple plot using defaults

.. plot:: EXAMPLES/mu_axnumdemosimple.py
   :include-source:
   :align: center

**Explanation:**

The script opens an existing FITS file. Its header is parsed by methods
in module :mod:`wcs` and methods from classes in module :mod:`wcsgrat`
calculate the graticule data. A plot is made with Matplotlib.

The recipe:

   * Given a FITS file on disk (example1test.fits) we want to plot
     a graticule for the spatial axes in the FITS file.
   * The necessary information is retrieved from the FITS header
     with PyFITS through class :class:`maputils.FITSimage()`.
   * To plot something we need to tell method :meth:`maputils.FITSimage.Annotatedimage`
     in which frame it must plot. Therefore we need a Matplotlib figure instance
     and a Matplotlib Axes instance (which we call a frame in the context of *maputils*).
   * A graticule representation is calculated and stored in object *grat*
     from :class:`maputils.Graticule`. The maximum number of defaults
     are used.
   * Finally we tell the Annotated image object *mplim* to plot itself and display
     the result with Matplotlib's function *show()*.
     
The :mod:`wcsgrat` module estimates the ranges in world coordinates
in the coordinate system defined in your FITS file.
It calculates 'nice' numbers to
annotate the plot axes and it sets default plot attributes.
For Matplotlib these are the attributes listed in the appropriate
class descriptions of Matplotlib (http://matplotlib.sourceforge.net)

   
**Hint**: Matplotlib versions older than 0.98 use module *pylab* instead of *pyplot*.
You need to change the import statement to:
`from matplotlib import pylab as plt`


Probably you already have many questions about what :mod:`wcsgrat` can do more:

   * Is it possible to draw labels only and no graticule lines?
   * Can I change starting point and sep size for the coordinate labels?
   * Can I change the default titles along the axes?
   * Is it possible to highlight (e.g. by changing color) just one graticule line?
   * Can I plot graticules in maps with one spatial- and one spectral coordinate?
   * Can I control the aspect ratio of the plot?
   * Is it possible to set limits on pixel coordinates?

We will give a number of examples to answer most of the questions.

   
Selecting axes for graticule or grid lines
...........................................

For data sets with **more** than **2** axes or data sets with swapped axes
(e.g. Declination as first axis and Right Ascension as second), we need to make a choice
of the axes and axes order. To demonstrate this we created a FITS file with
four axes. The order of the axes is uncommon and should only demonstrate the
flexibility of the :mod:`maputils` module. We list the data for these axes
in this 'artificial' FITS file::

   Filename: manyaxes.fits
   No.    Name         Type      Cards   Dimensions   Format
   0    PRIMARY     PrimaryHDU      44  (10, 50, 50, 4)  int32
   Axis  1 is FREQ   runs from pixel 1 to    10  (crpix=5 crval,cdelt=1.37835, 9.76563e-05 GHZ)
   Axis  2 is DEC    runs from pixel 1 to    50  (crpix=30 crval,cdelt=45, -0.01 DEGREE)
   Axis  3 is RA     runs from pixel 1 to    50  (crpix=25 crval,cdelt=30, -0.01 DEGREE)
   Axis  4 is POL    runs from pixel 1 to     4  (crpix=1 crval,cdelt=1000, 10 STOKES)

You can download the file `manyaxes.fits <http://www.astro.rug.nl/software/kapteyn/EXAMPLES/manyaxes.fits>`_
for testing. The world coordinate system is arbitrary.

Example: mu_manyaxes.py - Selecting WCS axes from a FITS file

.. plot:: EXAMPLES/mu_manyaxes.py
   :include-source:
   :align: center

The plot shows a system of grid lines that correspond to non spatial axes. and it will be no
surprise that the graticule is a rectangular system.
The example follows the same recipe as the previous and it shows how one
selects the required plot axes in a FITS file.
The axes are set with :meth:`maputils.FITSimage.set_imageaxes`
with two numbers. The first axis of a set
is axis 1, the second 2, etc. (i.e. FITS standard). The default is
(1,2) i.e. the first two axes in a FITS header.

For a R.A.-Dec. graticule one should enter for this FITS file:

>>> f.set_imageaxes(3,2)


.. note:: 

   If a FITS file has data which has more than two dimensions or
   it has two dimensions but you want to swap the x- and y axis then you need
   to specify the relevant FITS axes with :meth:`maputils.FITSimage.set_imageaxes`.
   The (FITS) axes numbers correspond to the number n in the FITS keyword CTYPEn
   (e.g. CTYPE3='FREQ' then the frequency axis corresponds to number 3).


This example shows an important feature of the underlying module :mod:`wcsgrat` and that is
its functionality to change properties graticules, ticks and labels.
We summarize:

   * Graticule line properties are set with :meth:`wcsgrat.Graticule.setp_gratline`
     or the equivalent :meth:`wcsgrat.Graticule.setp_lineswcs1` or
     :meth:`wcsgrat.Graticule.setp_lineswcs1`. The properties are all Matplotlib
     properties given as keyword arguments. One can apply these to all graticule
     lines, to one of the wcs types or to one graticule line (identified by
     its position in world coordinates).
   * Graticule ticks (the intersections with the borders) are modified by
     method :meth:`wcsgrat.Graticule.setp_tick`.
     Ticks are identified by either the wcs axis (e.g. longitude or latitude)
     or by one of the four rectangular plot axes or by a position in
     world coordinates. Combinations of these are also possible.
     There is only one parameter that sets a property of the tick line
     (*markersize*) the others change properties of the text labels.
     Plot properties are given as Matplotlib keyword arguments. The labels can be
     scaled and formatted with parameters *fun* and *fmt* 
   * The titles along one of the rectangular plot axes can be modified with
     :meth:`wcsgrat.Graticule.setp_plotaxis`. A label is set with parameter *label*
     and the plot properties are given as Matplotlib keyword arguments.
     For each 'plotaxis' one can set which ticks (i.e. from which 'wcsaxis')
     should be plotted and which not (think of rotated graticules).
   * Properties of labels inside a plot are set in the constructor
     :meth:`wcsgrat.Graticule.Insidelabels`.
 
Let's study the plot in more detail:

   * The header shows a Stokes axes with an uncommon value for ``CRVAL`` and ``CDELT``.
     We want to label four graticule lines with the familiar Stokes parameters.
     With the knowledge we have about this ``CRVAL`` and ``CDELT`` we tell
     the Graticule constructor to create 4 graticule lines (``starty=1000, deltay=10``).
   * The four positions are stored in attribute *ystarts* as in ``grat.ystarts``.
     we use these numbers to change the coordinate labels into Stokes parameters with
     method :meth:`wcsgrat.Graticule.setp_tick`

     >>> grat.setp_tick(plotaxis=wcsgrat.left, position=1000, color='m', fmt="I")

   * We used :meth:`wcsgrat.Graticule.Insidelabels` to add coordinate labels
     inside the plot. We marked a position near ``CRVAL`` and plotted a label
     and with the same method we added a single label at that position.
     


More 'axnum' variations -- Position Velocity diagrams
-----------------------------------------------------

For the next example we used a FITS file with the following header information::

   Axis 1: RA---NCP  from pixel 1 to   100  {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
   Axis 2: DEC--NCP  from pixel 1 to   100  {crpix=51 crval=60.1539 cdelt=0.007166 (DEGREE)}
   Axis 3: VELO-HEL  from pixel 1 to   101  {crpix=-20 crval=-243 cdelt=4.2 (km/s)}

Example: mu_axnumdemo.py - Show different axes combinations for the same FITS file

.. plot:: EXAMPLES/mu_axnumdemo.py
   :include-source:
   :align: center


We used Matplotlib's *add_subplot()* methode to create 4 plots in one figure with minimum effort.
The top panel shows a plot with the default axis numbers which are 1 and 2.
This corresponds to the axis types RA and DEC and therefore the map is a spatial map.
The next panel has axis numbers 3 and 2 representing a *position-velocity* or *XV map* with DEC
as the spatial axis X. The default annotation is offset in spatial distances.
The next panel is a copy but we changed the annotation from the default
(i.e. offsets) to position labels. This could make sense if the map is unrotated.
The bottom panel has RA as the spatial axis X. World coordinate labels
are added inside the plot with a special method: :meth:`wcsgrat.Graticule.Insidelabels`.
These labels are not formatted to hour/min/sec or deg/min/sec for spatial axes.

The two calls to this method need some extra explanation::

   graticule4.Insidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10,
                           color='r', ha='right')
   graticule4.Insidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')

The first line sets labels that correspond to positions
in world coordinates inside a plot. It copies the positions of the velocities,
set by the initialization of the graticule object. It plots those labels at a
Right Ascension equal to 20h36m which is equal to -51 (==309) degrees.
It rotates these labels with angle 90 degrees and
sets the size, color and alignment of the font. The second line does something similar for
the Right Ascension labels, but it adds a format for numbers.

Note also the line:
   
   >>> graticule4 = mplim4.Graticule(offsety=False)
   >>> graticule4.setp_tick(plotaxis="left", fun=lambda x: x+360, fmt="$%.1f^\circ$")

Default the module would plot labels which are offsets because we have only one spatial axis.
We overruled this behaviour with keyword parameter *offsety=False*. Then we get world coordinates
which are default formatted in hour/minutes/seconds. But we want these labels to be
plotted in another format. With parameter *fun* we define a transformation and with
*fmt* we set the format for the output. Note the use of the TeX symbol for degrees.

Finally note that the alignment of the titles along the left axis (which is a Matplotlib
method) works in the frame of the graticule. It is important to realize that a *maputils* plot
usually is a stack of matplotlib Axes objects (frames). The graticule object sets these
axis labels and therefore we must align them in that frame (which is an attribute
of the graticule object) as in:

   >>> graticule1.frame.yaxis.set_label_coords(labelx, 0.5)


Setting an aspect ratio
.......................

For images and graticules representing spatial data it is important that the aspect 
ratio (CDELTy/CDELTx) remains constant if you resize the plot. 
A graticule object initializes itself with an aspect ratio based on the pixel
sizes found in (or derived from) the header. It also calculates an appropriate
figure size and size for the actual plot window in normalized device coordinates
(i.e. in interval [0,1]). You can use these values in a script to set
the relevant values for Matplotlib as we show in the next example.

Example: mu_figuredemo.py - Plot figure in correct aspect ratio and fix the aspect ratio.

.. plot:: EXAMPLES/mu_figuredemo.py
   :include-source:
   :align: center

.. note::

   For astronomical data we want equal steps in spatial distance in any direction correspond
   to equal steps in figure size. If one changes the size of the figure interactively,
   the aspect ratio should not change. To enforce this, the constructor of an
   object of class :class:`Annotatedimage` modifies the input frame so that the
   aspect ratio is the aspect ration of the pixels. This aspect ratio os preserved
   when the size of a window is changed.
   One can overrule this default by manually setting an aspect ratio with method
   :meth:`set_aspectratio` as in:

   >>> frame = fig.add_subplot(k,1,1)
   >>> mplim = f.Annotatedimage(frame)
   >>> mplim.set_aspectratio(0.02)

Combinations of graticules
..........................

An object of class :class:`wcsgrat.Plotversion` is a container for graticules,
pixel labels and rulers.
The number of plotable objects is not restricted to one. One can easily add a
second graticule for a different sky system or a couple of rulers etc.
These are all added to the container with method :meth:`wcsgrat.Plotversion.add`
and the contents is plotted with method :meth:`wcsgrat.Plotversion.plot`.

The next example shows a combination of two graticules for two different sky systems.
It demonstrates also the use of attributes to changes plot properties.

Example: mu_skyout.py - Combine two graticules in one frame

.. plot:: EXAMPLES/mu_skyout.py
   :include-source:
   :align: center


**Explanation:**

This plot shows a graticule for equatorial coordinates and galactic coordinates in the
same figure. The center of the image is the position of the galactic pole. That is why
the graticule for the galactic system shows circles. The galactic graticule is also
labeled inside the plot using method :meth:`wcsgrat.Graticule.Insidelabels`.
To get an impression of arbitrary positions expressed in pixels coordinates,
we added pixel coordinate labels for the top and right axes with
method :meth:`wcsgrat.Graticule.Pixellabels`.


**Plot properties:**

   * Use attribute *boxsamples* to get a better estimation of the ranges in galactic
     coordinates. The default sampling does not sample enough in the neighbourhood of the galactic
     pole causing a gap in the plot.
   * Use method :meth:`wcsgrat.Graticule.setp_lineswcs0` to change the color of the
     longitudes (and *linewcs1* for the latitudes) for the equatorial system.
   * Method :meth:`wcsgrat.Graticule.setp_tick`
     sets for both plot axis (0 == x axis, 1 = y axis)
     the tick length with *markersize*. The value is negative to force a 
     tick that points outwards. Also the color and the font size of the tick labels 
     is set. Note that these are Matplotlib keyword arguments.
   * With :meth:`wcsgrat.Graticule.setp_plotaxis` we allow galactic coordinate labels and ticks 
     to be plotted along the top and right plot axis. Default, the labels along these axes
     are set to be invisible, so we need to make them visible with keyword argument *visible=True*.
     Also a title is set for these axes.
     
.. note:: 
   
     There is a difference between plot axes and wcs axes. The first always represent a rectangular
     system while the system of the graticule lines (wcs axes) usually is curved (sometimes
     they are even circular. Therefore many plot properties are either associated with one
     or more plot axes and other with one or both world coordinate axes.



Spectral translations
.....................

To demonstrate what is possible with spectral coordinates and module :mod:`wcsgrat`
we use real interferometer data from a set called *mclean.fits*. A summary of what can be 
found in its header::
   
   Axis  1: RA---NCP  from pixel 1 to   512  {crpix=257 crval=178.779 cdelt=-0.0012 (DEGREE)}
   Axis  2: DEC--NCP  from pixel 1 to   512  {crpix=257 crval=53.655 cdelt=0.00149716 (DEGREE)}
   Axis  3: FREQ-OHEL from pixel 1 to    61  {crpix=30 crval=1.41542E+09 cdelt=-78125 (HZ)}

Its spectral axis number is 3. The type is frequency. The extension tells us that an
optical velocity in the heliocentric system is associated with the frequencies. In the
header we found that the optical velocity is 1050 Km/s.
The header is a legacy GIPSY header and module :mod:`wcs` can parse it.
We require the frequencies to be expressed as wavelengths.

Example: wcsg_wave.py - Plot a graticule in a position wavelength diagram.

.. plot:: EXAMPLES/mu_wave.py
   :include-source:
   :align: center

**Explanation:**
  
  * With PyFITS we open the fits file on disk and read its header
  * A Matplotlib Figure- and Axes instance are made
  * The range in pixel coordinates in x is decreased
  * A Graticule object is created and for FITS axis 3 along x and FITS axis 2
    along y. The spectral axis is expressed in wavelengths with method :meth:`wcs.Projection.spectra`.
    Note that we omitted a code for the conversion algorithm and instead entered three
    question marks so that the *spectra()* method tries to find the appropriate code.
  * The tick labels along the x axis (the wavelengths) are formatted. The S.I. unit is
    meter, but we want it to be plotted in cm. A function to convert the values is 
    given with `fun=lambda x: x*100`. A format for the printed numbers is given with:
    `fmt="%.3f"`

.. note::
   
   The spatial axis is expressed in offsets. Default it starts with an offset equal
   to zero in the middle of the plot. Then a suitable step size is calculated and
   the corresponding labels are plotted. For spatial offsets we need also
   a value for the missing spatial axis. If not specified with parameter *mixpix*
   in the constructor of class *Graticule*, a default value is assumed equal to CRPIX
   corresponding to the missing spatial axis.
   

For the next example we use the same FITS file (mclean.fits).
 
Example: mu_spectraltypes.py - Plot grid lines for different spectral translations

.. plot:: EXAMPLES/mu_spectraltypes.py
   :include-source:
   :align: center


**Explanation:**

  * With PyFITS we open the fits file on disk and read its header
  * We created a :class:`wcs.Projection` object for this header to get a 
    list with allowed spectral translations (attribute *altspec*). We need
    this list before we create the graticules 
  * A Matplotlib Figure- and Axes instance are made
  * The native FREQ axis (label in red) differs from the FREQ axis in the
    next plot, because a legacy header was found and its freqencies were transformed
    to a barycentric/heliocentric system.


Rulers
------

Rulers in :mod:`wcsgrat` are objects derived from a Graticule object.
A ruler is always plotted
as a straight line, whatever the projection is (so it doesn't necessarily
follow graticule lines).
A ruler plots ticks and labels and the *spatial* distance between any two ticks is
a constant. This makes rulers ideal to put nearby a feature in your map to
give an idea of the physical size of that feature. Rulers can be plotted in maps
with one or two spatial axes. 

Example: mu_manyrulers.py - Ruler demonstration

.. plot:: EXAMPLES/mu_manyrulers.py
   :include-source:
   :align: center

Ruler tick labels can be formatted so that we can adjust them. In the next plot we
want offsets to be plotted in arcminutes.

Example: mu_arminrulers.py - Rulers with non default labels

.. plot:: EXAMPLES/mu_arcminrulers.py
   :include-source:
   :align: center

It is possible to put a ruler in a map with only one spatial coordinate
(as long there is a matching axis in the header) like a Position-Velocity diagram.
It will take the pixel coordinate of the slice as a constant so even for XV maps
we have reliable offsets. In the next example we created two rulers.
The red ruler is in fact the same as the Y-axis offset labeling. The blue
ruler show the same offsets in horizontal direction. That is because only the
horizontal direction is spatial. Such a ruler is probably not very useful but
is a nice demonstration of the flexibility of method :meth:`wcsgrat.Graticule.ruler`.

Note that we set Matplotlib's *clip_on* to *True* because if we pan the image in Matplotlib
we don't want the labels to be visible outside the border of the frame.

Example: mu_xvruler.py - Ruler in a XV map

.. plot:: EXAMPLES/mu_xvruler.py
   :include-source:
   :align: center


Pixel/Grid labels
-----------------

In the previous section we showed an example of multiple Graticule
objects plotted in one plot. Also in that figure we labeled the pixel
coordinates. Also we plotted a grid with dashed lines. This functionality is
provided by method 
:meth:`wcsgrat.Graticule.Pixellabels`. This method consists of Matplotlib
routines and therefore we don't need a special method to set its attributes because
attributes can be set by keyword arguments as in the next code example::
   
>>> pixellabels = grat.Pixellabels(plotaxis=(2,3), gridlines=True, color='c', markersize=-3, fontsize=7)


Labels inside a plot
--------------------

Method :meth:`wcsgrat.Graticule.Insidelabels` sets for a given set of world coordinates
coordinate labels. A number of examples are includes in previous sections.


Glossary
--------

.. glossary::

   graticule
      the network of lines of latitude and longitude upon which a map is drawn




Example: mu_figsize.py

.. plot:: EXAMPLES/mu_figsize.py
   :include-source:
   :align: center

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

Example: mu_colbarwithlines.py - Add lines representing contours in plot to dummy colorbar

.. plot:: EXAMPLES/mu_colbarwithlines.py
   :include-source:
   :align: center


The plot shows two sets of contours. The first set is to plot all contours
in a straightforward way. The second is to plot annotated contours.
For this second set we don't see any contours if a label could not be fitted
that's why we first plot all the contours. Note that now we can use the properties methods
for single contours because we can identify these contours by their corresponding level.

Example: mu_neggativecontours.py - Contours with different line styles for negative values

.. plot:: EXAMPLES/mu_negativecontours.py
   :include-source:
   :align: center

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


Example: mu_externaldata.py - Using external FITS header and data

.. plot:: EXAMPLES/mu_externaldata.py
   :include-source:
   :align: center


The data from a FITS file is stored in a NumPy array. Then it
is straightforward to maniplate this data. NumPy has many methods for this.
We apply a Fourier transform to an image of M101. We show how to use
functions *abs* and *angle* with a complex array as argument to get images of
the amplitude and the fase of the transform. With the transform we test the inverse
procedure and show the residual. There seems to be some systematic structure in
the residual map but notice that the maximum is very small compared to
the smallest image value in the original (which is around 1500).
We used NumPy's FFT functions to calculate the transform. Have a look at the code:


.. plot:: EXAMPLES/mu_fft.py
   :include-source:
   :align: center


.. centered:: Fig. mu_fft.py - FFT: another use of external data

The example shows that we van use external data with the correct shape
in combination with the original FITS header. Note that we used Matplotlib's
*text()* method instead of *xlabel()*. The reason is that the primary
frame has all her axes set to invisible. We can set them to visible but
besides a label, one also get numbers along the axes and that was what we
want to avoid.

Example: mu_histeq.py - Using histogram equalization

.. plot:: EXAMPLES/mu_histeq.py
   :include-source:
   :align: center



Example: mu_channelmaps1.py - Adding two slices

.. plot:: EXAMPLES/mu_channelmaps1.py
   :include-source:
   :align: center

Example: mu_channelmosaic.py - A mosaic of channelmaps

.. plot:: EXAMPLES/mu_channelmosaic.py
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

If you a a number of luts then use Python's glob function to read them all
(or a selection) as in the next example:

.. literalinclude:: EXAMPLES/mu_interactive3.py

Blanks
........

.. literalinclude:: EXAMPLES/mu_imagewithblanks.py


Movies
.......

.. literalinclude:: EXAMPLES/mu_movie.py

