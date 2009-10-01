Tutorial wcsgrat module
=======================

.. highlight:: python
   :linenothreshold: 10

.. _wcsgrat_tutorial:


Introduction
------------

Module :mod:`wcsgrat` creates a :term:`graticule` for a given header with WCS information.
That implies that it finds positions on a curve in 2-dimensional data
for which one of the world coordinates is a constant value.
These positions are stored in an object derived from class
:class:`wcsgrat.Graticule`. The positions at which these lines cross one of the sides
of the rectangle (made up by the limits in pixels in both x- and y-direction),
are stored in an object from class :class:`wcsgrat.WCStick`, together with a
text label showing the world coordinate of the crossing. In principle this is
all what this module does. To make it more useful we added a class
with a method that plots the graticule. Currently we support *Matplotlib*
as plot software, but similar methods for other plot packages
should be possible to add (e.g. ppgplot).


Simple example
--------------

Example: wcsg_axnumdemosimple.py - Simple plot using defaults

.. plot:: EXAMPLES/wcsg_axnumdemosimple.py
   :include-source:
   :align: center

**Explanation:**

The script opens an existing FITS file. Its header is parsed by methods
in module :mod:`wcs` and methods from classes in module :mod:`wcsgrat`
calculate graticule data. A plot is made with Matplotlib.

The recipe:

   * Given a FITS file on disk (example1test.fits) we want to plot
     a graticule for the spatial axes in the FITS file.
   * The necessary information is retrieved from the FITS header
     with PyFITS.
   * A graticule representation is calculated with object *grat*
     from :class:`wcsgrat.Graticule`. The maximum number of defaults
     are used.
   * Set the plot package for :class:`wcsgrat.Plotversion`.
     Its parameters are the string 'matplotlib', a Matplotlib figure instance
     and a Matplotlib Axes instance (which we call a frame in the context of
     :mod:`wcsgrat`).
   * The graticule object *grat* is added to a container (*gratplot*) with objects
     supported by :mod:`wcsgrat` (graticules, rulers, grid lines etc.) with
     :meth:`wcsgrat.Plotversion.add`
   * A plot of all the objects in the container *gratplot* is created with
     method :meth:`wcsgrat.Plotversion.plot`.
   
     
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
------------------------------------------
   
For data sets with more than two axes or data sets with swapped axes
(e.g. Declination before Right Ascension), we need to make a choice
of the axes and axes order. To demonstrate this we created a FITS file with
four axes. The order of the axes is uncommon and should only demonstrate the
flexibility of the :mod:`wcsgrat` module. We list the data for these axes
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

Example: wcsg_manyaxes.py - Selecting WCS axes from a FITS file

.. plot:: EXAMPLES/wcsg_manyaxes.py
   :include-source:
   :align: center

The plot shows a system of grid lines that correspond to non spatial axes. and it will be no
surprise that the graticule is a rectangular system.
The example follows the same recipe as the previous and it shows how one
selects the required plot axes in a FITS file. The parameter is *axnum* and
you need to enter a tuple or list with two numbers. The first axis of a set
is axis 1, the second 2, etc. (i.e. FITS standard). The default in
:class:`wcsgrat.Graticule` is
*axnum=(1,2)*. For a R.A.-Dec. graticule one should enter axnum=(3,2).

.. note:: 

   If a FITS file has data which has more than two dimensions or
   it has two dimensions but you want to swap the x- and y axis then you need
   to specify the relevant FITS axes with parameter *axnum* to extract a graticule.
   The (FITS) axes numbers correspond to the number n in the FITS keyword CTYPEn.


This example shows an important feature of module :mod:`wcsgrat` and that is
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
     :meth:`wcsgrat.Graticule.insidelabels`.
 
Let's study the plot in more detail:

   * The header shows a Stokes axes with an uncommon value for ``CRVAL`` and ``CDELT``.
     We want to label four graticule lines with the familiar Stokes parameters.
     With the knowledge we have about this ``CRVAL`` and ``CDELT`` we tell
     the Graticule constructor to create 4 graticule lines (``starty=1000, deltay=10``).
   * The four positions are stored in attribute *ystarts* as in ``grat.ystarts``.
     we use these numbers to change the coordinate labels into Stokes parameters with
     method :meth:`wcsgrat.Graticule.setp_tick`

     >>> grat.setp_tick(plotaxis=wcsgrat.left, position=1000, color='m', fmt="I")

   * We used :meth:`wcsgrat.Graticule.insidelabels` to add coordinate labels
     inside the plot. We marked a position near ``CRVAL`` and plotted a label
     and with the same method we added a single label at that position.
     


More 'axnum' variations -- Position Velocity diagrams
-----------------------------------------------------

For the next example we used a FITS file with the following header information::

   Axis 1: RA---NCP  from pixel 1 to   100  {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
   Axis 2: DEC--NCP  from pixel 1 to   100  {crpix=51 crval=60.1539 cdelt=0.007166 (DEGREE)}
   Axis 3: VELO-HEL  from pixel 1 to   101  {crpix=-20 crval=-243 cdelt=4.2 (km/s)}

Example: wcsg_axnumdemo.py - Show different axes combinations for the same FITS file

.. plot:: EXAMPLES/wcsg_axnumdemo.py
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
are added inside the plot with a special method: :meth:`wcsgrat.Graticule.setinsidelabels`.
These labels are not formatted to hour/min/sec or deg/min/sec for spatial axes.

The two calls to this method need some extra explanation::

   ilabs1 = grat3.insidelabels(wcsaxis=0, constval=-51,
                               rotation=90, fontsize=10, color='r')
   ilabs2 = grat3.insidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')

The first line sets labels that correspond to positions
in world coordinates inside a plot. It copies the positions of the velocities,
set by the initialization of the graticule object. It plots those labels at a
Right Ascension equal to -51. It rotates these labels with angle 90 degrees and
sets the size and color of the font. The second line does something similar for
the Right Ascension labels, but it adds a format for numbers.


Setting an aspect ratio
-----------------------

For images and graticules representing spatial data it is important that the aspect 
ratio (CDELTy/CDELTx) remains constant if you resize the plot. 
A graticule object initializes itself with an aspect ratio based on the pixel
sizes found in (or derived from) the header. It also calculates an appropriate
figure size and size for the actual plot window in normalized device coordinates
(i.e. in interval [0,1]). You can use these values in a script to set
the relevant values for Matplotlib as we show in the next example.

Example: wcsg_figuredemo.py - Plot figure in correct aspect ratio and fix the aspect ratio.

.. plot:: EXAMPLES/wcsg_figuredemo.py
   :include-source:
   :align: center

.. note::

   For astronomical data we want equal steps in spatial distance in any direction correspond
   to equal steps in figure size. If one changes the size of the figure interactively,
   the aspect ratio should not change. To enforce this, tell Matplotlib to keep
   the aspect ratio constant with keyword parameters *adjustable='box'* and
   *aspect='equal'* in constructors for Matplotlib Axes objects as in:

   `frame = fig.add_axes(grat.axesrect, aspect=grat.aspectratio, adjustable='box')` or:
   
   `frame = fig.add_subplot(1,1,1, aspect=grat.aspectratio, adjustable='box')`


Combinations of graticules
--------------------------

An object of class :class:`wcsgrat.Plotversion` is a container for graticules,
pixel labels and rulers.
The number of plotable objects is not restricted to one. One can easily add a
second graticule for a different sky system or a couple of rulers etc.
These are all added to the container with method :meth:`wcsgrat.Plotversion.add`
and the contents is plotted with method :meth:`wcsgrat.Plotversion.plot`.

The next example shows a combination of two graticules for two different sky systems.
It demonstrates also the use of attributes to changes plot properties.

Example: wcsg_skyout.py - Combine two graticules in one frame

.. plot:: EXAMPLES/wcsg_skyout.py
   :include-source:
   :align: center


**Explanation:**

This plot shows a graticule for equatorial coordinates and galactic coordinates in the
same figure. The center of the image is the position of the galactic pole. That is why
the graticule for the galactic system shows circles. The galactic graticule is also
labeled inside the plot using method :meth:`wcsgrat.Graticule.setinsidelabels`.
To get an impression of arbitrary positions expressed in pixels coordinates,
we added pixel coordinate labels for the top and right axes with
method :meth:`wcsgrat.Graticule.pixellabels`.


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
----------------------

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

.. plot:: EXAMPLES/wcsg_wave.py
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
 
Example: wcsg_spectraltypes.py - Plot grid lines for different spectral translations

.. plot:: EXAMPLES/wcsg_spectraltypes.py
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

Example: wcsg_manyrulers.py - Ruler demonstration

.. plot:: EXAMPLES/wcsg_manyrulers.py
   :include-source:
   :align: center

Ruler tick labels can be formatted so that we can adjust them. In the next plot we
want offsets to be plotted in arcminutes.

Example: wcsg_arminrulers.py - Rulers with non default labels

.. plot:: EXAMPLES/wcsg_arcminrulers.py
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

Example: wcsg_xvruler.py - Ruler in a XV map

.. plot:: EXAMPLES/wcsg_xvruler.py
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
