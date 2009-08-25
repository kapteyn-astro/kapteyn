Tutorial wcsgrat module
=======================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------

Module :mod:`wcsgrat` creates a graticule for a given header with WCS information.
That implies that it finds positions on a curve in 2-dimensional data
for which one of the world coordinates is a constant value.
These positions are stored in an object derived from class
:class:`wcsgrat.Graticule`. The positions at which these lines cross one of the sides
of the rectangle (made up by the limits in pixels in both x- and y-direction),
are stored in an object from class :class:`wcsgrat.Graticule.WCStick`, together with a
text label showing the world coordinate of the crossing. In principle this is
all what this module does. To make it more useful we added a class
with a method that plots the graticule. Currently we support *Matplotlib*
as plot software, but similar methods for other plot packages
should be possible to add (e.g. ppgplot).


Simple example
--------------

Example: ex_axnumdemo.py - Simple plot using defaults

.. plot:: EXAMPLES/ex_axnumdemo.py
   :include-source:
   :align: center

**Explanation:**

The script opens an existing FITS file. Its header is parsed by methods
in module :mod:`wcs` and methods from classes in module :mod:`wcsgrat`
calculate graticule data. A plot is made with Matplotlib.

The recipe:

   * Given a FITS file on disk (example1test.fits) we want to plot
     a :term:`graticule` for the spatial axes in the FITS file.
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

Example: ex_manyaxes.py - Selecting WCS axes from a FITS file

.. plot:: EXAMPLES/ex_manyaxes.py
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

If you create an object from :class:`wcsgrat.Graticule`, you can modify
its behaviour in many ways using keyword arguments.
In another example we show how to mix spatial and non spatial axes.


More 'axnum' variations -- Position Velocity diagrams
-----------------------------------------------------

For the next example we used a FITS file with the following header information::

   Axis 1: RA---NCP  from pixel 1 to   100  {crpix=51 crval=-51.2821 cdelt=-0.007166 (DEGREE)}
   Axis 2: DEC--NCP  from pixel 1 to   100  {crpix=51 crval=60.1539 cdelt=0.007166 (DEGREE)}
   Axis 3: VELO-HEL  from pixel 1 to   101  {crpix=-20 crval=-243 cdelt=4.2 (km/s)}

Example: ex_axnumdemo.py - Show different axes combinations for the same FITS file

.. plot:: EXAMPLES/ex_axnumdemo.py
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

   grat3.setinsidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10, color='r')
   grat3.setinsidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')

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

Example: ex_figuredemo.py - Plot figure in correct aspect ratio and fix the aspect ratio.

.. plot:: EXAMPLES/ex_figuredemo.py
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

Example: ex_skyout.py - Combine two graticules in one frame

.. plot:: EXAMPLES/ex_skyout.py
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
header we found that the optical velocity given by DRVAL3 or VELR is 1050 Km/s.
The header is a legacy GIPSY header and module :mod:`wcs` can parse it.
We require the frequencies to be expressed as wavelengths.

Example: ex_wave.py - Plot a graticule in a position wavelength diagram.

.. plot:: EXAMPLES/ex_wave.py
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
   

For the next example we use the same FITS file (mclean.fits) and demonstrate how
to retrieve allowed spectral translations for this data and how to create plots
with graticules for maps with one spatial axis and one spectral axis.

Example: ex_spectraltypes.py - Plot grid lines for different spectral translations

.. plot:: EXAMPLES/ex_spectraltypes.py
   :include-source:
   :align: center


**Explanation:**

  * With PyFITS we open the fits file on disk and read its header
  * We created a :class:`wcs.Projection` object for this header to get a 
    list with allowed spectral translations (attribute *altspec*). We need
    this list before we create any graticules 
  * A Matplotlib Figure- and Axes instance are made. For each plot we create 
    a new Axes instance with Matplotlib's method *add_subplot()*.
  * The first plot (top) represents the native spectral coordinate. This is a topocentric
    frequency (CTYPE3='FREQ-OHEL'). The second plot has also frequency as its spectral axis but
    this is a converted helocentric frequency (Using the fact that an optical velocity of 1050 Km/s
    is given for the heliocentric system).
    The conversion is set with parameter *spectrans*.
    
We centered the plot in pixel coordinates around CRPIX3. So we expect that the
corresponding value in optical velocity (1050 Km/s) appears in the center
of the plot for VOPT. The figure above confirms this.
  
  
Plotting wcs Rulers
-------------------

In publications one often encounters images of astronomical objects with
a small ruler added to mark offsets in spatial distance. Usually these 
objects are small enough to allow a linear ruler. However for bigger objects 
and some projections, these ruler are not accurate enough. We implemented rulers
that are accurate for all sizes and all projections.

A ruler object from module :mod:`wcsgrat` is created with method
:meth:`wcsgrat.Graticule.ruler`. It needs a starting point and an end point
in either pixel coordinates or world coordinates.
You need also to enter the position at which we want the offset to be 0
(with parameter *lamda0*).
The ruler applies to a spatial system (or to a XV map) and therefore
the units of the offsets are degrees. One can alter this by 
entering a function with parameter *fun* and a format in *fmt*.

Ruler fine tuning
.................

Plotting rulers is a bit more difficult than plotting other
objects because its defaults must cover many different situations 
and is therefore less useful. In the next example we show how to
'fine tune' a ruler.

Example: ex_ruleroffset.py - Put a ruler with distance offsets in arcmin.
 
.. plot:: EXAMPLES/ex_ruleroffset.py
   :include-source:
   :align: center


**Explanation:**
Most of the lines are discussed in other examples. We can focus on the call to
method :meth:`wcsgrat.Graticule.ruler` as in:

`ruler3 = grat.ruler(23*15,30,22*15,15, 0.5, 2, world=True, fmt=r"$%6.0f^\prime$", fun=lambda x: x*60.0, mscale=4.5)`

Let's discuss each parameter:
   
   1. `23*15` : Start value of ruler in world coordinates. Value is 23 hours * 15 degrees.
   2. `30` : Start point in world coordinates for the declination in degrees.
   3. `22*15` : End point in Right Ascension.
   4. `15` End point in declination.
   5. `0.5` : Offset 0 is exactly in the center of the ruler 
   6. `2` : Step size for offset labels is 2 degrees
   7. `world=True` : The start- and end points are in world coordinates
   8. `fmt=r"$%6.0f^\prime$"` : We want to format the offset labels. The field width
      seems to be unimportant when we format the string in TeX. The TeX string
      must be a Python raw string and starts character 'r'. The prime symbol
      is used to indicate that the offset is in minutes of arc.
   9. `fun=lambda x: x*60.0` : The default units are degrees. To print offsets
      in minutes of arc, multiply the values with 60.
   10. `mscale=4.5` : A scale factor to move the label to or from the ruler
       to create a better layout for the offset labels. 


Ruler for headers with alternative units
........................................

The input parameters for start- and end point and the step size
in the constructor of a ruler are always in the same units (i.e. degrees).
even when the units in the header are not degrees. The next example 
shows this fact. 

Example: ex_rulerarcmins.py - Ruler with non standard header units
 
.. plot:: EXAMPLES/ex_rulerarcmins.py
   :include-source:
   :align: center

The example shows that it is possible to change the properties of the tick labels
to facilitate interpretation of the plot. For instance we distinguish Right Ascensions
and declinations by color and we prevent two labels to intersect by rotating one of them.
The method we used is :meth:`wcsgrat.Graticule.setp_tick`. It applies changes on world
coordinate axes, but this can be refined by setting a 'plotaxis' as in:
   
`grat.setp_tick(wcsaxis=1, plotaxis=(wcsgrat.bottom), rotation=30, ha='right')`

The keyword arguments *rotation* and *ha* are Matplotlib parameters.

Ruler in an Position-Velocity map
.................................

Example: ex_rulerxvmap.py - A ruler in a map with only one spatial axis
 
.. plot:: EXAMPLES/ex_rulerxvmap.py
   :include-source:
   :align: center


**Explanation:**

This is an example of a ruler in a Position-Velocity diagram. The header data belongs
to a FITS file with axes Right Ascension, declination and frequency (RA,DEC,FREQ).
We did not specify at which Right Ascension we made a slice, so the value of
CRPIX1 is assumed. For that pixel coordinate we plot the declinations along
the left y axis and offsets along the right y axis. Note that the default 
in the Graticule constructor plots offsets along the y axis because this plot
has only one spatial axis. But the map is not rotated so we can plot the declination
axis as usual without confusing the reader. Note also the use of the methods
that change the properties of some objects. We used:
    
    * :meth:`wcsgrat.Graticule.ruler.setp_line`
    * :meth:`wcsgrat.Graticule.ruler.setp_labels`
    * :meth:`wcsgrat.Graticule.setp_plotaxis`
    * :meth:`wcsgrat.Graticule.setp_tick`
    * :meth:`wcsgrat.Graticule.setp_gratline`

.. note::
   
   Methods *setp_tick()* and *setp_gratline()* change properties of one of the two
   world coordinate axes (usually in a not rectangular system) while
   *setp_plotaxis()* changes properties of one of the four plot axes (rectangular system).
   
   

Glossary
--------

.. glossary::

   graticule
      the network of lines of latitude and longitude upon which a map is drawn

