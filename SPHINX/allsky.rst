All sky plots and graticules
============================

.. highlight:: python
   :linenothreshold: 10

.. _allsky_tutorial:


All Sky plots
-------------

An all sky plot is a plot where the range in longitude is
[0,360> and the range in latitude is [-90,90>.
There are many examples in Calabretta's article
`Representations of celestial coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf>`_
We tried to reproduce these figures both to prove that the modules in the Kapteyn Package
have the functionality to do it and to facilitate users who want to set up
an all-sky plot with module :mod:`maputils`. For this purpose we
created for each figure the minimal required FITS headers.
Header and other code is listed below the examples.
Click on the hires link to get a plot which shows more details.
With the information in this document, it should be easy to compose
a Python program that creates just a single plot which then can be
enhanced to fit your needs.

The first plot is a stand alone version. The others are generated with
different Python scripts and the service module 
:download:`service.py <EXAMPLES/service.py>`
(see also the source code at the end of this document).

.. plot:: EXAMPLES/allsky_single.py
   :align: center
   :include-source:


**The recipe**

The next session shows a gallery of all sky plots, all based on the same recipe.

  * One starts with a self-made header which ensures a complete coverage of the sky by
    stretching the values of the ``CDELT``'s.
  * Then an object from class :class:`maputils.Annotatedimage.Graticule` is 
    created with explicit limits
    for the world coordinates in both directions.
  * For these plots we don't have intersections of the graticule with
    an enclosing rectangle so we cannot plot standard axis labels for the coordinates.
    Instead we use method :meth:`wcsgrat.Graticule.Insidelabels` to plot labels
    inside the plot. In the plots we show different examples how one can manipulate
    these labeling.



All sky plot gallery
--------------------

In the title of the plots we refer to the figure numbers in
Calabretta's article
`Representations of celestial coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf>`_.
These references start with the abbreviation **Cal.**. 

Note that the labels along the enclosing plot rectangle only indicate the
types for the longitude and latitude axes and their main direction.

The code which was used to produce a figure is listed just above the plot.
If you want to reproduce a plot then you need this source and the
service module 
:download:`service.py <EXAMPLES/service.py>`.

For plots where it is possible to plot a marker at position (120 deg, 60 deg)
we plot a small circle with: ``annim.Marker(pos=markerpos, marker='o', color='red')``
This code is part of the service module 
:download:`service.py <EXAMPLES/service.py>`

Note that positions in parameter *pos* in method :meth:`maputils.Annotatedimage.Marker`
can be entered in different formats. Have a look at :mod:`positions` for examples.

Definitions
++++++++++++

The definitions in this section are consistent with [Ref2]_ and  [Ref2]_ but simplified.
For the FITS keywords we ommitted the suffix for the axis number and the alternate
header description (e.g. as in CRVAL2Z).

   * CTYPE : Type of celestial system and projection system
   * CRPIX : Pixel coordinate of a coordinate reference point
   * CRVAL : The world coordinate at the reference point
   * CDELT : World coordinate increment at the reference point 
   * CUNIT : Units of the world coordinates. For celestial coordinates the 
     required units are 'deg'. However for wcs/WCSLIB the following lines
     are equal, because the units are parsed and converted to degrees if necessary:

     ``'CRVAL1' : 120.0*60, 'CRPIX1' : 50, 'CUNIT1' : 'arcmin', 'CDELT1' : 5.0*60``
     
     ``'CRVAL1' : 120.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : 5.0``

   * CROTA : Usually the angle in degrees between the axis for which CROTA is given
     and the North. Rotations can also be included in the FITS CD and PC matrix. 
   * The pixel position that corresponds to the values of CRPIX is denoted 
     with :math:`(r_x, r_y)`.
   * An arbitrary pixel coordinate is denoted with :math:`(p_x, p_y)`.
   * If we apply an operation M involving the PC or CD matrix or CDELT and CROTA
     we get :math:X = M.(p-r)`. The result is a position denoted with :math:`(x, y)` and
     its coordinates are called *intermediate world coordinates* and in this context 
     we refer to them as *projection plane coordinates*.
   * The coordinate reference point is at :math:`(x, y) = (0, 0)`.
   * Intermediate coordinates are in degrees. 
   * With the projection information in CTYPE one converts projection plane coordinates
     :math:`(x, y)` to *native longitude and latitude*, :math:`(\phi, \theta)`.
     Additional information for this transformation can be given in FITS PV keywords.
   * Native longitude and latitude :math:`(\phi, \theta)` are transformed 
     to *celestial coordinates*  :math:`(\alpha, \delta)` using the world coordinates
     (CRVAL) of the reference point.
   * The coordinates of the reference point given by CRVAL is denoted with 
     :math:`(\alpha_0, \delta_0)` If we assume that a longitude axis is associated
     with axis 1 in in the FITS file and latitude with axis 2 then
     :math:`(\alpha_0, \delta_0) = (CRVAL1, CRVAL2)`.
   * :math:`(\alpha_0, \delta_0)` is associated with the native longitude and latitude
     reference point :math:`(\phi_0, \theta_0)`
   * This :math:`(\phi_0, \theta_0)` depends on the projection or its values are  
     given in FITS PV
     keywords. With the previous assumption of the axis order, these elements will
     be :math:PV1_1 and PV1_2
   * For the transformation from native longitude and latitude :math:`(\phi, \theta)` 
     to *celestial coordinates*
     we need the **native** longitude and latitude of the **celestial** pole :math:`(\phi_p, \theta_p)`.
     Either defaults are taken or values are copied from FITS keywords LONPOLE, LATPOLE
     or PV1_3 and PV1_4.
     Also we need the **celestial** position of the **native** pole  :math:`(\alpha_p, \delta_p)`.
     (and :math:`\delta_p = \theta_p` ).
   
We summarize what van be varied in the FITS headers we used to plot all aky graticules.

   * :math:`(\alpha_0, \delta_0) \leftrightarrow  (CRVAL1, CRVAL2)`
   * native longitude and latitude of
     reference point :math:`(\phi_0, \theta_0)  \leftrightarrow  (PV1\_1, PV1\_2)`
   * native latitude of the celestial pole 
     :math:`(\phi_p, \theta_p) \leftrightarrow (LONPOLE, LATPOLE) \leftrightarrow (PV1\_3, PV1\_4)`


Standard versus Oblique
+++++++++++++++++++++++ 

Fig.0: Linear equatorial coordinate system
...........................................

Before the non-linear coordinate transformations, world coordinates were calculated in a
linear way using the number of pixels from the reference point in CRPIX times the 
increment in world coordinate and added to that the value of CRVAL. We demonstrate this
system bu creating a header where we omitted the code in CTYPE that sets the projection system.

WCSLIB does not recognize a valid projection system and defaults to linear transformations.
The header is a Python dictionary. With method :meth:`maputils.Annotatedimage.Graticule` we draw 
the graticules. The graticule lines that we want to draw are given by their start position
*startx=* and *starty=*. The labels inside the plot are set by *lon_world* and *lat_world*.
To be consistent with fig.2 in {Ref2]_, we inserted a positive CDELT for the longitude. 

.. note:: 

   In most of the figures in this section we plot position :math:`(120^\circ,60^\circ)`
   as a small solid red circle.

.. plot:: EXAMPLES/allskyf1A.py
   :align: center
   :include-source:



Fig.1: Oblique Plate Carree projection (CAR)
............................................

In [Ref2]_ we read that only CTYPE needs to be changed to get the next figure.
For CTYPE the projection code *CAR* is added. For a decent plot we need to draw a border.
The trick for plotting borders for oblique versions is to change header values
to the non-oblique version and then to draw only the limiting graticule lines. 
In method :meth:`maputils.Annotatedimage.Graticule` 
we use parameters *startx* and *starty* to specify these limits as in:
``startx=(180-epsilon,-180+epsilon), starty=(-90,90))``

This plot shows an oblique version. A problem with oblique all sky plots
is drawing a closed border. The trick that we applied a number of times
is to overlay the border of the non-oblique version. 

.. plot:: EXAMPLES/allskyf1.py
   :align: center
   :include-source:


Fig.2: Plate Carree projection non-oblique (CAR)
.................................................

To get a non oblique version of the previous system we need to change the 
value of :math:`\delta_0` (as given in CRVAL2) to 0 because for this projection 
:math:`\phi_p = 0`.
In the header we changed CRVAL2 to 0.

.. plot:: EXAMPLES/allskyf2.py
   :align: center
   :include-source:


Zenithal projections
++++++++++++++++++++

Fig.3: Slant zenithal (azimuthal) perspective projection (AZP)
..............................................................

This figure shows a projection for which we need to specify extra
parameters in the so called PV header keywords as in:
``'PV2_1'  : mu, 'PV2_2'  : gamma``
It uses a formula given in Calabretta's article to get a value for the border:
``lowval = (180.0/numpy.pi)*numpy.arcsin(-1.0/mu) + 0.00001``

.. plot:: EXAMPLES/allskyf3.py
   :align: center
   :include-source:


Fig.4: Slant zenithal perspective (SZP)
.......................................

The plot shows two borders. We used different colors to distinguish them.
The cyan colored border is calculated with a border formula given in [Ref2]_ and the
red border is calculated with a brute force method :meth:`wcsgrat.Graticule.scanborder`
which uses a bisection method in X and Y direction to find the position
of a transition between a valid world coordinate and an invalid coordinate.
Obviously the border that is plotted accoding to the algorithm is less accurate.
The brute force method gives a more accurate border but needs a user to
enter start positions for the bisection.

.. plot:: EXAMPLES/allskyf4.py
   :align: center
   :include-source:   

Fig.5: Gnomonic projection (TAN)
................................

In a Gnomonic projection all great circles are
projected as straight lines.
This is nice example of a projection which diverges at certain latitude.
We chose to draw the last border at 20 deg. and plotted it with dashes
using method :meth:`wcsgrat.Graticule.setp_lineswcs1` as in 
``grat.setp_lineswcs1(20, color='g', linestyle='--')`` and identified
the graticule line with its position i.e. latitude 20 deg.

.. plot:: EXAMPLES/allskyf5.py
   :align: center
   :include-source:   

Fig.6: Stereographic projection (STG)
.....................................

.. plot:: EXAMPLES/allskyf6.py
   :align: center
   :include-source:   

Fig.7: Slant orthographic projection (SIN)
.............................................

The green colored border is calculated with a border formula given in [Ref2]_

.. plot:: EXAMPLES/allskyf7.py
   :align: center
   :include-source:   

Fig.8: Zenithal equidistant projection (ARC)
............................................

.. plot:: EXAMPLES/allskyf8.py
   :align: center
   :include-source:      

Fig.9: Zenithal polynomial projection (ZPN)
...........................................

Diverges at some latitude depending on the selected parameters
in the PV keywords. Note that the inverse of the polynomial 
cannot be expressed analytically and there is no function that can transform 
our marker at :math:`(120^\circ,60^\circ)` to pixel coordinates.

.. plot:: EXAMPLES/allskyf9.py
   :align: center
   :include-source:   

Fig.10: Zenith equal area projection (ZEA)
..........................................

.. plot:: EXAMPLES/allskyf10.py
   :align: center
   :include-source:   

Fig.11: Airy projection (AIR)
.............................

.. plot:: EXAMPLES/allskyf11.py
   :align: center
   :include-source:


Cylindrical Projections
+++++++++++++++++++++++

The native coordinate system origin of a Cylindrical projections coincides 
with the reference point. Therefore we set  :math:`(\phi_0, \theta_0) = (0,0)` 

Fig.12: Gall's stereographic projection (CYP)
.............................................
       
.. plot:: EXAMPLES/allskyf12.py
   :align: center
   :include-source:

Fig.13: Lambert's equal area projection (CEA)
.............................................

.. plot:: EXAMPLES/allskyf13.py
   :align: center
   :include-source:

Fig.14: Plate Carree projection (CAR)
.....................................

.. plot:: EXAMPLES/allskyf14.py
   :align: center
   :include-source:

Fig.15: Mercator's projection (MER)
...................................

.. plot:: EXAMPLES/allskyf15.py
   :align: center
   :include-source:

Pseudocylindrical projections
++++++++++++++++++++++++++++++

Fig.16: Sanson-Flamsteed projection (SFL)
..........................................

.. plot:: EXAMPLES/allskyf16.py
   :align: center
   :include-source:

Fig.17: Parabolic projection (PAR)
..................................

.. plot:: EXAMPLES/allskyf17.py
   :align: center
   :include-source:

Fig.18: Mollweide's projection (MOL)
....................................

.. plot:: EXAMPLES/allskyf18.py
   :align: center
   :include-source:

Fig.19: Hammer Aitoff projection (AIT)
......................................

.. plot:: EXAMPLES/allskyf19.py
   :align: center
   :include-source:

Conic projections
+++++++++++++++++

Fig.20: Conic perspective projection (COP)
..........................................

.. plot:: EXAMPLES/allskyf20.py
   :align: center
   :include-source:

Fig.21: Conic equal area projection (COE)
.........................................

.. plot:: EXAMPLES/allskyf21.py
   :align: center
   :include-source:

Fig.22: Conic equidistant projection (COD)
..........................................

.. plot:: EXAMPLES/allskyf22.py
   :align: center
   :include-source:

Fig.23: Conic orthomorfic projection (COO)
..........................................

.. plot:: EXAMPLES/allskyf23.py
   :align: center
   :include-source:

Polyconic and pseudoconic projections
+++++++++++++++++++++++++++++++++++++

Fig.24: Bonne's equal area projection (BON)
...........................................

.. plot:: EXAMPLES/allskyf24.py
   :align: center
   :include-source:

Fig.25: Polyconic projection (PCO)
..................................

Near the poles we have a problem to draw graticule lines at constant latitude.
With the defaults for the Graticule constructor we would observe a
horizontal line that connects longitudes -180 and 180 deg. near the poles.
>From a plotting point of view the range -180 to 180 deg. means a closed
shape (.e.q. a circle near a pole).
To prevent horizontal jumps in such plots we defined a jump in terms of pixels.
If the distance between two points is much bigger than the pixel sampling
then it must be a jump. However, in some projections (like this one), the
jump near the pole becomes so small that we cannot avoid a horizontal connection.
By increasing the number of samples in parameter *gridsamples* we force
the size of a jump relatively to be bigger. With a value *gridsamples=2000*
we avoid the unwanted connections.

The reason that sometimes line sections are connected which
are not supposed to be connected has to
do with the fact that in :mod:`wcsgrat` the range in world coordinates is increased a little
bit to be sure we cross borders so that we are able to plot
ticks. But in the gaps (see the plot below) this can result in the fact that we start
to sample on the wrong side of the gap. Then there is a gap
and the sampling continues on the other side of the gap. The algorithm
thinks these points should be connected because the gap is too small
to be detected as a jump.

Note that we could also have decreased the size of the range
in world coordinates in longitude (e.g. ``wxlim=(-179.9, 179.9)``) but this
results in small gaps near all borders.


.. plot:: EXAMPLES/allskyf25.py
   :align: center
   :include-source:

Quad cube projections projections
+++++++++++++++++++++++++++++++++

Fig.26: Tangential spherical cube projection (TSC)
..................................................

For all the quad cube projections we plotted a border
by converting edges in world coordinates into pixels coordinates
and connected them in the right order. 

.. plot:: EXAMPLES/allskyf26.py
   :align: center
   :include-source:

Fig.27: COBE quadrilateralized spherical cube projection (CSC)
..............................................................

.. plot:: EXAMPLES/allskyf27.py
   :align: center
   :include-source:

Fig.28: Quadrilateralized spherical cube projection (QSC)
..........................................................

.. plot:: EXAMPLES/allskyf28.py
   :align: center
   :include-source:

Oblique projections
+++++++++++++++++++

Fig.29: Zenith equal area projection (ZEA) oblique
.....................................................

.. plot:: EXAMPLES/allskyf29.py
   :align: center
   :include-source:

Fig.30: Zenith equal area projection (ZEA) oblique
..................................................

.. plot:: EXAMPLES/allskyf30.py
   :align: center
   :include-source:

Fig.31: Zenith equal area projection (ZEA) oblique with PV1_3 element
......................................................................

.. plot:: EXAMPLES/allskyf31.py
   :align: center
   :include-source:

Fig.32: Zenith equal area projection (ZEA) oblique with PV1_3 element II
.........................................................................

.. plot:: EXAMPLES/allskyf32.py
   :align: center
   :include-source:

Fig.33: Conic equidistant projection (COD) oblique
..................................................

.. plot:: EXAMPLES/allskyf33.py
   :align: center
   :include-source:

Fig.34: Hammer Aitoff projection (AIT) oblique
...............................................

.. plot:: EXAMPLES/allskyf34.py
   :align: center
   :include-source:

Fig.35: COBE quadrilateralized spherical cube projection (CSC) oblique
......................................................................

.. plot:: EXAMPLES/allskyf35.py
   :align: center
   :include-source:

Miscellaneous
+++++++++++++

Fig.36: Earth in zenithal perspective (AZP)
...........................................

The coastline used in this example is read from file
:download:`world.txt <EXAMPLES/WDB/world.txt>` which is composed
from a plain text version of the CIA World DataBank II map database
made by Dave Pape (http://www.evl.uic.edu/pape/data/WDB/).

We used values 'TLON', 'TLAT' for the ``CTYPE``'s.
These are recognized by WCSlib as longitude and latitude.
Any other prefix is also valid.

Note the intensive use of methods to set label/tick- and plot properties.

   * :meth:`wcsgrat.Graticule.setp_lineswcs0`, 
   * :meth:`wcsgrat.Graticule.setp_lineswcs1`, 
   * :meth:`wcsgrat.Graticule.setp_tick` and
   * :meth:`wcsgrat.Graticule.setp_linespecial`

.. plot:: EXAMPLES/allskyf36.py
   :align: center
   :include-source:

Fig.37: WCS polyconic
......................

Without any tuning, we would observe jumpy behaviour near the
dicontinuity of the green border. The two vertical parts would
be connected by a small horizontal line. We can improve the plot by
increasing the value of parameter *gridsamples* in the Graticule constructor
from 1000 (which is the default value) to 4000.
See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/PGSBOX/index.html

.. plot:: EXAMPLES/allskyf37.py
   :align: center
   :include-source:

Fig.38: WCS conic equal area projection
.......................................

See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/PGSBOX/index.html

.. plot:: EXAMPLES/allskyf38.py
   :align: center
   :include-source:

Fig.39: Bonne's equal area projection (BON) II

See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/

.. plot:: EXAMPLES/allskyf39.py
   :align: center
   :include-source:


Projection aliases
+++++++++++++++++++

Table A.1. in [Ref2]_ list many alternative projection. These are either one of the 
projections listed in previous sections or they have special projection parameters.
This table lists those parameters and in table 13 of [Ref2]_ one can find the corresponding
PV keyword.

For example if we want a Peter's projection then in table A.1. we read that this is in fact 
a Gall's orthographic projection (CEA) with :math:`\lambda = 1/2`. In table 13 we find that for
projection CEA the parameter :math:`\lambda` corresponds to keyword PVi_1a. This keyword is associated with the
latitude axis. If this is the second axis in a FITS header and if we don't use an alternate
header description, then this keyword is PV1_1.


Source code of the service module program
------------------------------------------

.. literalinclude:: EXAMPLES/service.py

