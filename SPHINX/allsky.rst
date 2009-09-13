All sky plots and graticules
============================

.. highlight:: python
   :linenothreshold: 10

All Sky plots
-------------

An all sky plot is a plot where the longitude is in the range
[0,360> and the latitude in the range [-90,90>.
There are many examples in Calabretta's article
`Representations of celestial coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf>`_
We tried to reproduce these figures both to prove that the modules in the Kapteyn Package
have the functionality to do it and to facilitate users who want to set up
an all-sky plot with module :mod:`wcsgrat`. For this purpose we
created for each figure minimal FITS headers.
Header and other code is listed below the examples.
Click on the hires link to get a plot which shows more details.
With the information in this document, it should be easy to compose
a Python program that creates just a single plot which then can be
enhanced to fit your needs.

The first plot is a stand alone version. The others are generated with
Python aplplication :download:`allsky.py <EXAMPLES/allsky.py>`
(see also the source code at the end of this document).

.. plot:: EXAMPLES/allsky_single.py
   :include-source:


**The recipe**

The next session shows a gallery of all sky plots, all based on the same recipe.
Usually these plots need a lot of fine tuning. Where we added code to the recipe below,
it will be marked with a comment.

  * One starts with a self-made header which ensures a complete coverage of the sky by
    stretching the values of the ``CDELT``'s.
  * Then an object from class :class:`wcsgrat.Graticule` is created with explicit limits
    for the world coordinates in both directions.
  * For these plots we don't have intersections of the graticule with
    an enclosing rectangle so we cannot plot standard labels for the coordinates.
    Instead we use method :meth:`wcsgrat.Graticule.setinsidelabels` to plot labels
    inside the plot. Usually these labels are unformatted.



All sky plot gallery
--------------------

In the title of the plots we refer to the figure numbers in
Calabretta's article
`Representations of celestial coordinates in FITS <http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf>`_.
The figure numbers for the plots itself
correspond to the figure numbers in this code of :download:`allsky.py <EXAMPLES/allsky.py>`.

Note that the labels along the enclosing plot rectangle only indicate the
types for the longitude and latitude axes and their main direction.


Standard versus Oblique
+++++++++++++++++++++++ 

Fig.1: Plate Carree projection (CAR)
....................................

With ``pixel = grat.gmap.topixel((120.0,60))`` we marked
a world coordinate. We used method :meth:`wcs.Projection.topixel`
of the :mod:`wcs.Projection` class to convert a world coordinate into a pixel coordinate.
The Projection object in this code is *grat.gmap*.


.. plot:: EXAMPLES/allskyfig1.py


Fig.2: Oblique Plate Carree projection (CAR)
............................................

This plot shows an oblique version. A problem with oblique all sky plots
is drawing a closed border. The trick that we applied a number of times
is to overlay the border of the non-oblique version. In the constructor
of the :class:`wcsgrat.Graticule` object we use parameters *startx* and *starty*
to specify the borders as in:
``startx=(180-epsilon,-180+epsilon), starty=(-90,90))``

.. plot:: EXAMPLES/allskyfig2.py
   


Zenithal projections
++++++++++++++++++++

Fig.3: Slant zenithal (azimuthal) perspective projection (AZP)
..............................................................

This figure shows a projection for which we need to specify extra
parameters in the so called PV header keywords as in:
``'PV2_1'  : mu, 'PV2_2'  : gamma``
It uses a formula given in Calabretta's article to get a value for the border:
``lowval = (180.0/numpy.pi)*numpy.arcsin(-1.0/mu) + 0.00001``

.. plot:: EXAMPLES/allskyfig3.py
   


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

.. plot:: EXAMPLES/allskyfig4.py
   

Fig.5: Gnomonic projection (TAN)
................................

This is nice example of a projection which diverges at certain latitude.
We chose to draw the last border at 20 deg. and plotted it with dashes
using method :meth:`wcsgrat.Graticule.setp_linewcs1` as in 
``grat.setp_lineswcs1(20, color='g', linestyle='--')`` and identified
the graticule line with its position i.e. latitude 20 deg.


.. plot:: EXAMPLES/allskyfig5.py
   

Fig.6: Stereographic projection (STG)
.....................................

.. plot:: EXAMPLES/allskyfig6.py
   

Fig.7: Slant orthograpic projection (SIN)
.........................................

The green colored border is calculated with a border formula given in [Ref2]_

.. plot:: EXAMPLES/allskyfig7.py
   

Fig.8: Zenithal equidistant projection (ARC)
............................................

.. plot:: EXAMPLES/allskyfig8.py
   

Fig.9: Zenithal polynomial projection (ZPN)
...........................................

Diverges at some latitude depending on selected parameters
in the PV elements.

.. plot:: EXAMPLES/allskyfig9.py
   

Fig.10: Zenith equal area projection (ZEA)
..........................................

.. plot:: EXAMPLES/allskyfig10.py
   

Fig.11: Airy projection (AIR)
.............................

.. plot:: EXAMPLES/allskyfig11.py
   


Cylindrical Projections
+++++++++++++++++++++++

Fig.12: Gall's stereographic projection (CYP)
.............................................
       
.. plot:: EXAMPLES/allskyfig12.py
   

Fig.13: Lambert's equal area projection (CEA)
.............................................

.. plot:: EXAMPLES/allskyfig13.py
   

Fig.14: Plate Carree projection (CAR)
.....................................

.. plot:: EXAMPLES/allskyfig14.py
   

Fig.15: Mercator's projection (MER)
...................................

.. plot:: EXAMPLES/allskyfig15.py
   

Fig.16: Sanson-Flamsteed projection (SFL)
..........................................

.. plot:: EXAMPLES/allskyfig16.py
   

Fig.17: Parabolic projection (PAR)
..................................

.. plot:: EXAMPLES/allskyfig17.py
   

Fig.18: Mollweide's projection (MOL)
....................................

.. plot:: EXAMPLES/allskyfig18.py
   

Fig.19: Hammer Aitoff projection (AIT)
......................................

.. plot:: EXAMPLES/allskyfig19.py
   

Conic projections
+++++++++++++++++

Fig.20: Conic perspective projection (COP)
..........................................

.. plot:: EXAMPLES/allskyfig20.py
   

Fig.21: Conic equal area projection (COE)
.........................................

.. plot:: EXAMPLES/allskyfig21.py
   

Fig.22: Conic equidistant projection (COD)
..........................................

.. plot:: EXAMPLES/allskyfig22.py
   

Fig.23: Conic orthomorfic projection (COO)
..........................................

.. plot:: EXAMPLES/allskyfig23.py
   

Polyconic and pseudoconic projections
+++++++++++++++++++++++++++++++++++++

Fig.24: Bonne's equal area projection (BON)
...........................................

.. plot:: EXAMPLES/allskyfig24.py
   

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

 

.. plot:: EXAMPLES/allskyfig25.py
   

Quad cube projections projections
+++++++++++++++++++++++++++++++++

Fig.26: Tangential spherical cube projection (TSC)
..................................................

For all the quad cube projections we plotted a border
by converting edges in world coordinates into pixels coordinates
and connected them in the right order. 

.. plot:: EXAMPLES/allskyfig26.py
   

Fig.27: COBE quadrilateralized spherical cube projection (CSC)
..............................................................

.. plot:: EXAMPLES/allskyfig27.py
   

Fig.28: Quadrilateralized spherical cube projection (QSC)
..........................................................

.. plot:: EXAMPLES/allskyfig28.py
   

Oblique projections
+++++++++++++++++++

Fig.29: Zenith equal area projection (ZEA) oblique
.....................................................

.. plot:: EXAMPLES/allskyfig29.py
   

Fig.30: Zenith equal area projection (ZEA) oblique
..................................................

.. plot:: EXAMPLES/allskyfig30.py
   

Fig.31: Zenith equal area projection (ZEA) oblique with PV1_3 element
......................................................................

.. plot:: EXAMPLES/allskyfig31.py
   

Fig.32: Zenith equal area projection (ZEA) oblique with PV1_3 element II
.........................................................................

.. plot:: EXAMPLES/allskyfig32.py
   

Fig.33: Conic equidistant projection (COD) oblique
..................................................

.. plot:: EXAMPLES/allskyfig33.py
   

Fig.34: Hammer Aitoff projection (AIT) oblique
...............................................

.. plot:: EXAMPLES/allskyfig34.py
   

Fig.35: COBE quadrilateralized spherical cube projection (CSC) oblique
......................................................................

.. plot:: EXAMPLES/allskyfig35.py
   

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
:meth:`wcsgrat.Graticule.setp_lineswcs0`, 
:meth:`wcsgrat.Graticule.setp_lineswcs1`, 
:meth:`wcsgrat.Graticule.setp_tick` and
:meth:`wcsgrat.Graticule.setp_linespecial`

.. plot:: EXAMPLES/allskyfig36.py
   

Fig.37: WCS polyconic
......................

Without any tuning, we would observe jumpy behaviour near the
dicontinuity of the green border. The two vertical parts would
be connected by a small horizontal line. We can improve the plot by
increasing the value of parameter *gridsamples* in the Graticule constructor
from 1000 (which is the default value) to 4000.
See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/PGSBOX/index.html

.. plot:: EXAMPLES/allskyfig37.py
   

Fig.38: WCS conic equal area projection
.......................................

See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/PGSBOX/index.html

.. plot:: EXAMPLES/allskyfig38.py
   

Fig.39: Bonne's equal area projection (BON) II

See equivalent plot at http://www.atnf.csiro.au/people/mcalabre/WCS/

.. plot:: EXAMPLES/allskyfig39.py
   



Source code of all sky plot program
-----------------------------------

.. literalinclude:: EXAMPLES/allsky.py

