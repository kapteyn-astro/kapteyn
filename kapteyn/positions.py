#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    positions.py
# PURPOSE: Provides functions for the conversion of positions to grids
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    Nov 20, 2009
# UPDATE:  Nov 20, 2009
# VERSION: 0.1
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#----------------------------------------------------------------------

"""
Module positions
================


In module :mod:`wcs` we provided two methods of the Projection object for
transformations between pixels and world coordinates. These methods are
:meth:`wcs.Projection.topixel` and :meth:`wcs.Projection.toworld` and they
allow (only) numbers as their input parameters. These transformation methods apply to
the coordinate system for which the Projection object is created and it is not
possible to enter world coordinates from other sky systems or with other units.

Often one wants more flexibility. For instance, in interaction with the user, positions
can be used to plot markers on a map or to preset the location of labels and
graticule lines. But what to do if you have positions that need to be marked and the
positions are from a FK5 catalog while your current map is given in
Galactic coordinates? Or what to do if you need to know,
given a radio velocity, what the optical velocity is for a spectral axis
which has frequency as its primary type? For these situations we
wrote function :func:`str2pos`.

This module enables a user/programmer to specify positions in either
pixel- or world coordinates. Its functionality is provided by a parser
which converts strings with position information into pixel coordinates
and world coordinates. Let's list some options with examples how to use
function :func:`str2pos` which is the most important method in this module.

Assume we have a projection object *pr* and you
want to know the world coordinates *w* and the pixels *p* for a given
string. Further, assume *u* are the units of the world coordinates
and *e* is an error message. Both *u* and *e* are output parameters.
Here are some examples how to use
:func:`str2pos`. We will give detailed descriptions of the options
in later sections.

   * | Expressions for the input of **numbers**.
     | Example: ``w,p,u,e = str2pos('[pi**2::3], [1:3]', pr)``
   * | Use of **physical constants**.
     | Example: ``w,p,u,e = str2pos('c_/299792458.0,  G_/6.67428e-11', pr)``
   * | Use of **units** to set world coordinates
     | Example: ``w,p,u,e = str2pos('178.7792 deg  53.655 deg', pr)``
   * | **Mix** of pixels and world coordinates.
     | Example: ``w,p,u,e = str2pos('5.0, 53.655 deg', pr)``
   * | Support of **sky definitions**.
     | Example: ``w,p,u,e = str2pos('{eq, B1950,fk4, J1983.5} 178.12830409  {} 53.93322241', pr)``
   * | Support for **spectral translations**.
     | Example: ``w,p,u,e = str2pos('vopt 1050 km/s', pr)``
   * | Coordinates from **text file** on disk.
     | Example: ``w,p,u,e = str2pos('readcol("test123.txt", col=2)', pr)``
   * | Support for maps with only **one spatial** axis (e.g. XV maps).
     | Example: ``w,p,u,e = str2pos('{} 53.655 1.415418199417E+03 Mhz', pr, mixpix=6)``
   * | Use of **sexagesimal** notation of spatial world coordinates.
     | Example: ``w,p,u,e = str2pos('11h55m07.008s 53d39m18.0s', pr)``
   * | Read **header** items.
     | Example: ``w,p,u,e = str2pos("{} header('crval1') {} header('crval2')", pr)``
   * Units, sky definitions and spectral translations are case insensitive and
     **minimal matched** to the full names.

Examine next small script that uses the syntax described in this document to
set marker positions:

**Example: mu_markers.py - Demonstrate the use of strings for a position**

.. literalinclude:: EXAMPLES/mu_markers.py




Introduction
------------

Physical quantities, in a data structure which represents a measurement of an astronomical 
phenomenon, are usually
measurements at fixed positions in the sky, sometimes at some spectral value such as a
Doppler shift, frequencies or velocity. These positions are examples of so called
**World Coordinates**. To identify a world coordinate in a measured data structure,
we use a coordinate system based on the pixels in that structure. Often the data
structures are FITS files and the coordinate system is subject to a set of rules.
For FITS files the first pixel on an axis is labeled with coordinate
1 and it runs to the value of *NAXISn* which is a FITS header item
that sets the length of the n-th axis in the data structure.

Assume you have a data structure representing an optical image of a part of the sky
and you need to mark a certain feature in the image or need to retrieve the intensity
of a pixel at a certain location. Then it is possible to identify the pixel using
pixel coordinates. But when you have positions from external sources like
catalogs, then these are not related to a FITS file and therefore given in world
coordinates coupled to a certain coordinate system (e.g. a sky system).
Then it would be convenient if you could specify positions exactly in those coordinates.

This module uses two other modules from the Kapteyn Package:
Module :mod:`wcs` provides methods for conversions between
pixel coordinates and world coordinates given a description of the world coordinate
system as defined in a (FITS) header). Module :mod:`celestial` converts world coordinates
between different sky- and reference systems and/or epochs.
In this module we combine the functionality of :mod:`wcs` and :mod:`celestial`
to write a coordinate parser to convert world coordinates to pixel coordinates (and back)
given a header that describes the WCS.
Note that a description of a world coordinate system can be either derived from a FITS header or
a Python dictionary with FITS keywords.


How to use this module
----------------------

This module is used in several modules of the Kapteyn Package, but
it can also be imported in your own scripts so that you are able to convert
positions (given as a string) to pixel- and world coordinates.
It is also possible to use this module as a test application.
If you want to see the test run then 
type: ``python positions.py`` on the command line.
The source of the test strings with positions can be found in function :func:`dotest` in this module.

To get the idea, we list a short example starting with the definition of a header::

   from kapteyn import wcs, positions

   header = { 'NAXIS'  : 2,
              'CDELT1' : -1.200000000000E-03, 'CDELT2' : 1.497160000000E-03,
              'CRPIX1' : 5, 'CRPIX2' : 6,
              'CRVAL1' : 1.787792000000E+02, 'CRVAL2' : 5.365500000000E+01,
              'CTYPE1' : 'RA---NCP', 'CTYPE2' : 'DEC--NCP',
              'CUNIT1' : 'DEGREE', 'CUNIT2' : 'DEGREE',
              'NAXIS1' : 10, 'NAXIS2' : 10,
            }
   
   pr = wcs.Projection(header)
   w,p,u,e = positions.str2pos('5, 6', pr)
   if e == '':
      print "pixels:", p
      print "world coordinates:", w, u

Its output (which is always a NumPy array) is::

   pixels: [[ 5.  6.]]
   world coordinates: [[ 178.7792   53.655 ]] ('deg', 'deg')

Remember, *p* are the pixel coordinates, *w* the world coordinates and *u*
is a tuple with units.
We have valid coordinates if the string *e* is empty.
If it is not empty then there is an error condition and the string is an error message.
The parser does not raise exceptions but it stores a message after an exception
in the error message. This is to simplify the use of :func:`str2pos`.
If you want to extract just one position
then give the index in the output array, for example ``W0 = w[0]``. The x and y coordinates
are in this case: ``wx = W0[0]; wy = W0[1]``.


**Structure of output**

The function :func:`str2pos` returns a tuple with four items:

      * *w*: an array with positions in world coordinates. One position has
        *n* coordinates and *n* is the dimension of your data structure
        which 1 for structure with one axis, 2 for a map, 3 for a cube etc.
      * *p*: an array with positions in pixel coordinates. It has the same structure
        as *w*.
      * *u*: an array with the units of the world coordinates
        These units are derived from the projection object with
        an optional alternative sky system and/or an optional
        spectral translation. The number of units in the list is the
        number of coordinates in a position.
      * *e*: an error message. If the length of this string is not 0, then
        it represents an error message and the arrays *w* and *p* are empty.



Position syntax
---------------

Number of coordinates
.......................

A position has the same number of coordinates as the number of axes that are
defined by the Projection object. So each position in a 2-dim map has two coordinates.
One can enter 1 position or a sequence of positions as in:

>>> pos="0,1  4,5  2,3"

Numbers are separated either by a space or a comma.

So also:

>>> pos="0 1 4 5 2 3"
>>> pos="0,1,4,5,2,3"

give the same result.


Numbers in expressions
.......................

Numbers can be given as valid (Python) expressions.
A selection of functions and operators known to module NumPy can be used.
The functions are:

  * abs, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, 
    arctanh, cos, cosh, degrees, exp, log2, log10, mean, median, min, max, 
    pi, radians, sin, sinc, sqrt, sum, tan, tanh, 
    rand, randn, ranf, randint, sign
  * Aliases: acos = arccos, acosh = arccosh, asin = arcsin, 
    asinh = arcsinh, atan = arctan, atan2 = arctan2, atanh = arctanh, 
    ln = log10(x)/log10(e), log=log10, deg=degrees, rad=radians
  * arange, linspace

The functions allow a NumPy array as argument. Here its definition starts and
ends with a square bracket. Its elements are separated by a comma.
But note, it is not a Python list.
In addition to the selection of mathematical functions we also include
the functions :func:`arange` and :func:`linspace` from NumPy to
be able to generate arrays.

Examples:

  * ``arange(4)`` -> [0, 1, 2, 3]
  * ``max(arange(4))`` -> 3
  * ``linspace(1,2,5)`` -> [1., 1.25,  1.5,  1.75,  2.]
  * ``randint(0,10,3)`` -> [6, 4, 3]
  * ``sin(ranf(4))`` -> [0.66019925,  0.24063844,  0.28068498,  0.23582177]
  * ``median([-1,3,5,-2,5,1])`` -> 2.0
  * ``mean(arange(4))`` -> 1.5
  * ``log(10**[1,2,3])`` -> [1, 2, 3]
  * ``log(100) log10(100)`` -> [2, 2]
  * ``log2(e), ln(e)`` -> [1.44269504,  1.]
  * ``log2(2**[1,2,3,4])`` -> [1, 2, 3, 4]


Note the difference between the result of ``[pi]*3`` when ``[pi]`` is a
Python list (then a new list is created with elements [pi,pi,pi]), and
the array ``[pi]``. 
The array in our context is multiplied (element-wise) by 3.
This is also true for other operators.
So it is also valid to write:

   * ``[1,2,3,4]`` -> [1, 2, 3, 4]
   * ``pi*[1,2,3]`` -> [3.14159265,  6.28318531,  9.42477796]
   * ``[1,2,3]**2`` -> [1.,  4.,  9.]
   * ``[1,2,3]-100`` -> [-99., -98., -97.]
   * ``[1,2,3]/0.3`` -> [ 3.33333333,   6.66666667,  10.]

The array syntax also allows for the generation of ranges.
A range follows the
syntax ``start:end:step`` and *start* may be smaller than *end*. Here we deviate
also from Python. That is, we include always the values *start* and *end* in
the result:
Some examples:

   * ``[1:4]`` -> [ 1.,  2.,  3.,  4.]
   * ``[-1:-5]`` -> [-1., -2., -3., -4., -5.]
   * ``[-1:-5:-2]`` -> [-1., -3., -5.]
   * ``[5:1:1]`` -> []             # Note that increment is positive
   * ``[1:3, 10:12, 100]`` -> [1.,    2.,    3.,   10.,   11.,   12.,  100.]
   * ``[1*pi:2*pi]`` -> [3.14159265,  4.14159265,  5.14159265,  6.14159265,  7.14159265]

If one prefers the *non-inclusive* Python style ranges, then function :func:`arange` is
available. Another function is :func:`linspace` which generates a (given) number of
equidistant samples between a start and end value.

   * :func:`arange()`. For example ``arange(1,4)**3`` results in an
     array with elements 1, 2, 3 and all these elements are taken to the power of 3
   * :func:`linspace`. The arguments for 'linspace' are a start value,
     an end value and and the number of samples. For example ``linspace(1,3,4)`` results in an
     array with elements 1, 1.66666667, 2.33333333, 3

A range with a number of identical elements is created using a syntax with two
subsequent colons:

   * ``[1::3]`` -> [1, 1, 1]
   * ``[1**2::2, pi::2]`` -> [1, 1, 3.14159265, 3.14159265]

.. note::

   * Functions can have scalars, lists and arrays as arguments.
   * Mathematical expressions can be applied on all array elements at the same time.
   * Note that x to the power of y is written as x**y and not as
     x^y (which is a *bitwise or*).


To get information about NumPy functions you have to read the Python documentation
(e.g. on the command line in a terminal, type: ``ipython``. On the ipython command line
type: ``import numpy; help(numpy.linspace)``).
Here are some examples how to use ranges in the input of positions:

>>> pos = "degrees(pi) e"                 # pixel coordinates: 180, 2.71828183
>>> pos = "degrees(atan2(1,1)) abs(-10)"  # pixel coordinates: 45, 10.
>>> pos = "[pi::3]**2, [1:3]**3"
>>> pos = "[1,6/3,3,4]**3, pi*[1,2,3,4]"
>>> pos = "[1:10], [10,1]"
>>> pos = "[sin(pi):-10:-2]  range(6)"
>>> pos = "linspace(0,3,200), tan(radians(linspace(0,3,200)))"



Grouping of numbers
....................

Coordinates can also be **grouped**. Elements in a group are processed in one pass
and they represent only one coordinate in a position.
A group of numbers can be prepended by a sky definition or spectral translation
or be appended by a unit.
Then the unit applies to all the elements in the group. We will see examples of this
in one of the next sections.
For the first example we could have grouped the coordinates as follows:

>>> pos="'0,4,2' '1,5,3'"

or, using the more powerful array generator, as:

>>> pos="[0,4,2] [1,5,3]"

Coordinates enclosed by single quotes or square brackets are parsed
by Python's expression evaluator *eval()*  as one expression.
The elements in a group can also be expressions.
If square brackets are part of the expression, the expression represents
a Python list and not an array! Examine the next expressions:

>>> pos = "'[pi]+[1,2]' range(3)"   # [pi, 1, 2]  [0, 1, 2]
>>> pos = "'[pi]*3' range(3)"       # [pi, pi, pi]  [0, 1, 2]
>>> pos = "'[sin(x) for x in range(4)]' range(4)"

In this context the square brackets define a list. In the examples we demonstrate
the list operator '+' which concatenates lists, '*' which repeats the elements in a list
and list comprehension.
Note that Python's :func:`eval()` function requires that the elements in an expression
are separated by a comma.

It is important to remember that without quotes, the square brackets define an array.
The list operators '+' and '*' have a different meaning for lists and arrays.
For arrays they add or multiply element-wise as shown in:

>>> pos = "[0,4,2]+10 [1,5,3]*2"  # is equivalent with "[10,14,12]  [2,10,6]"

Other examples of grouping are listed in the section about reading data from
disk with :func:`readcol()` and in the section about the :func:`eval()` function.


Pixel coordinates
.................

All numbers, in a string representing a position, which are not recognized
as world coordinates are returned as pixel coordinates.
The first pixel on an axis has coordinate 1. Header value *CRPIX* sets the
position of the reference pixel. If this is an integer number, the reference is
located at the center of a pixel. This reference sets the location of of the
world coordinate given in the (FITS) header in keyword *CRVAL*. 

For the examples below you should use function :func:`str2pos` to test the conversions.
However, for this function you need a (FITS) header. In the description at :func:`str2pos`
you will find an example of its use.

Examples of two pixel coordinates in a two dimensional world coordinate system (wcs):
   
>>> pos = "10 20"       # Pixel position 10, 20
>>> pos = "10 20 10 30" # Two pixel positions
>>> pos = "(3*4)-5 1/5*(7-2)"
>>> pos = "abs(-10), sqrt(3)"
>>> pos = "sin(radians(30)), degrees(asin(0.5))"
>>> pos = "cos(radians(60)), degrees(acos(0.5))"
>>> pos = "pi, tan(radians(45))-0.5, 3*4,0"        # 2 positions
>>> pos = "atan2(2,3), 192"
>>> pos = "[pi::3], [e**2::3]*3"   # [pi, pi, pi], [3*e**2, 3*e**2, 3*e**2]


Special pixel coordinates
..........................

For the reference position in a map we can use symbol 'PC' (Projection center).
The center of your data structure is set with symbol 'AC'.
You can use either one symbol or the same number of symbols as there are
axes in your data structure.

>>> pos = "pc"     # Pixel coordinates of the reference pixel
>>> pos = "PC pc"  # Same as previous. Note case insensitive parsing
>>> pos = "AC"     # Center of the map in pixel coordinates



Constants
..........

A number of global constants are defined and these can be used in the
expressions for positions. The constants are case sensitive.
These constants are::

      c_ = 299792458.0             # Speed of light in m/s
      h_ = 6.62606896e-34          # Planck constant in J.s
      k_ = 1.3806504e-23           # Boltzmann in J.K^-1
      G_ = 6.67428e-11             # Gravitation in m^3. kg^-1.s^-2
      s_ = 5.6704e-8               # Stefan-Boltzmann in J.s^-1.m^-2.K^-4
      M_ = 1.9891e+30              # Mass of Sun in kg
      P_ = 3.08567758066631e+16    # Parsec in m



World coordinates
..................

World coordinates can be distinguished from pixel coordinates. A world
coordinate is:

   * a coordinate followed by a (compatible) unit. Note that the
     units of the world coordinate are given in the (FITS) header in keyword *CUNIT*.
     If there is no CUNIT in the header or it is an empty string or you
     don't remember the units, then use either:

       * The wildcard symbol '?'
       * A case insensitive minimal match for the string 'UNITS'
       
   * a coordinate prepended by a definition for a sky system or a spectral system.
   * a coordinate entered in sexagesimal notation. (hms/dms)

.. note::

   One can mix pixel- and world coordinates in a position.

Units
,,,,,,,


For a two dimensional data structure (e.g. an optical image of part of the sky)
we can enter a position in world coordinates as:

>>> pos = 178.7792 deg  53.655 deg

But we can also use compatible units:

>>> pos = "178.7792*60 arcmin  53.655 deg"    # Use of a compatible unit if CUNIT is "DEGREE"
>>> pos = "10 1.41541820e+09 Hz"              # Mix of pixel coordinate and world coordinate
>>> pos = "10 1.41541820 GHz"                 # Same position as previous using a compatible unit

Units are minimal matched against a list with known units. The parsing of units
is case insensitive. The list with known units is:

   * angles: 'DEGREE','ARCMIN', 'ARCSEC', 'MAS', 'RADIAN'
     'CIRCLE', 'DMSSEC', 'DMSMIN', 'DMSDEG', 'HMSSEC', 'HMSMIN', 'HMSHOUR'
   * distances: 'METER', 'ANGSTROM', 'NM', 'MICRON', 'MM', 'CM',
     'INCH', 'FOOT', 'YARD', 'M', 'KM', 'MILE', 'PC', 'KPC', 'MPC', 'AU', 'LYR'
   * time: 'TICK', 'SECOND', 'MINUTE', 'HOUR', 'DAY', 'YR'
   * frequency: 'HZ', 'KHZ','MHZ', 'GHZ'
   * velocity: 'M/S', 'MM/S', 'CM/S', 'KM/S'
   * temperature: 'K', 'MK'
   * flux (radio astr.): 'W/M2/HZ', 'JY', 'MJY'
   * energy: 'J', 'EV', 'ERG', 'RY'

It is also possible to convert between inverse units like the wave number's 1/m
which, for  example, can be converted to 1/cm.

For a unit, one can also substitute the wildcard symbol '?'. This is the same as
setting the units from the header (conversion factor is 1.0). The symbol is
handy to set coordinates to world coordinates. But it is essential if there are
no units in the header like the unitless STOKES axis. One can also use the string
*units* which has the same role as '?'.

>>> pos = "[0, 3, 4] ?"
>>> pos = "7 units"
>>> pos = "[5, 6.3] U"



Sky definitions
,,,,,,,,,,,,,,,,,

The detailed information about sky definitions can be found in:

   * :ref:`celestial-skysystems`
   * :ref:`celestial-refsystems`
   * :ref:`celestial-epochs`


If a coordinate is associated with a sky definition it is parsed as a world coordinate.
A sky definition is either a case insensitive minimal match from the list::

  'EQUATORIAL','ECLIPTIC','GALACTIC','SUPERGALACTIC'

or it is a definition between curly brackets which can contain one or
more items from the following list:
*sky system, reference system, equinox* and *epoch of observation*.

An empty string between curly brackets e.g. {}, followed by a number,
implies a world coordinate in the native sky system. 

Examples:

>>> pos = "{eq} 178.7792  {} 53.655"
    # As a sky definition between curly brackets
>>> pos = "{} 178.7792 {} 53.655"
    # A world coordinate in the native sky system
>>> pos = "{eq,B1950,fk4} 178.12830409  {} 53.93322241"
    # With sky system, reference system and equinox
>>> pos = "{fk4} 178.12830409  {} 53.93322241"
    # With reference system only.
>>> pos = "{eq, B1950,fk4, J1983.5} 178.1283  {} 53.933"
    # With epoch of observation (FK4 only)
>>> pos = "{eq B1950 fk4 J1983.5} 178.1283  {} 53.933"
    # With space as separator
>>> pos = "ga 140.52382927 ga 61.50745891"
    # Galactic coordinates
>>> pos = "ga 140.52382927 {} 61.50745891"
    # Second definition copies from first
>>> pos = "su 61.4767412, su 4.0520188"
    # Supergalactic
>>> pos = "ec 150.73844942 ec 47.22071243"
    # Ecliptic
>>> pos = "{} 178.7792 6.0"
    # Mix world- and pixel coordinate
>>> pos = "5.0, {} 53.655"
    # Mix with world coordinate in native system

.. note::

   * Mixing sky definitions for one position is not allowed i.e. one cannot
     enter *pos = "ga 140.52382927 eq 53.655"*
   * If you mix a pixel- and a world coordinate in a spatial system
     then this world coordinate must be defined in the native system, i.e. *{}*
     

We can also specify positions in data structures with only one spatial axis
and a non-spatial axis (e.g. position velocity diagrams). The conversion function
:func:`str2pos` needs a pixel coordinate for the missing spatial axis.
If one of the axes is a spectral axis, then one can enter world coordinates
in a compatible spectral system:

>>> pos = "{} 53.655 1.415418199417E+09 hz"
    # Spatial and spectral world coordinate
>>> pos = "{} 53.655 1.415418199417E+03 Mhz"
    # Change Hz to MHz
>>> pos = "53.655 deg 1.415418199417 Ghz"
    # to GHz
>>> pos = "{} 53.655 vopt 1.05000000e+06"
    # Use spectral translation to enter optical velocity
>>> pos = "{} 53.655 , vopt 1050 km/s"
    # Change units
>>> pos = "10.0 , vopt 1050000 m/s"
    # Combine with a pixel position
>>> pos = "{} 53.655 vrad 1.05000000e+06"
    # Radio velocity
>>> pos = "{} 53.655 vrad 1.05000000e+03 km/s"
    # Radio velocity with different unit
>>> pos = "{} 53.655 FREQ 1.41541820e+09"
    # A Frequency
>>> pos = "{} 53.655 wave 21.2 cm"
    # A wave length with alternative unit
>>> pos = "{} 53.655 vopt c_/285.51662
    # Use speed of light constant to get number in m/s


.. note::

   For positions in a data structure with one spatial axis, the other
   (missing) spatial axis is identified by a pixel coordinate. Usually it's
   a slice).
   This restricts the spatial world coordinates to their native wcs.
   We define a world coordinate in its native sky system
   with *{}* 

.. note::

   A sky definition needs not to be repeated. Only one definition is allowed
   in a position. The second definition therefore can be empty as in *{}*. 

.. note::

   World coordinates followed by a unit, are supposed to be compatible
   with the Projection object. So if you have a header with spectral type FREQ but
   with a spectral translation set to VOPT, then ``"{} 53.655 1.415418199417E+09 hz"``
   is invalid, ``"10.0 , vopt 1050000 m/s"`` is ok and
   also ``"{} 53.655 FREQ 1.415418199417e+09"`` is correct.

Sexagesimal notation
,,,,,,,,,,,,,,,,,,,,,,,

Read the documentation at :func:`parsehmsdms` for the details.
Here are some examples:

>>> pos = "11h55m07.008s 53d39m18.0s"
>>> pos = "{B1983.5} 11h55m07.008s {} -53d39m18.0s"
>>> pos = -33d 0d


Reading from file with function *readcol()*, *readhms()* and *readdms()*
..........................................................................

Often one wants to plot markers at positions that are stored in a text 
file (Ascii) on disk.

In practice one can encounter many formats for coordinates in text files.
Usually these coordinates are written in columns. For example one can expect
longitudes in degrees in the first column and latitudes in degrees in the second.
But what do these coordinates represent? Are they galactic or ecliptic positions?
If your current plot represents an equatorial system can we still plot the markers
from the file if these are given in the galactic sky system? And there are more
questions:

  * Assume you have a file with three columns with hours, minutes and seconds as longitude
    and three columns with degrees, minutes and seconds as latitude. Is it possible
    to read these columns and combine them into longitudes and latitudes?
    Assume you have a file and the Right Ascensions are given in decimal hours,
    is it possible to convert those to degrees?
  * Assume your file has numbers that are in a unit that is not the same unit
    as the axis unit in your plot. Is it possible to change the units of the
    data of the column in the text file?
  * Assume you have several (hundreds of) thousands marker positions.
    Is reading the marker positions fast?
  * If a file has comment lines that start with another symbol than '!' or '#',
    can one still skip the comment lines?
  * If a file has columns separated by something else than whitespace,
    is it still possible then to read a column?

All these questions can be answered with *yes* if you use this module.
We provided three functions: :func:`readcol()`, :func:`readhms()` and :func:`readdms()`.
These functions are based on module :mod:`tabarray`. The routines in this
module are written in C and as a result of that, reading data from file is very fast.
The arguments of these functions are derived from those in
:func:`tabarray.readColumns` with the exception that
argument *cols=* is replaced by *col=* for function *readcol()* because
we want to read only one column per coordinate to keep the syntax
easy and flexible.
In the functions :func:`readhms()` and :func:`readdms()`, which are
also based on :func:`tabarray.readColumns`, the *cols=* argument is replaced by
arguments *col1=, col2=, col3=*. These functions read three columns at once and
combine the columns into one.
Tabarray routines count with 0 as the first column, first row etc. The routines
that we describe here count with 1 as the first column or row etc.

**syntax**

>>> readcol(filename, col=1, fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=999.999, fromrow=None, torow=None, rowstep=None)


>>> readhms(filename, col1=1, col2=2, col3=3,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=999.999,
            fromrow=None, torow=None, rowstep=None)

Function :func:`readdms()` has the same syntax as :func:`readhms()`


The parameters are:

    * filename - a string with the name of a text file containing the table.
      The string must be entered with double quotes. Single quotes
      have a different function in this parser (grouping).
    * col - a scalar that sets the column number.
    * fromline - Start line to be read from file (first is 1).
    * toline - Last line to be read from file. If not specified, the end of the file is assumed.
    * comment - a string with characters which are used to designate comments in the input file. The occurrence of any of these characters on a line causes the rest of the line to be ignored. Empty lines and lines containing only a comment are also ignored.
    * sepchar - a string containing the column separation characters to be used. Columns are separated by any combination of these characters.
    * rows - a tuple or list containing the row numbers to be extracted.
    * bad - a number to be substituted for any field which cannot be decoded as a number.
      The default value is 999.999
    * fromrow - number of row from the set of lines with real data to start reading
    * torow - number of row from the set of lines with real data to end reading. The *torow* line
      is included.
    * rowstep - Step size in rows. Works also if no values are given for *fromrow* and *torow*.

There is a difference between the *rows=* and the *fromline=* , *toline=*
keywords. The first reads the specified rows from the *parsed* contents
of the file( (*parsed* contents are lines that are not comment lines), while the line keywords specify which lines you want to read from file.
Usually comment characters '#' and '!' are used. If you expect another comment
character then change this keyword.
Keyword *sepchar=* sets the separation character. The default is a comma,
a space and a tab. *bad=* is the value
that is substituted for values that could not be parsed so that they can be
easily identified.

.. note::
      
      * Numbering of columns start with 1
      * Numbering of rows start with 1
      * Numbering of lines start with 1
      * The result is an array so it can be used in an expression
   
Some examples:

Assume a text file on disk with a number of rows with 2 dimensional marker positions
in pixel coordinates. The text file is called *pixmarks.txt*.
Then the simplest line to read this data is:

>>> pos = 'readcol("pixmarks.txt") readcol("pixmarks.txt",2)'
>>> annim.Marker(pos=pos, marker='o', markersize=10, color='r')

All parameters have defaults except the filename parameter.
The default column is 1, i.e. the first column.
For readability we prefer to write the positions as:

>>> pos = 'readcol("pixmarks.txt", col=1) readcol("pixmarks.txt",col=2)'

If you want all the data up to line 30 (and line 30 including) you should write:

>>> pos = 'readcol("pixmarks.txt", col=1, toline=30) readcol("pixmarks.txt",col=2, toline=30)'

If your file has relevant data from line 30 to the end of the file, one should write:

>>> pos = 'readcol("pixmarks.txt", col=1, fromline=30) readcol("pixmarks.txt",col=2, fromline=30)'

As stated earlier, we distinguish *lines* and *rows* in a file.
Lines are also those which are empty or which start with a comment.
Rows are only those lines with data. So if you want to read only the first
5 rows of data, then use:

>>> pos = 'readcol("pixmarks.txt", col=1, torow=5) readcol("pixmarks.txt",col=2, torow=5)'

Note that the parameters *toline* and *torow* include the given value. You can specify
a range of rows including a step size with:

>>> pos = 'readcol("pixmarks.txt", col=1, fromrow=10, torow=44, rowstep=2), .....'

to get row number 10, 12, ..., 44. Note that it is not possible to set a
step size if you use the *fromline* or *toline* parameter.

In some special circumstances you want to be able to read only
preselected rows from the data lines. Assume a user needs rows 1,3,7,12,44.
Then the position string should be:

>>> pos = 'readcol("pixmarks.txt", col=1, rows=[1,3,7,12,44]), .....'

Perhaps you wonder why you need to repeat the :func:`readcol` function for
each coordinate. It is easier to use it once and specify two columns instead
of one. We did not implement this feature because usually one will read world coordinates
from file and often we want to add units or a sky- or spectral conversion.
Then you must be able to read the data for each column separately. 
Assume we have a file on disk called 'lasfootprint' which stores two sets
of 2 dimensional positions (i.e. 4 coordinates) separated by an empty line.

::

   #  RA J2000  Dec      l         b         eta     lambda
      8.330    -1.874   225.624    19.107   -36.250   300.000
      8.663    -2.150   228.598    23.268   -36.250   305.000
      8.996    -2.409   231.763    27.369   -36.250   310.000
      9.329    -2.651   235.170    31.394   -36.250   315.000
      9.663    -2.872   238.878    35.320   -36.250   320.000
      .....     ......
      .....


It has a blank line at line 63. The first column represents Right Ascensions in
decimal hours.
If we want to read the positions given by column 1 and 2 of the second
segment (starting with line 66)
and column 1 is given in decimal hours, then you need the command:
   
>>> pos=  'readcol("lasfootprint", col=1,fromline=64)
                   HMShour readcol("lasfootprint", col=2,fromline=64) deg'

The first coordinate is followed by a unit, so it is a world coordinate.
We have a special unit that converts from decimal hours to degrees (*HMSHOUR*).
The last coordinate is followed by a unit (deg) so it is a world coordinate.
It was also possible to prepend the second coordinate with {} and omit the unit as in:
Between the brackets there is nothing specified. This means that we assume
the coordinates in the file (J2000) match the sky system of the world
coordinate system of your map.

>>> pos=  'readcol("lasfootprint", 1,64) HMShour {} readcol("lasfootprint", 2,64)'

Note that the third parameter is the *fromline* parameter.
If columns 3 and 4 in the file are galactic longitudes and latitudes, but
our basemap is equatorial, then we could have read the positions
with an alternative sky system as in (now we read the first data segment):

>>> pos=  '{ga} readcol("lasfootprint", 3, toline=63)  {} readcol("lasfootprint", 4, toline=63)'

The second sky definition is empty which implies a copy of the first
definition (i.e. {ga}).

.. note::

   The sky definition must describe the world coordinate system of the
   data on disk. It will be automatically converted to a position in
   the sky system of the Projection object which is associated with
   your map or axis.

Some files have separate columns for hour, degrees, minutes and seconds.
Assume you have an ASCII file on disk with 6 columns representing
sexagesimal coordinates. For example:

::

   ! Test file for Ascii data and the READHMS/READDMS command
   11 57 .008 53 39 18.0
   11 58 .008 53 39 17.0
   11 59 .008 53 39 16.0
   ....

Assume that this file is called *hmsdms.txt* and it contains equatorial
coordinates in *'hours minutes seconds degrees minutes seconds'* format,
then read this data with:

>>> pos= '{} readhms("hmsdms.txt",1,2,3) {} readdms("hmsdms.txt",4,5,6)'

Or with explicit choice of which lines to read:

>>> pos= '{} readhms("hmsdms.txt",1,2,3,toline=63) {} readdms("hmsdms.txt",4,5,6,toline=63)'

The data is automatically converted to degrees.
What if the format is **'d m s d m s'** and the coordinates are galactic.
Then we should enter;
   
>>> pos= 'ga readdms("hmsdms.txt",1,2,3) ga readdms("hmsdms.txt",4,5,6)'

if your current sky system is galactic then it also possible to enter:

>>> pos= 'readdms("hmsdms.txt",1,2,3) deg  readdms("hmsdms.txt",4,5,6) deg'

If the columns are not in the required order use the keyword names:

>>> pos= 'readdms("hmsdms.txt",col3=0,col2=1,col3=2) deg  readdms("hmsdms.txt",4,5,6) deg'

The result of one of the functions described in this section is an array and therefore
suitable to use in combination with functions and operators:

>>> pos='1.1*readhms("hmsdms.txt",1,2,3)-5 sin(readdms("hmsdms.txt",4,5,6)-10.1)'


Reading header items with function *header()*
..............................................

Command *header* reads an item from the header that was used to create the Projection
object. The header item must represent a number.

>>> pos= 'header("crpix1") header("crpix2")'

.. note::

   * Header keys are case insensitive
   * A key must be given with double quotes

Parser errors messages
.......................

The position parser is flexible but there are some rules. If the input
cannot be transformed into coordinates then an appropriate message will be
returned. In some cases the error message seems not to be related to the problem
but that seems often the case with parsers. If a number is wrong, the parser tries
to parse it as a sky system or a unit. If it fails, it will complain about
the sky system or the unit and not about the number.


Testing the parser
...................

You can run the module's 'main' (i.e. execute the module) to test pre installed
expressions and to experiment with your own positions entered at a prompt.
Please copy the module *positions.py* to your working directory first!
The program will display
a couple of examples before it prompts for user input. Then your are prompted
to enter a string (no need to enclose it with quotes because it is read as a string).
Enter positions for a two dimensional data structure with axes R.A. and Dec.
Start the test with:

>>> python positions.py

GIPSY's grids mode
......................

FITS pixel coordinates start with number one and the last pixel
for axis n is the value of header item *NAXISn*. Pixel value
*CRPIXn* is the pixel that corresponds to *CRVALn*. The value
of *CRPIXn* can be non-integer.
There are also systems that implement a different numbering.
For example the Groningen Image Processing SYstem (GIPSY) uses an offset.
There we call pixel *CRPIXn* grid 0, so
grid 0 corresponds to *CRVALn*. It has the advantage that these grid coordinates
are still valid after cropping  the input data. For FITS data we need to change
the value for *CRPIXn* after slicing the data and writing it to a new FITS file.
But then your original pixel coordinates for the same positions need to be shifted too.
The Projection object can be set into GIPSY's grid mode using attribute
:attr:`gridmode` (True or False).


Functions
---------

.. autofunction:: str2pos
.. autofunction:: parsehmsdms
.. autofunction:: unitfactor

"""

# Imports

from re import split as re_split
from re import findall as re_findall
from string import whitespace, ascii_uppercase
import six
from numpy import nan as unknown
from numpy import asarray, zeros, floor, array2string
from numpy import  array, ndarray
from kapteyn import wcs                          # The WCSLIB binding
from kapteyn.celestial import skyparser
# Next functions are imported for eval()
from kapteyn.tabarray import readColumns
from numpy import arange, linspace
from numpy import abs, arccos, arccosh, arcsin, arcsinh, arctan, arctan2
from numpy import arctanh, cos, cosh, degrees, exp, log2, log10, mean, median, min, max
from numpy import pi, radians, sin, sinc, sqrt, sum, tan, tanh, sign
from numpy.random import rand, randn, ranf, randint

# py2/3 comp:
#from operator import isSequenceType
def isSequenceType(obj):
    try:
        from collections import Sequence
    except ImportError:
        from operator import isSequenceType
        return operator.isSequenceType(obj)
    else:
        return isinstance(obj, Sequence)

# Euler's number
e = 2.7182818284590452353602874713527

# Function aliases
acos = arccos
acosh = arccosh
asin = arcsin
asinh = arcsinh
atan = arctan
atan2 = arctan2
atanh = arctanh
log = log10
deg = degrees
rad = radians


badval = 99999.999  # Set bad number in tabarray routines to this value
sepchar=', \t'      # Default separation characters for readcol function


def ln(x):
   return log10(x)/log10(e)


class __a(object):
#-------------------------------------------------------
# Create array objects with square bracket syntax
# Allow for lists in a list.
#-------------------------------------------------------
   def __init__(self,inclusive=False):
      self.incl = int(inclusive)
   def __getitem__(self, key):
      if not isSequenceType(key):
         key = (key,)
      result = []
      for element in key:
         if isinstance(element, slice):
            startval = float(element.start)
            if element.stop is None:                      # v::n
               for value in [element.start]*element.step:
                  result.append(value)
            else:
               endval   = float(element.stop)
               if element.step is not None:               # va:vb:incr
                  incr = float(element.step)
               elif startval>endval:                      # va:vb
                  incr = -1.0
               else:
                  incr = +1.0
               endval = endval+0.5*self.incl*incr
               for value in arange(startval, endval, incr):
                  result.append(value)
         elif isSequenceType(element):
            for value in element:
               result.append(float(value))
         else:
            result.append(float(element))
      return array(result)
#ar  = __a(inclusive=False)  # We don't need this at the moment
a = __a(inclusive=True)

# Define constants for use in eval()
c_ = 299792458.0    # Speed of light in m/s
h_ = 6.62606896e-34 # Planck constant in J.s
k_ = 1.3806504e-23  # Boltzmann in J.K^-1
G_ = 6.67428e-11    # Gravitation in m^3. kg^-1.s^-2
s_ = 5.6704e-8      # Stefan- Boltzmann in J.s^-1.m^-2.K^-4
M_ = 1.9891e+30     # Mass of Sun in kg
P_ = 3.08567758066631e+16 # Parsec in m


def issequence(obj):
  return isinstance(obj, (list, tuple, ndarray))


def usermessage(token, errmes):
   return "Error in '%s': %s" % (token, errmes)


def readcol(filename, col=1, fromline=None, toline=None, rows=None, comment="!#",
            sepchar=sepchar, bad=badval, fromrow=None, torow=None, rowstep=None):
#-------------------------------------------------------------
   """
   Utility to prepare a call to tabarray's readColumns() function
   We created a default for the 'comment' argument and changed the
   column argument to accept only one column.
   """
#-------------------------------------------------------------
   if issequence(col):
      column = col[0]
   else:
      column = col
   column = [column-1]  # First column is 1 but for readColumns it must be 0
   if rows != None:
      if not issequence(rows):
         rows = [rows]
      rows = [i-1 for i in rows] 
   lines = None
   if not fromline is None or not toline is None:
      if fromline is None:
         fromline = 0
      if toline is None:
         toline = 0
      lines = (fromline, toline)
   rowslice = (None, )
   if not fromrow is None or not torow is None or not rowstep is None:
      if not fromrow is None:
         fromrow -= 1
      rowslice = (fromrow, torow, rowstep)
   colslice = (None, )
   c = readColumns(filename=filename, comment=comment, cols=column, sepchar=sepchar,
               rows=rows, lines=lines, bad=bad, rowslice=rowslice, colslice=colslice)
   return c.flatten()


def readhmsdms(filename, col1=1, col2=2, col3=3,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=sepchar, bad=badval, fromrow=None, torow=None, rowstep=None, mode='hms'):
#-------------------------------------------------------------
   """
   Helper function for readhms() and readdms()
   """   
#-------------------------------------------------------------
   column = [col1-1, col2-1, col3-1]    # Make it zero based for readColumns()
   if rows != None:
      if not issequence(rows):
         rows = [rows]
      rows = [i-1 for i in rows]
   lines = None
   if not fromline is None or not toline is None:
      if fromline is None:
         fromline = 0
      if toline is None:
         toline = 0
      lines = (fromline, toline)
   rowslice = (None, )
   if not fromrow is None or not torow is None or not rowstep is None:
      if not fromrow is None:
         fromrow -= 1
      rowslice = (fromrow, torow, rowstep)
   colslice = (None, )
   c = readColumns(filename=filename, comment=comment, cols=column, sepchar=sepchar,
               rows=rows, lines=lines, bad=bad, rowslice=rowslice, colslice=colslice)
   if mode == 'hms':
      h = c[0]; m = c[1]; s = c[2]
      vals = (h+m/60.0+s/3600.0)*15.0
   else:
      d = c[0]; m = c[1]; s = c[2]
      # Take care of negative declinations
      vals = sign(d)*(abs(d)+abs(m)/60.0+abs(s)/3600.0)
   return asarray(vals)


def readhms(filename, col1=1, col2=2, col3=3,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=sepchar, bad=badval, fromrow=None, torow=None, rowstep=None):
#-------------------------------------------------------------
   """
   Utility to prepare a call to tabarray's readColumns() function
   We created a default for the 'comment' argument and changed the
   column argument to accept only one column.
   """
#-------------------------------------------------------------
   return readhmsdms(filename=filename, col1=col1, col2=col2, col3=col3,
                     fromline=fromline, toline=toline, rows=rows, comment=comment,
                     sepchar=sepchar, bad=bad, fromrow=fromrow, torow=torow, rowstep=rowstep,
                     mode='hms')
   

def readdms(filename, col1=1, col2=2, col3=3,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=sepchar, bad=badval, fromrow=None, torow=None, rowstep=None):
#-------------------------------------------------------------
   """
   Utility to prepare a call to tabarray's readColumns() function
   We created a default for the 'comment' argument and changed the
   column argument to accept only one column.
   """
#-------------------------------------------------------------
   return readhmsdms(filename=filename, col1=col1, col2=col2, col3=col3,
                     fromline=fromline, toline=toline, rows=rows, comment=comment,
                     sepchar=sepchar, bad=bad, fromrow=fromrow, torow=torow, rowstep=rowstep,
                     mode='dms')


source = {}
def header(key):
#-------------------------------------------------------------
   """
   This function should be used as method of the Coordparser
   routine.
   However we need it here to be able to use it in the
   restricted version of eval(). It must read its items from
   a header, so we made this header global. It is set in the
   Coordparser method.
   """
#-------------------------------------------------------------
   return float(source[key.upper()])



# Restrict available functions etc. to eval()
eval_restrictlist = ['arange', 'linspace',
 'abs', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2',
 'arctanh', 'cos', 'cosh', 'degrees', 'exp', 'log2', 'log10',
 'mean', 'median', 'min', 'max',
 'pi', 'radians', 'sin', 'sinc', 'sqrt', 'sum', 'tan', 'tanh',
 'rand', 'randn', 'ranf', 'randint', 'acos', 'acosh', 'asin', 'asinh',
 'atan', 'atan2', 'atanh', 'e', 'a', 'ln', 'log', 'deg', 'rad', 'sign',
 'readcol', 'readhms', 'readdms', 'header', 
 'c_', 'h_', 'k_', 'G_', 's_', 'M_', 'P_']

# Filter the local namespace
eval_dict = dict([(k, locals().get(k, None)) for k in eval_restrictlist])

# We need some builtins
eval_dict['abs'] = abs
eval_dict['range'] = range


def eval_restrict(arg):
   return eval(arg, {"__builtins__":None}, eval_dict)


def minmatch(userstr, mylist, case=0):
#--------------------------------------------------------------
   """
   Purpose:    Given a list 'mylist' with strings and a search string
              'userstr', find a -minimal- match of this string in
               the list.

   Inputs:
    userstr-   The string that we want to find in the list of strings
     mylist-   A list of strings
       case-   Case insensitive search for case=0 else search is
               case sensitive.

   Returns:    1) None if nothing could be matched
               2) -1 if more than one elements match
               3) >= 0 the index of the matched list element
   """
#--------------------------------------------------------------
   indx = None
   if case == 0:
      ustr = userstr.upper()
   else:
      ustr = userstr
   for j, tr in enumerate(mylist):
      if case == 0:
         liststr = tr.upper()
      else:
         liststr = tr
      if ustr == liststr:
         indx = j
         break
      i = liststr.find(ustr, 0, len(tr))
      if i == 0:
         if indx == None:
            indx = j
         else:
            indx = -1
   return indx


def unitfactor(unitfrom, unitto):
#-----------------------------------------------------------------------------------
   """
   Return the conversion factor between two units.

   :param unitfrom:
      Units to convert from. Strings with '1/unit' or '/unit' are
      also allowed. If this parameter is '?' then the incoming
      unit is a wildcard character and the conversion factor 1.0
      is returned. The same holds for a case insensitive minimum match
      of the string 'UNITS'. This option is necessary for the option
      to use world coordinates when there are no units given in the header
      of the data (i.e. there is no CUNITn keyword or its contents is empty).
   :type unitfrom: String
   :param unitto:
      Units to convert to. Strings with '1/unit' or '/unit' are
      also allowed.
   :type axtype: String

   :Returns:

      The conversion factor to convert a number in 'unitsfrom'
      to a number in 'unitsto'.

   :Notes:
   
   :Examples:

      >>> print unitfactor('1/m', '1/km')
      (1000.0, '')
      >>> print positions.unitfactor('1/mile', '1/km')
      (0.62137119223733395, '')
      >>> print positions.unitfactor('mile', 'km')
      (1.6093440000000001, '')

   """
#-----------------------------------------------------------------------------------
   errmes = ''
   # Process the wildcard options
   if unitfrom == '?':
      # Then the wildcard was used to set the unit
      return 1.0, errmes
   i = minmatch(unitfrom, ['UNITS'])
   if i != None and i >= 0:
      # Then user entered a string that sets the conversion factor to 1
      return 1.0, errmes

   units = {'DEGREE' :      (1,                        1.0),
            'ARCMIN' :      (1,                        1.0/60.0),
            'ARCSEC' :      (1,                        1.0/3600.0),
            'MAS' :         (1,                        1.0 / 3600000.0),
            'RADIAN' :      (1,                       57.2957795130823208767),
            'CIRCLE' :      (1,                      360.0),
            'DMSSEC' :      (1,                        0.0002777777777777778),
            'DMSMIN' :      (1,                        0.0166666666666666667),
            'DMSDEG' :      (1,                        1.0000000000000000000),
            'HMSSEC' :      (1,                       15.0*0.0002777777777777778),
            'HMSMIN' :      (1,                       15.0*0.0166666666666666667),
            'HMSHOUR':      (1,                       15.0000000000000000000),
            'METER' :       (2,                        1.0000000000000000000),
            'ANGSTROM' :    (2,                        0.0000000001000000000),
            'NM' :          (2,                        0.0000000010000000000),
            'MICRON' :      (2,                        0.0000010000000000000),
            'MM' :          (2,                        0.0010000000000000000),
            'CM' :          (2,                        0.0100000000000000000),
            'INCH' :        (2,                        0.0254000000000000000),
            'FOOT' :        (2,                        0.3048000000000000000),
            'YARD' :        (2,                        0.9144000000000000000),
            'M' :           (2,                        1.0000000000000000000),
            'KM' :          (2,                     1000.0000000000000000000),
            'MILE' :        (2,                     1609.3440000000000000000),
            'PC' :          (2,        30800000000000000.0000000000000000000),
            'KPC' :         (2,     30800000000000000000.0000000000000000000),
            'MPC' :         (2,  30800000000000000000000.0000000000000000000),
            'AU' :          (2,                        1.49598e11),
            'LYR' :         (2,                        9.460730e15),
            'TICK' :        (3,                        1.0000500000000000000),
            'SECOND' :      (3,                        1.0000000000000000000),
            'MINUTE' :      (3,                       60.0000000000000000000),
            'HOUR' :        (3,                     3600.0000000000000000000),
            'DAY' :         (3,                    86400.0000000000000000000),
            'YR' :          (3,                 31557600.0000000000000000000),
            'HZ' :          (4,                        1.0000000000000000000),
            'KHZ' :         (4,                     1000.0000000000000000000),
            'MHZ' :         (4,                  1000000.0000000000000000000),
            'GHZ' :         (4,               1000000000.0000000000000000000),
            'M/S' :         (5,                        1.0000000000000000000),
            'MM/S' :        (5,                        0.0010000000000000000),
            'CM/S' :        (5,                        0.0100000000000000000),
            'KM/S' :        (5,                     1000.0000000000000000000),
            'K' :           (6,                        1.0000000000000000000),
            'MK' :          (6,                        0.0010000000000000000),
            'W/M2/HZ':      (7,                        1.0),
            'JY' :          (7,                        1.0e-26                ), # Watts / m^2 / Hz
            'MJY' :         (7,                        1.0e-29                ),
            'TAU' :         (9,                        1.000000000000000000),
            'J' :           (10,                       1.0),
            'EV':           (10,                       1.60217733e-19),
            'ERG':          (10,                       1.0e-7),
            'RY' :          (10,                       2.179872e-18),
            'UNITS':        (11,                       1.0)
           }

   # There is a special case for units like 1/m or /m
   # Then the factor needs to be inverted.
   inverse = inversefrom = inverseto = False
   if unitfrom.startswith('/') or unitfrom.startswith('1/'):
      inversefrom = True
   if unitto.startswith('/') or unitto.startswith('1/'):
      inverseto = True
   if (inversefrom and not inverseto) or (inverseto and not inversefrom): 
      errmes = "[%s] cannot be converted to [%s]" % (unitfrom, unitto)
      return None, errmes
   inverse = inversefrom and inverseto
   if inverse:
      unitfrom = unitfrom.split('/')[1]
      unitto = unitto.split('/')[1]
      
   mylist = list(units.keys())
   i = minmatch(unitfrom, mylist)
   if i != None:
      if i >= 0:
         key = list(units.keys())[i]
         typ1 = units[key][0]
         fac1 = units[key][1]
      else:
         errmes = "Ambiguous unit [%s]" % unitto
         return None, errmes
   else:
      errmes = "[%s] should be a unit but is unknown!" % unitfrom
      return None, errmes
   i = minmatch(unitto, mylist)
   if i != None:
      if i >= 0:
         key = list(units.keys())[i]
         typ2 = units[key][0]
         fac2 = units[key][1]
      else:
         errmes = "Ambiguous unit [%s]" % unitto
         return None, errmes
   else:
      errmes = "[%s] should be a unit but is unknown!" % unitto
      return None, errmes
   
   if typ1 == typ2:
      unitfactor = fac1 / fac2
   else:
      errmes = "Cannot convert between [%s] and [%s]" % (unitfrom, unitto)
      return None, errmes

   if inverse:
      unitfactor = 1.0/unitfactor
   return unitfactor, errmes



def nint(x):
   """--------------------------------------------------------------
   Purpose:    Calculate a nearest integer compatible with then
               definition used in GIPSY's coordinate routines.

   Inputs:
         x-    A floating point number to be rounded to the nearest
               integer

   Returns:    The nearest integer for 'x'.

   Notes:      This definition adds a rule for half-integers. This
               rule is implemented with function floor() which implies
               that the left side of a pixel, in a sequence of
               horizontal pixels, belongs to the pixel while the
               right side belongs to the next pixel. This definition 
               of a nearest integer differs from the Fortran
               definition used in pre-April 2009 versions of GIPSY. 

   -----------------------------------------------------------------"""
   return floor(x+0.5)


def parseskysystem(skydef):
   #--------------------------------------------------------------------
   """
   Helper function for skyparser()
   """
   #--------------------------------------------------------------------
   try:
      sky = skyparser(skydef)
      return sky, ""
   except ValueError as message:
      errmes = str(message)
      return None, errmes


def parsehmsdms(hmsdms, axtyp=None):
   #--------------------------------------------------------------------
   """
   Given a string, this routine tries to parse its contents
   as if it was a spatial world coordinate either in
   hours/minutes/seconds format or degrees/minutes/seconds
   format.

   :param hmsdms:
      A string containing at least a number followed by
      the character 'h' or 'd' (case insensitive) followed by
      a number and character 'm'. This check must be performed
      in the calling environment.
      The number can be a negative value. The string cannot
      contain any white space.
   :type hmsdms: String
   :param axtype:
      Distinguish formatted coordinates for longitude and latitude.
   :type axtype: String

   :Returns:

      The parsed world coordinate in degrees and an empty error message
      **or**
      *None* and an error message that the parsing failed.

   :Notes:

      A distinction has been made between longitude axes and
      latitude axes. The hms format can only be used on longitude
      axes. However there is no check on the sky system (it should
      be equatorial).
      The input is flexible (see examples), even expressions are allowed.

   :Examples:

      >>> hmsdms = '20h34m52.2997s'
      >>> hmsdms = '60d9m13.996s'
      >>> hmsdms = '20h34m52.2997'     # Omit 's' for seconds
      >>> hmsdms = '60d9m13.996'
      >>> hmsdms = '20h34m60-7.7003'   # Expression NOT allowed
      >>> hmsdms = '-51.28208458d0m'   # Negative value for latitude

      * The 's' for seconds is optional
      * Expressions in numbers are not allowed because we cannot use Python's
        eval() function, because this function interprets expressions like '08'
        differently (octal).
      * dms format always allowed, hms only for longitude axes.
        Both minutes and seconds are optional. The numbers
        need not to be integer.
   """
   #-----------------------------------------------------------------------
   if ('h' in hmsdms or 'H' in hmsdms) and axtyp != None and axtyp != 'longitude':
      return None, "'H' not allowed for this axis"
   parts = re_split('([hdmsHDMS])', hmsdms.strip())  # All these characters can split the string
   number = 0.0
   total = 0.0
   sign = +1                                     # Keep track of the sign
   lastdelim = ' '
   prevnumber = True
   converthour2deg = False
   
   for p in parts:
      try:
         # Use float and not eval because eval cannot convert '08' like numbers
         number = float(p)                       # Everything that Python can parse in a number
         prevnumber = True
         adelimiter = False
      except:
         f = None
         if not p in whitespace:
            delim = p.upper()
            if delim == 'H':
               converthour2deg = True
               f = 3600.0
            elif delim == 'D':
               f = 3600.0
            elif delim == 'M':
               f = 60.0
            elif delim == 'S':
               f = 1.0
            else:
               return None, "Invalid syntax for hms/dms"

            # Use the fact that H/D M and S are in alphabetical order
            if prevnumber and delim > lastdelim and not (lastdelim == 'D' and delim == 'H'):
               if number < 0.0:
                  if delim in ['H', 'D']:        # Process negative numbers
                     number *= -1.0
                     sign = -1
                  else:
                     return None, "Invalid: No negative numbers allowed in m and s"
               if delim in ['M', 'S']:
                  if number >= 60.0:
                     return None, "Invalid: No number >= 60 allowed in m and s"
               total += number * f
               lastdelim = delim
            else:
               return None, "Invalid syntax for sexagesimal numbers"
            prevnumber = False
            adelimiter = True

   if prevnumber and not adelimiter:
      total += number                            # Leftover. Must be seconds because 's' is assumed if nothing follows
   if converthour2deg:
      total *= 15.0                              # From hours to degrees
   return [sign*total/3600.0], ''                # Return as a list because it will be transformed to a NumPy array



def mysplit(tstring):
   """--------------------------------------------------------------------
   Purpose:       This function splits a string into tokens. Whitespace
                  is a separator. Characters between parentheses or
                  curly brackets and quotes/double quotes are parsed 
                  unaltered.

   Inputs:
      tstring-    A string with expression tokens

   Returns:       A list with tokens

   Notes:         Parenthesis are used for functions e.g. atan().
                  Curly brackets are used to identify sky definitions.
                  Square brackets allow the use of lists e.g. [1,2,3,4].
                  Quotes group characters into one token.
                  The square bracket used within quotes is not parsed.
                  Without quotes, '[' is replaced by 'a[' which
                  uses the array generator from class __a.
   -----------------------------------------------------------------------"""
   pardepth = 0
   brackdepth = 0
   sqbdepth = 0
   quote = False
   tokens = ['']
   ws = whitespace + ','                         # Extend separators with comma

   for ch in tstring :
      if ch == '(':
         pardepth += 1
      elif ch == ')':
         pardepth -= 1
      elif ch == '{':
         brackdepth +=1
      elif ch == '}':
         brackdepth -=1
      elif ch == '[':
         sqbdepth += 1
      elif ch == ']':
         sqbdepth -= 1
      elif ch in ('"', "'") :
         quote = not quote
         if ch != '"':
            ch = '' # Copy quotes or not
      if ch in ws and not (sqbdepth or brackdepth or pardepth or quote):
         if tokens[-1] != '' : tokens.append('')
      else:
         if ch == '[' and not quote:         # Start syntax for array generator
            tokens[-1] += 'a'
         tokens[-1] += ch

   return tokens



class Coordparser(object):
   """--------------------------------------------------------------------
   Purpose: Parse a string which represents position(s). Return an object
   with the sorted input grids and world coordinates.
   
   First a pre parser finds the tokens in the string. Tokens are
   separated by a comma or whitespace.
   A group of characters enclosed by single or double quotes form one token.
   This enables a user to group numbers (with a sky system, a spectral
   translation and/or a unit)
   Characters enclosed by (), {} or [] are transferred unparsed. This allows
   a user to enter:
   1) parameters for functions, e.g. pos="atan2(x,y)"
   2) group parameters of a sky system, e.g. pos="{eq, J1983.5}"
   3) lists and arrays, e.g. POS="[0,1,2,3]"
   4) expressions for Python's eval() 'restricted' parser

   Strings between single quotes are parsed unaltered. Except the quotes themselves.
   They are removed.
   Strings between double quotes are parsed unaltered. This includes the double quotes.
   This is necessary to pass file names
   
   Description of the token parser:

   token END:    #
   token FILE    A file on disk
   token READCOL A file on disk
   token NUM     is a plain number
   token SNUM    is a sexagesimal number
   token UNIT    is a unit
   token WORLD   NUM followed by UNIT
   token SKY     One of EQ, EC, GA, SG or [SKY,parameters]
   token SPECTR  A compatible spectral translation
   
   goal:                positions END
   positions:           N (coordinates)*3 or datafromfile
   coordinate:          a grid or a world coordinate or sequence from file
   grid:                NUM: valid result of evalexpr() or result of Pythons eval() function
                        or result of READCOL
   unit:                UNIT: valid result of unitfactor
   world:               SNUM or NUM UNIT or sky or spectral
   sky:                 SKY world or SKY NUM
   spectral:            SPECTR world or SPECTR NUM
   ---------------------------------------------------------------------"""
   def __init__(self, tokenstr, ncoords, siunits, types, crpix,
                naxis, translations, source,
                gipsygrids=False):
      """--------------------------------------------------------------------
      Purpose:    Initialize the coordinate parser.
      
      Inputs:
        tokenstr- String with coordinate information, to be parsed by this
                  routine.
         ncoords- The number of axes in the data structure for which we want
                  to parse positions. One position is 'ncoords' coordinates
         siunits- A list with si units for each axis in the data structure
           types- A list with axis types (e.g. 'longitude', 'latitude'). With
                  this list the parser can decide whether a sky system or a
                  spectral translation could be applied.
           crpix- A list with reference pixels for each axis in the data
                  structure. This is needed to parse symbol 'PC'.
           naxis- A list with lengths of each axes. This is needed to parse
                  symbol 'AC'.
    translations- A list with all the spectral translations that are
                  possible for the selected data set.
      gipsygrids- A Boolean that sets the GIPSY flag for using the grid
                  system instead of pixel coordinates. Grid 0 corresponds to
                  the value of CRPIX in the header.


      Returns:    This constructor instantiates an object from class 'Coordparser'. The
                  most important attributes are:
      
          errmes- which contains an error message if the parsing was 
                  not successful.
       positions- zero, one or more positions (each position is 'ncoords'
                  numbers)
                  One position for ncoords=2 could be something like this:
                  [([308.71791541666664], 'w', '', ''), 
                                        ([60.153887777777783], 'w', '', '')]
                  It contains two tuples for the coordinates.
                  One coordinate is a tuple with:
                  1) A list with zero, one or more numbers
                  2) A character 'g' to indicate that these numbers are grids
                     or a character 'w' to indicate that these numbers are
                     world coordinates.
                  3) A number or a tuple that sets the sky system
                  4) A spectral translation
      -----------------------------------------------------------------------"""
      # Start pre-parsing the token string
      # This means that data between single quotes and curly brackets
      # are stored as one token, so that the evaluation can be postponed
      # and processed by special evaluators
      tokstr = tokenstr.strip() + ' #'
      tokens = mysplit(tokstr)

      self.tokens = []
      # This is a pre parsing step to replace one instance of the symbols 'PC' or 'AC'
      # by 'ncoords' instances. Each symbol then is parsed for the corresponding axis
      for i, t in enumerate(tokens):
         if t.upper() == 'PC':
            for j in range(ncoords):
               self.tokens.append(t)
         elif t.upper() == 'AC':
            for j in range(ncoords):
               self.tokens.append(t)
         else:
            self.tokens.append(t)

      self.tokens.append('#')
      self.ncoords = ncoords                     # i.e. the subset dimension
      self.END = '#'
      self.tokens.append(self.END)               # Append a symbol to indicate end of token list
      self.positions = []
      self.errmes = ""
      self.siunits = siunits
      self.types = types
      self.crpix = crpix
      self.naxis = naxis
      self.prevsky = None                        # A previous sky definition. Symbols {} will copy this
      self.source = source
      self.gipsygrids = gipsygrids
      if translations:
         self.strans, self.sunits = list(zip(*translations))
      else:
         self.strans = []
         self.sunits = []
      self.goal()


   def goal(self):
      #-------------------------------------------------------------------
      # The final goal is to find a number of positions which each
      # consist of 'ncoords' coordinates. The variable 'tpos' keeps
      # track of where we are in the token list.
      #-------------------------------------------------------------------
      tpos = 0
      while self.tokens[tpos] != self.END:
         position, tpos = self.getposition(tpos)
         if position == None:
            return
         self.positions.append(position)
         self.errmes = ''
         if tpos >= len(self.tokens):   # Just to be save
            break
      return

   
   def getposition(self, tpos):
      #-------------------------------------------------------------------
      # We need ncoords coordinates to get one position.
      # In the return value, the type is included. The type is
      # either 'g' for a pixel, 'w' for a world coordinate
      # and 'x' for a real error that should stop parsing.
      #-------------------------------------------------------------------
      numcoords = 0
      p = []
      numval = None
      while numcoords < self.ncoords and self.tokens[tpos] != self.END:
         val, typ, sky, spec, tposdelta = self.getcoordinate(tpos, numcoords)
         if val == None:
            return None, tpos
         lval = len(val)
         if numval == None:
            numval = lval
         elif lval != numval:
            self.errmes = "Error: Different number elements in first=%d, second=%d" % (numval, lval)
            return None , tpos
         tpos += tposdelta
         numcoords += 1
         p.append((val, typ, sky, spec))
      if numcoords != self.ncoords:
         self.errmes = "Error: Not enough coordinates for a position"
         return None, tpos
      return p, tpos



   def getcoordinate(self, tpos, coordindx):
      #-------------------------------------------------------------------
      # What is a coordinate? It can be a plain number (a grid) or a sequence
      # of plain numbers. It could also be a world coordinate associated
      # with a sky system or a world coordinate followed by a unit
      #-------------------------------------------------------------------
      number, typ, tposdelta = self.getnumber(tpos, coordindx)         
      if number != None:
         if typ == 'g' and self.gipsygrids:
            offgrid2pix = nint(self.crpix[coordindx])
            number = [w+offgrid2pix for w in number]
         return number, typ, '', '', tposdelta
      else:
         if typ != 'x':
            if self.types[coordindx] in ['longitude', 'latitude']:
               # Another possibility: it could be a coordinate with sky
               world, sky, tposdelta  = self.getsky(tpos, coordindx)
               if world != None:
                  return world, 'w', sky, '', tposdelta
            elif self.types[coordindx] == 'spectral':
               world, spectral, tposdelta = self.getspectral(tpos, coordindx)
               if spectral != None:
                  return world, 'w', '', spectral, tposdelta
            else:
               self.errmes = "Error: Not a grid nor world coord. sky or spectral parameter"
      return None, '', '', '', 0



   def getnumber(self, tpos, coordindx, unit=None):
      #-------------------------------------------------------------------
      # Allow a different unit if the unit is changed by a spectral translation
      #
      # POS='0 1 4'  '242 243 244' km/s           # Grouping of 3 grids and 3 world coordinates with unit
      # POS= 0 -243 km/s 0 -244 km/s
      #-------------------------------------------------------------------
      global source
      tryother = False
      currenttoken = self.tokens[tpos]
      number = None
      if currenttoken.startswith('{'):   # Fast way out. Cannot be a number
         return None, '', 0

      source = self.source

      # Try it as argument for Python's eval() with retrictions
      try:
         x = eval_restrict(currenttoken)
         if isinstance(x, (tuple, ndarray)):
            x = list(x)
         if not isinstance(x, list):       # These two types cannot be combined. x = list(x) will raise except.
            x = [x]
         number = x
      except Exception as message:
         self.errmes = usermessage(currenttoken, message)
         tryother = True

      # Not a number or numbers from a file. Perhaps a sexagesimal number
      # candidate = re_findall('[hmsHMSdD]', currenttoken)
      if tryother:
         tokupper = currenttoken.upper()
         h_ind = tokupper.find('H')
         d_ind = tokupper.find('D')
         candidate = (h_ind >= 0 or d_ind >= 0) and not (h_ind >= 0 and d_ind >= 0)
         if candidate:
            m_ind = tokupper.find('M')
            if m_ind >= 0:
               candidate = (m_ind > h_ind and m_ind > d_ind)
         if candidate:
            world, errmes = parsehmsdms(currenttoken, self.types[coordindx])
            if errmes == '':
               return world, 'w', 1
            else:
               self.errmes = usermessage(currenttoken, errmes)
               return None, '', 0
         elif currenttoken.upper() == 'PC':
            # 'PC' represents the projection center for spatial axes
            # but more general, it is the position of the reference pixel.
            # In GIPSY grids, this position is located somewhere in grid 0.
            # Note that this routine does not know whether pixels or grids
            # are entered. In the context of GIPSY we have to force the
            # pixel that represents 'PC' to a grid, because the calling
            # environment (GIPSY) expects the input was a grid. The conversion
            # is done elsewhere in this class (getcoordinate()).
            pc = self.crpix[coordindx]
            if self.gipsygrids:
               # Go from FITS pixel to grid
               pc -= nint(self.crpix[coordindx])
            return [pc], 'g', 1
         elif currenttoken.upper() == 'AC':
            # Next code is compatible to code in cotrans.c only we made the expression
            # simpler by rewriting the formula so that cotrans' offset is not necessary.
            n = self.naxis[coordindx]
            ac = 0.5 * (n+1)
            if self.gipsygrids:
               # Go from FITS pixel to grid. See also comment at 'PC'
               # We have to do this because elsewhere pixels are
               # converted to grids if gipsygrids=True. So compensate
               # that correction here.
               ac -= nint(self.crpix[coordindx])
            return [ac], 'g', 1
         else:
            # No number nor a sexagesimal number
            return None, '', 0

      if number == None:      # Just to be sure
         return None, '', 0

      # One or more numbers are parsed. The numbers could be modified if a unit follows
      nexttoken = self.tokens[tpos+1]
      if nexttoken != self.END:
         if unit != None:
            siunit = unit
         else:
            siunit = str(self.siunits[coordindx])
         unitfact = None
         unitfact, message = unitfactor(nexttoken, siunit)
         if unitfact is None:
             self.errmes = usermessage(nexttoken, message)
         else:
            world = [w*unitfact for w in number]
            return world, 'w', 2                 # two tokens scanned

      return number, 'g', 1


   def getsky(self, tpos, coordindx):
      #-------------------------------------------------------------------
      # Process sky systems.
      # A sky system is always associated with a spatial axis.
      # It is either one of the list 'eq', 'ec', 'ga' 'sg'
      # or it is a list enclosed in curly brackets '{', '}'
      # Examples:
      # Assume an equatorial system and a subset with two spatial axes:
      #
      # POS=EQ 50.3 23              ; World coordinate in the equatorial system and a grid
      # POS=Eq 50.3 23              ; Same input. Sky definition is case insensitive
      # POS=eq 50.3 eq 10.0         ; Both coordinates are world coords
      # POS=eq 50.3 ga 10.0         ; Mixed sky systems not allowed. No warning
      # POS=ga 210  ga -30.3        ; Two world coordinates in the galactic II system
      # POS=g 210  g -30.3          ; These are two positions in grids because g is a number
      # POS=ga 140d30m ga 62d10m    ; Use sexagesimal numbers
      # POS=eq 50.3 [] 10.0         ; Repeat sky system for the last coordinate
      # POS=eq 50.3 10.0 deg        ; Same input as previous. Units of spatial axis is degrees
      # POS=eq 50.3 10*60 arcmin    ; Same input. Note use of expression and compatible units
      # POS={eq} 50.3 {} 10.0       ; Group the sky system with square brackets
      #
      # POS={eq,J1983.5,fk5} 20.2 {} -10.0 ;  A world coordinate defined in an equatorial systems
      #                                       at equinox J1983.5 in the reference system fk5.
      #                                       The second coordinate is a world coordinate
      #                                       in the same sky system.
      # POS={eq,J1983.5,fk5} 20.2 -10 deg  ;  Second coordinate is a world coordinate in the
      #                                       same sky system.
      # POS={eq,J1983.5,fk5} 20.2 -10      ;  Not allowed: A world coordinate defined in an
      #                                       equatorial systems at equinox J1983.5 in the reference
      #                                       system fk5. Followed by a grid. This cannot be evaluated
      #                                       because a solution of the missing coordinate can
      #                                       only be found in the native sky system.
      #-------------------------------------------------------------------
      self.errmess = ""
      currenttoken = self.tokens[tpos]

      try:
         sk, errmes = parseskysystem(currenttoken)
         if sk[0] == None:      # Empty skydef {}
            skydef = ''
            if self.prevsky != None:
               skydef = self.prevsky
         else:
            skydef = sk         # Copy the PARSED sky definition!
      except Exception as message:
         skydef = None
         self.errmes = usermessage(currenttoken, message)

      if skydef != None:
         nexttoken = self.tokens[tpos+1]
         if nexttoken != self.END:
            number, typ, tposdelta = self.getnumber(tpos+1, coordindx)
            if number != None:
               return number, skydef, tposdelta+1
            else:
               # No number no world coordinate
               self.errmes = "Error: '%s' is a sky system but not followed by grid or world coord." % currenttoken
               return None, '', 0
         else:
            # A sky but nothing to parse after this token
            self.errmes = "Error: '%s' is a sky system but not followed by grid or world coord." % currenttoken
            return None, '', 0
      return None, '', 0


   def getspectral(self, tpos, coordindx):
      #-------------------------------------------------------------------
      # This routine deals with spectral axes. A spectral translation must
      # be one of the allowed translations for the data for which a possible
      # is required. The translation option must be given before a number.
      # It can be followed by a unit. We expect the user has (FITS) knowledge
      # about the meaning of the translations.
      # Examples:
      # POS= 0 243 km/s
      # POS= 0 vopt 243 km/s
      # POS= 0 beta -243000/c      ; beta = v/c
      #-------------------------------------------------------------------
      currenttoken = self.tokens[tpos]           # One of VOPT, VRAD etc.
      indx = minmatch(currenttoken, self.strans, 0)
      if indx >= 0:
         spectral = self.strans[indx]
         unit = self.sunits[indx]
         nexttoken = self.tokens[tpos+1]
         if nexttoken != self.END:
            number, typ, tposdelta = self.getnumber(tpos+1, coordindx, unit)
            if number != None:
               return number, spectral, tposdelta+1
            else:
               # No number and no world coordinate
               self.errmes = "Error: '%s' is a spectral trans. but without grid or world c." %  currenttoken
               return None, '', 0
         else:
            # A spectral translation but nothing to parse after this token
            self.errmes = "Error: '%s' is a spectral trans. but without grid or world c." %  currenttoken
            return None, '', 0
      else:
         # Not a spectral translation:
         return None, '', 0


def dotrans(parsedpositions, subproj, subdim, mixpix=None):
   #-------------------------------------------------------------------
   """
   This routine expects pixels in gcoord and will also return pixels
   """
   #-------------------------------------------------------------------
   skyout_orig = subproj.skyout            # Store and restore before return
   errmes = ''                             # Init error message to no error 
   r_world = []
   r_pixels = []
   subsetunits = None
   #if gipsygrids:
   #   # First we determine the -integer- offsets to transform grids
   #   # into 1-based FITS pixels
   #   offset = [0.0]*subdim
   #   for i in range(subdim):
   #       offset[i] = nint(subproj.crpix[i])
          
   for p in parsedpositions:
      wcoord = [unknown]*subdim            # A list with tuples with a number and a conversion factor
      gcoord = [unknown]*subdim
      empty  = [unknown]*subdim
      # Reset sky system to original.
      subproj.skyout = None
      # p[i][0]: A list with one or more numbers
      # p[i][1]: the mode ('g'rid or 'w'orld)
      # p[i][2]: the sky definition
      # p[i][3]: the spectral definition
      skyout = None                        # Each position can have its own sky system
      for i in range(subdim):              # A position has 'subdim' coordinates
         try:
            numbers = asarray(p[i][0])        # Contents of coordinate number 'i' (can be a list with numbers)
         except:
            errmes = "Sequence not ok. Perhaps array is not flat"
            return [], [], [], errmes
         # Numbers here is always a LIST with 1 or more numbers. Make a NumPy
         # array of this list to facilitate grid to pixel conversions
         if numbers.shape == ():
            N = 1
         else:
            N = numbers.shape[0]
         if p[i][1] == 'g':
            # Convert from grid to pixel
            gcoord[i] = numbers
            #if gipsygrids:
            #   gcoord[i] += offset[i]
            wcoord[i] = asarray([unknown]*N)
         else:
            gcoord[i] = asarray([unknown]*N)
            wcoord[i] = numbers
         empty[i] = asarray([unknown]*N)
         nsky = p[i][2]
         # We parsed the skyout to the tuple format so we can compare 2 systems
         # i.e. compare two tuples
         if nsky != '':
            if skyout == None:         # Not initialized: start with this sky
               skyout = nsky
            else:
               if nsky != skyout:
                  errmes = "Mixed sky systems not supported"
                  return [], [], [], errmes

      if mixpix != None:
         gcoord.append(asarray([mixpix]*N))
         wcoord.append(asarray([unknown]*N))
         empty.append(asarray([unknown]*N))

      spectrans = None                  # Input spectral translation e.g. POS=vopt 105000
      for i in range(subdim):
         # The spectral axis could be any coordinate in a position, so
         # check them all. WCSLIB allows for only one spectral
         # axis in a dataset (which in practice makes sense).
         # So break if we find the first spectral translation
         spectrans = p[i][3]
         if spectrans:
            break
      if spectrans:
         newproj = subproj.spectra(spectrans)
      else:
         newproj = subproj
      if skyout != None and skyout != "":
         newproj.skyout = skyout
      else:
         newproj.skyout = None             # Reset sky system

      # The mixed method needs two tuples with coordinates. Each coordinate
      # can be a list or a numpy array. The mixed routine recognizes
      # pixel only input and world coordinate only input and is optimized
      # to deal with these situations.
      try:
         wor, pix = newproj.mixed(tuple(wcoord), tuple(gcoord))
      except wcs.WCSerror as message:
         errmes = str(message.args[1])  # element 0 is error number
         # Restore to the original projection object
         # Note that 'newproj' could be pointer to 'subproj' which shares the same skyout
         # and the skyout could have been changed.
         subproj.skyout = skyout_orig
         return [], [], [], errmes

      # Now we have the pixels and want the world coordinates in the original
      # system. Then first reset the skyout attribute.
      subproj.skyout = skyout_orig
      # Get world coordinates in system of input projection system
      wor = subproj.toworld(tuple(pix))
      subsetunits = subproj.cunit     # Set units to final units
      
      # pix is a tuple with 'subdim' coordinates. But note: each coordinate
      # can be an array with one or more numbers.
      # Make a NumPy array of this tuple and transpose the array
      # to get one position (i.e. subdim coordinates) in one row.
      wt = asarray(wor).T
      pt = asarray(pix).T
      # Append to the results list. Note that list appending is more flexible than
      # NumPy array concatenation.
      for w, p in zip(wt, pt):
         r_world.append(w)
         r_pixels.append(p)
   return asarray(r_world), asarray(r_pixels), subsetunits, errmes


def str2pos(postxt, subproj, mixpix=None, gridmode=False):
   #-------------------------------------------------------------------
   """
   This function accepts a string that represents a position in the
   world coordinate system defined by *subproj*. If the string
   contains a valid position, it returns a tuple with numbers that
   are the corresponding pixel coordinates and a tuple with
   world coordinates in the system of *subproj*. One can also
   enter a number of positions. If a position could not be
   converted then an error message is returned.

   :param postxt:   The position(s) which must be parsed.
   :type postxt:    String
   :param subproj:  A projection object (see :mod:`wcs`).
                    Often this projection object will describe
                    a subset of the data structure (e.g. a
                    channel map in a radio data cube).
   :type subproj:   :class:`wcs.Projection` object
   :param mixpix:   For a world coordinate system with one spatial
                    axis we need a pixel coordinate for the missing
                    spatial axis to be able to convert between
                    world- and pixel coordinates.
   :type mixpix:    Float
   :param gridmode: If True, correct pixel position for CRPIX to
                    get grid coordinates where the pixel at CRPIX is 0
   :type gridmode:  Boolean
   
   
   :Returns:

   This method returns a tuple with four elements:

   * a NumPy array with the parsed positions in world coordinates
   * a NumPy array with the parsed positions in pixel coordinates
   * A tuple with the units that correspond to the axes
     in your world coordinate system.
   * An error message when a position could not be parsed

   Each position in the input string is returned in the output as an
   element of a numpy array with parsed positions. A position has the same
   number of coordinates are there are axes in the data defined by
   the projection object.

   :Examples:
      ::
         
         from kapteyn import wcs, positions

         header = {  'NAXIS'  : 2,
                     'BUNIT'  :'w.u.',
                     'CDELT1' : -1.200000000000E-03,
                     'CDELT2' : 1.497160000000E-03,
                     'CRPIX1' : 5,
                     'CRPIX2' : 6,
                     'CRVAL1' : 1.787792000000E+02,
                     'CRVAL2' : 5.365500000000E+01,
                     'CTYPE1' :'RA---NCP',
                     'CTYPE2' :'DEC--NCP',
                     'CUNIT1' :'DEGREE',
                     'CUNIT2' :'DEGREE'}
         
         proj = wcs.Projection(header)
         
         position = []
         position.append("0 0")
         position.append("eq 178.7792  eq 53.655")
         position.append("{eq} 178.7792  {} 53.655")
         position.append("{} 178.7792  {} 53.655")
         position.append("178.7792 deg  53.655 deg")
         position.append("11h55m07.008s 53d39m18.0s")
         position.append("{eq, B1950,fk4} 178.7792  {} 53.655")
         position.append("{eq, B1950,fk4} 178.12830409  {} 53.93322241")
         position.append("{fk4} 178.12830409  {} 53.93322241")
         position.append("{B1983.5} 11h55m07.008s {} 53d39m18.0s")
         position.append("{eq, B1950,fk4, J1983.5} 178.12830409  {} 53.93322241")
         position.append("ga 140.52382927 ga  61.50745891")
         position.append("su 61.4767412, su 4.0520188")
         position.append("ec 150.73844942 ec 47.22071243")
         position.append("eq 178.7792 0.0")
         position.append("0.0, eq 53.655")
         for pos in position:
            poswp = positions.str2pos(pos, proj)
            if poswp[3] != "":
               raise Exception, poswp[3]
            world = poswp[0][0]
            pixel = poswp[1][0]
            units = poswp[2]
         print pos, "=", pixel, '->',  world , units

   """
   #-------------------------------------------------------------------
   if not isinstance(postxt, six.string_types):
      raise TypeError("str2pos(): parameter postxt must be a String")
   subdim = len(subproj.types)
   if mixpix != None:
      subdim -= 1
   r_world = []
   r_pixels = []

   parsedpositions = Coordparser(postxt,         # Text containing positions as entered by user
                                 subdim,         # The number of coordinates in 1 position
                                 subproj.units,  # Units (for conversions) in order of subset axes
                                 subproj.types,  # Axis types to distinguish spatial and spectral coords.
                                 subproj.crpix,  # Crpix values for 'PC' (projection center)
                                 subproj.naxis,  # Axis lengths for center 'AC'
                                 subproj.altspec,# List with allowed spectral translations
                                 subproj.source, # Get access to header
                                 gipsygrids=gridmode)

   if parsedpositions.errmes:
      if postxt != '':
         return [], [], [], parsedpositions.errmes
   else:
      # Note that the array with parsed positions cannot contain grids,
      # because the routine that converts them expects pixel
      # coordinates (because mixpix is a pixelcoordinate)
      wor, pix, subsetunits, errmes = dotrans(parsedpositions.positions,
                                              subproj,
                                              subdim,
                                              mixpix)
      if errmes != '':
         return [], [], [], errmes

   return wor, pix, subsetunits, ''


def dotest():
   def printpos(postxt, pos):
      # Print the position information
      world, pixels, units, errmes = pos
      print(("Expression        : %s"%postxt))
      if errmes == '':
         print(("World coordinates :", world, units))
         print(("Pixel coordinates :", pixels))
      else:
         print(errmes)
      print("")
      
   header = { 'NAXIS'  : 3,
              'BUNIT'  : 'w.u.',
              'CDELT1' : -1.200000000000E-03,
              'CDELT2' : 1.497160000000E-03,
              'CDELT3' : -7.812500000000E+04, 
              'CRPIX1' : 5,
              'CRPIX2' : 6,
              'CRPIX3' : 7,
              'CRVAL1' : 1.787792000000E+02,
              'CRVAL2' : 5.365500000000E+01,
              'CRVAL3' : 1.4154482500E+09,  # Tuned to fit Vopt
              'CTYPE1' : 'RA---NCP',
              'CTYPE2' : 'DEC--NCP',
              'CTYPE3' : 'FREQ-OHEL',
              'CUNIT1' : 'DEGREE',
              'CUNIT2' : 'DEGREE',
              'CUNIT3' : 'HZ',
              'DRVAL3' : 1.050000000000E+03,
              'DUNIT3' : 'KM/S',
              'FREQ0'  : 1.420405752e+9,
              'INSTRUME' : 'WSRT',
              'NAXIS1' : 10,
              'NAXIS2' : 10,
              'NAXIS3' : 10
            }
   #wcs.debug=True
   origproj = wcs.Projection(header)

   print("-------------- Examples of numbers and constants, missing spatial--------------\n")
   proj = origproj.sub((1,3,2))
   mixpix = 6
   userpos = ["(3*4)-5 1/5*(7-2)",
              "abs(-10), sqrt(3)",
              "sin(radians(30)), degrees(asin(0.5))",
              "cos(radians(60)), degrees(acos(0.5))",
              "pi, tan(radians(45))-0.5, 3*4,0",
              "sin(arange(10)), range(10)",
              "atan2(2,3), 192",
              "atan2(2,3)  192",
              "[pi,2]*3, [e**2,tan(pi)]*3",
              "[1,2, atan2(2,0.9)] [pi**2::3]",
              "c_/299792458.0,  G_/6.67428e-11",
              "sin([1,2,3]*pi),cos([1,2,3]*pi)",
              "[1,2,3,4], sin(radians([0,30,60,90]))",
              "10**[1,2,3], log2([1,2,3])",
              "[1,2,3,4], sin(radians([0,30,60,90]))",
              "deg([1,2,3,4]), rad([0,30,60,90])",
              "[pi::3], [1,2,3]",
              "[pi::3]*3, [1:3]**3",
              "[1,6/3,3,4]**3, pi*[1,2,3,4]",
              "[10:1], [1:10]",
              "[10:0:-2], [0:10:2]",
              "linspace(0,3,4), tan(radians(linspace(3,0,4)))",
              "'1/2 ,sin(pi), 4' range(3)",
              "[3:5,10]/2 range(4)",
              "'[pi]+[1,2]' [1::3]",
              "'[pi]*3' range(3)",
              "'[sin(x) for x in range(4)]' range(4)"
             ]
   for postxt in userpos:
      wp = str2pos(postxt, proj, mixpix=mixpix)
      printpos(postxt, wp)
   print('')


   print("-------------- Examples of 1 spatial axis and a missing spatial--------------\n")
   proj = origproj.sub((1,2))
   mixpix = 6
   userpos = ["(3*4)",
              "10",
              "178.7792*60 arcmin",
              "{} 178.7792",
              "{B2000} 178.7792",  # Not allowed
              "'178.7792, 178.7794, 178.7796' deg",
              "[178.7792, 178.7794, 178.7796] deg",
              "[178.7792:178.7796:0.0002] deg",
              "arange(178.7792, 178.7796, 0.0002) deg",
              "linspace(178.7792, 178.7796, 4) deg",
              "linspace(178.7792, 178.7796, 4) ?",
              "linspace(178.7792, 178.7796, 4) Units",
              "linspace(178.7792, 178.7796, 4) Un",
              "3*arange(178.7792/3, 178.7796/3, 0.0002) deg",
              "eq 178.7792",  # Not allowed
              "11h55m07.008s",
              "178d40m",
              "178d",
              "178d10m 178d20m30.5s"
             ]
   for postxt in userpos:
      wp = str2pos(postxt, proj, mixpix=mixpix)
      printpos(postxt, wp)
   print('')


   print("-------------- Examples of units, spectral translations and grouping -----------\n")
   proj = origproj.sub((3,))
   userpos = ["7 0",
              "1.4154482500E+09 hz",
              "1.4154482500E+03 Mhz",
              "1.4154482500 Ghz",
              "vopt 1.05000000e+06",
              "vopt 1050 km/s",
              "vopt 0",
              "vrad 1.05000000e+06",
              # f/c is lambda. For this f (=CRVAL3) this gives lambda:
              # 299792458.0/1.4154482500E+09 = 0.21180036642102598
              # The wave number is 1/lambda. If we use this as world coordinate
              # then it should convert to crpix (=7)
              "wavn [100/21.180036642102598/100, 4.76/100, 4.7/100] 1/cm",
              "FREQ 1.4154482500E+09",  # Note FREQ is not FREQ-OHEL
              "0 7 10 20",
              "'1.41, 1.4154482500, 1.42, 1.43' Ghz",
              "[1.41, 1.4154482500, 1.42, 1.43] Ghz"
              ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')

   print("--------- Output of previous coordinates in terms of VOPT:----------\n")
   proj2 = proj.spectra('VOPT-???')
   userpos = ["7",
              "freq 1.4154482500E+09 hz",
              "fr 1.4154482500E+03 Mhz",
              "fr 1.4154482500 Ghz",
              "vopt 1.05000000e+06",
              "vopt 1050 km/s",
              "vopt 0",
              "vrad 1.05000000e+06",
              "FREQ 1.4154482500E+09",
              "0 7 10 20 70.233164383215",
              "FREQ '1.41, 1.4154482500, 1.42, 1.43' Ghz",
              "FR [1.41, 1.4154482500, 1.42, 1.43] Ghz"
              ]
   for postxt in userpos:
      wp = str2pos(postxt, proj2)
      printpos(postxt, wp)
   print('')

   print("--------- Sky systems and AC&PC ----------\n")
   proj = origproj.sub((1,2))
   userpos = ["0 0",
              "5,6 0 0 3,1",
              "eq 178.7792  eq 53.655",          # e 10 will not work because e is a symbol and an ambiguous sky system`
              "eq [178.7792:178.7796:0.0002] eq [53.655::3]",
              "{eq} 178.7792  {} 53.655",
              "178.7792 deg  53.655 deg",
              "11h55m07.008s 53d39m18.0s",
              "{eq, B1950,fk4} 178.7792  {} 53.655",
              "{eq, B1950,fk4} 178.12830409  {} 53.93322241",
              "{fk4} 178.12830409  {} 53.93322241",
              "{B1983.5} 11h55m07.008s {} 53d39m18.0s",
              "{eq, B1950,fk4, J1983.5} 178.12830409  {} 53.93322241",
              "ga 140.52382927 ga  61.50745891",
              "ga 140.52382927 {}  61.50745891",
              "su 61.4767412, su 4.0520188",
              "su [61.47674:61.47680:0.00002], {} [4.0520188::4]",
              "ec 150.73844942 ec 47.22071243",
              "{} 178.7792 6.0",
              "5.0, {} 53.655",
              "{eq} '178.779200, 178.778200, 178.777200' {} '53.655000, 53.656000, 53.657000'",
              "PC",
              "ac"]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')

   print("--------- Same as previous but in terms of Galactic coordinates ----------\n")
   sky_old = proj.skyout
   proj.skyout = "ga"
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')
   proj.skyout = sky_old

   print("--------- XV map: one spatial and one spectral axis ----------\n")
   proj = origproj.sub((2,3,1))
   mixpix = 5
   print(("Spectral translations: ", proj.altspec))
   userpos = ["{} 53.655 1.4154482500E+09 hz",
              "{} 53.655 1.4154482500E+03 Mhz",
              "53.655 deg 1.4154482500 Ghz",
              "{} 53.655 vopt 1.05000000e+06",
              "{} 53.655 , vopt 1050000 m/s",
              "0.0 , vopt 1050000 m/s",
              "10.0 , vopt 1050000 m/s",
              "{} 53.655 vrad 1.05000000e+06",
              "{} 53.655 FREQ 1.4154482500e+09",
              "{} 53.655 wave 21.2 cm",
              "{} [53.655, 53.6555] wave [21.2, 21.205] cm",
              "{} '53.655, 53.6555' wave '21.2, 21.205' cm",
              "{} [53.655::5] wave linspace(21.2,21.205,5) cm",
              "{} 53.655 vopt c_/300 m/s"]
   for postxt in userpos:
      wp = str2pos(postxt, proj, mixpix=mixpix)
      printpos(postxt, wp)
   print('')

   # Create an Ascii file with dummy data for testing the READCOL command
   # The data in the Ascii file is composed of a fixed sequence of grids
   # that are transformed to their corresponding galactic coordinates. 
   asciifile = "test123.txt"
   f = open(asciifile, 'w')
   s = "! Test file for Ascii data and the FILE command\n"
   f.write(s)
   s = "! Extra comment to distinguish between lines and rows\n"
   f.write(s)
   for i in range(10):
      f1 = 1.0* i; f2 = f1 * 2.0; f3 = f2 * 1.5; f4 = f3 * 2.5
      s = "%f %.12f %f %.12f\n" % (f1, f2, f3, f4)
      f.write(s)
   f.write("\n")
   for i in range(10,15):
      f1 = 1.0* i; f2 = f1 * 2.0; f3 = f2 * 1.5; f4 = f3 * 2.5
      s = "%f %.12f %f %.12f\n" % (f1, f2, f3, f4)
      f.write(s)
   f.close()


   asciifile = "hmsdms.txt"
   f = open(asciifile, 'w')
   s = "! Test file for Ascii data and the READHMS/READDMS command\n"
   f.write(s)
   s = "11 57 .008 53 39 18.0\n"; f.write(s)
   s = "11 58 .008 53 39 17.0\n"; f.write(s)
   s = "11 59 .008 53 39 16.0\n"; f.write(s)
   f.close()

   print("--------- Reading from file ----------\n")
   proj = origproj.sub((1,2))
   userpos = [ 'readcol("test123.txt") readcol("test123.txt",3)',
               '10*readcol("test123.txt") sin(readcol("test123.txt",3))',
               'readcol("test123.txt", col=1) readcol("test123.txt", col=3)',
               'readcol("test123.txt", col=1) readcol("test123.txt", col=3)',
               'readcol("test123.txt", col=1, toline=5) readcol("test123.txt", col=3, toline=5)',
               # There is an empty line at line 13
               'readcol("test123.txt", col=1, toline=14) readcol("test123.txt", col=3, toline=14)',
               'readcol("test123.txt", col=1, fromline=5) readcol("test123.txt", col=3, fromline=5)',
               'readcol("test123.txt", col=1, fromrow=5) readcol("test123.txt", col=3, fromrow=5)',
               'readcol("test123.txt", col=1, torow=5) readcol("test123.txt", col=3, torow=5)',
               'readcol("test123.txt", col=1, torow=12) readcol("test123.txt", col=3, torow=12)',
               'readcol("test123.txt", col=1, rowstep=2) readcol("test123.txt", col=3, rowstep=2)',
               'readcol("test123.txt", col=1, rows=[2,4,14]) readcol("test123.txt", col=3, rows=[2,4,14])',
               'readcol("test123.txt", col=1, fromrow=4, torow=14, rowstep=2) linspace(0,1,6)',
               'readcol("test123.txt", col=1, fromrow=4, torow=14, rowstep=2) [4:14:2]',
               '{} readcol("test123.txt", col=1) {} readcol("test123.txt", col=3)',
               'ga readcol("test123.txt", col=1) ga readcol("test123.txt", col=3)',
               'readcol("test123.txt", col=1) deg readcol("test123.txt", col=3) deg',
               '{} readhms("hmsdms.txt",1,2,3) {} readdms("hmsdms.txt",4,5,6)',
               '1.1*readhms("hmsdms.txt",1,2,3)-5 sin(readdms("hmsdms.txt",4,5,6)-10.1)',
               '{} readhms("hmsdms.txt",col1=1, col3=2, col2=3) {} readdms("hmsdms.txt",4,5,6)',
             ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')

   print("--------- Reading from header ----------\n")
   proj = origproj.sub((1,2))
   userpos = [ '{} header("crval1") {} header("crval2")',
               '3*header("crpix1") sin(header("crpix2"))' ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')

   print("--------- Problem strings and error messages ----------\n")
   proj = origproj.sub((1,2))
   userpos = ["33",
              "1 2 3",
              "eq 178, ga 40",
              "22 {}",
              "10, 53 heg",
              "readcol() readcol()",  # No file name
              'readcol("test123.txt, 1) readcol("test123.txt", 3)', # missing "
              'readcol("test123.txt", 1, range(1:4)) 3:5',   # 3:5 unknown syntax
              'readcol("test123.txt", 3, rows=[1,2,3,4])',
              'readcol("test123.txt", 1, rowsslice(5,None)) readcol("test123.txt", 2, rowslice=(5,None))',
              'readcol("test123.txt", 1, row=3) readcol("test123.txt", 2, row=3)',
              '{ga} readcol("test123.txt", 2) {} readcol("test123wcsRADECFREQ".txt, 4)', # mixed
              '{ga} readcol("test123.txt", col=1) {} readcol("test123.txt", col=3)',
              "'1, 2, a[1,2,3]' range(5)",  # Array in list is not allowed
              'readcol(exec saveeval.py) readcol(test123.txt,3)',
              'readcol("test123.txt", issequence(3)+1) readcol(test123.txt,3)',
              'readcol("test123.txt", eval("pi=2")) readcol(test123.txt,3)',
              "readcol('test123.txt') readcol('test123.txt',3)", # Use double quotes for keys
              "'[1:3], [4:7]', range(2)"  # List in list not allowed
            ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print('')

   import readline
   upos = 'xxx'
   proj = origproj.sub((1,2)); mixpix = None
   while upos != '':
      upos = eval(input("Enter position(s) ..... [quit loop]: "))
      readline.add_history(upos) 
      if upos != '':
         wp = str2pos(upos, proj, mixpix=mixpix)
         printpos(upos, wp)


if __name__ == "__main__":
      dotest() 