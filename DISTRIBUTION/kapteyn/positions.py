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
allow only numbers as their input parameters. Also the transformations apply to
the native coordinate system, i.e. it expects that world coordinates
are given for the system that is described by the Projection object.

Often one wants more flexibility. For instance in interaction with the user, positions
can be used to plot markers on a map. But what if you have positions
from a FK5 catalog and your map is in Galactic coordinates or you want to know
what the optical velocity is, given a radio velocity, for a spectral axis
which has frequency as its primary type?

This module enables a user/programmer to specify positions in either
pixel- or world coordinates. Its functionality is provided by a parser
which converts strings with position information into pixel coordinates
and world coordinates. Let's list some options with examples how to use
method :meth:`str2pos`.

Assume we have a projection object *pr* and you
want to know the world coordinates *w* and the pixels *p* for a given
string (*u* are the units of the world coordinates and *e* is an error message):

   * Expressions for the input of numbers.
     Example: ``w,p,e = str2pos('[pi]*3, [e**2]*3`', pr)``
   * Use of physical constants.
     Example: ``w,p,u,e = str2pos('c_/299792458.0,  G_/6.67428e-11', pr)``
   * Use of units to set world coordinates
     Example: ``w,p,u,e = str2pos('178.7792 deg  53.655 deg', pr)``
   * Mix of pixels and world coordinates.
     Example: ``w,p,u,e = str2pos('5.0, 53.655 deg', pr)``
   * Support of sky definitions.
     Example: ``w,p,u,e = str2pos('{eq, B1950,fk4, J1983.5} 178.12830409  {} 53.93322241', pr)``
   * Support for spectral translations.
     Example: ``w,p,u,e = str2pos('vopt 1050 km/s', pr)``
   * Coordinates from text file on disk.
     Example: ``w,p,u,e = str2pos('readcol("test123positions.txt", col=2)', pr)``
   * Support for maps with only one spatial axis (e.g. XV maps).
     Example: ``w,p,u,e = str2pos('{} 53.655 1.415418199417E+03 Mhz', p, mixpix=6)``
   * Use of sexagesimal notation of spatial world coordinates.
     Example: ``w,p,u,e = str2pos('11h55m07.008s 53d39m18.0s', pr)``
   * Read header items.
     Example: ``w,p,u,e = str2pos("{} header('crval1') {} header('crval2')", pr)``
   * Units, sky definitions and spectral translations are minimal matched
     to the full names.


Introduction
------------

Physical quantities in a data structure that represents a physical measurement are usually
measurements at fixed positions in the sky or at spectral positions such as
Doppler shifts, frequencies or velocities. These positions are examples of so called
**World Coordinates**. In these data structures the quantities are identified by their
pixel coordinates. Following the rules for FITS files, a pixel coordinate starts with
1 for the first pixel on an axis and runs to *NAXISn* which is a header item
that sets the length of the n-th axis in the structure.

Assume you have a data structure representing an optical image of a part of the sky
and you need to mark a certain feature in the image or need to retrieve the intensity
of a certain pixel. Then usually it is easy to identify the pixel using
pixel coordinates. But sometimes you have positions (e.g. from external sources like
catalogs) given in world coordinates and then it would be convenient if you could specify
positions in those coordinates. Module :mod:`wcs` provides methods for conversions between
pixel coordinates and world coordinates given a description of the world coordinate
system (as defined in a header). Module :mod:`celestial` converts world coordinates
between different sky- and reference systems and/or epochs.
In this module we combined the functionality of :mod:`wcs` and :mod:`celestial`
to write a position parser.
Note that a description of a world coordinate system can be either a FITS header or
a Python dictionary with FITS keywords.


How to use this module
----------------------

This module is included in other modules of the Kapteyn Package, but
it can also be imported in your own scripts so that you are able to convert
positions in a string to pixel- and world coordinates.
It is also possible to use it as a test application (on the command line
type: ``python positions.py`` where you can add your own strings for conversion.
The source of this test run can be found in function :func:`dotest` in this module.

To get the idea, we list a short example::

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

Its output is::

   pixels: [[ 5.  6.]]
   world coordinates: [[ 178.7792   53.655 ]] ('deg', 'deg')


Position syntax
---------------

Number of coordinates
.......................

A position has the same number of coordinates as the number of axes that are
defined by the Projection object. So a position in a 2-dim map has two coordinates.
One can enter 1 position or a sequence of positions as in:

>>> pos="0,1  4,5  2,3"

Numbers are separated either by a space or a comma.
Coordinates can also be **grouped**. Elements in a group are processed in one pass
and they represent one coordinate in a position.
They can be prepended by a modifier or appended by a unit. The position above
could also have been written as:

>>> pos="'0,4,2' '1,5,3'"

Coordinates enclosed by single quotes are parsed by Python's expression evaluator,
*eval()*.
The arguments of this function are separated by comma's only.
Other examples of grouping are listed in the section about reading data from
disk with *READCOL()* and in the section about the *EVAL()* function.


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

Two pixel coordinates in a two dimensional wcs:
   
>>> pos = "10 20"       # Pixel position 10, 20
>>> pos = "10 20 10 30" # Two pixel positions

One can enter everything that Python's *eval()* function can parse.
Lists and list comprehension is allowed.

>>> pos = "(3*4)-5 1/5*(7-2)",
>>> pos = "abs(-10), sqrt(3)",
>>> pos = "sin(radians(30)), degrees(asin(0.5))",
>>> pos = "cos(radians(60)), degrees(acos(0.5))",
>>> pos = "pi, tan(radians(45))-0.5, 3*4,0",
>>> pos = "[sin(x) for x in range(10)], range(10)",
>>> pos = "atan2(2,3), 192",
>>> pos = "atan2(2,3)  192",
>>> pos = "[pi]*3, [e**2]*3"

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


Special pixel coordinates
..........................

For the reference position in a map we can use symbol 'PC' (Projection center).
The center of your data structure is set with symbol 'AC'.
You can use either one symbol or the same number of symbols as there are
axes in your data structure.

>>> pos = "pc"     # Pixel coordinates of the reference pixel
>>> pos = "PC pc"  # Same as previous. Note case insensitive parsing
>>> pos = "AC"     # Center of the map in pixel coordinates


World coordinates
..................

World coordinates can be distinguished from pixel coordinates. A world
coordinate is:

   * a coordinate followed by a (compatible) unit. Note that the
     units of the world coordinate are given in the (FITS) header in keyword *CUNIT*.
   * a coordinate prepended by a definition for a sky system or a spectral system.
   * a coordinate entered in sexagesimal notation. (hms/dms)

.. note::

   One can mix pixel- and world coordinates in a description of a position.

**Units**

For a two dimensional data structure (e.g. an optical image of part of the sky)
we can enter a position in world coordinates as:

>>> pos = 178.7792 deg  53.655 deg

But we can also use compatible units:

>>> pos = "178.7792*60 arcmin  53.655 deg"   # Use of a compatible unit if CUNIT is "DEGREE"
>>> pos = "0 1.41541820e+09 Hz"              # Mix of pixel coordinate and world coordinate
>>> pos = "0 1.41541820 GHz"                 # Same position as previous using a compatible unit

Units are minimal matched against a list with known units. The parsing of units
is case insensitive. The list with known units is:

   * angles: ['DEGREE','ARCMIN', 'ARCSEC', 'MAS', 'RADIAN'
     'CIRCLE', 'DMSSEC', 'DMSMIN', 'DMSDEG', 'HMSSEC', 'HMSMIN', 'HMSHOUR']
   * distances: ['METER', 'ANGSTROM', 'NM', 'MICRON', 'MM', 'CM',
     'INCH', 'FOOT', 'YARD', 'M', 'KM', 'MILE', 'PC', 'KPC', 'MPC', 'AU', 'LYR']
   * time: ['TICK', 'SECOND', 'MINUTE', 'HOUR', 'DAY', 'YR']
   * frequency: ['HZ', 'KHZ','MHZ', 'GHZ']
   * velocity: ['M/S', 'MM/S', 'CM/S', 'KM/S']
   * temperature: ['K', 'MK']
   * flux (radio astr.): ['W/M2/HZ', 'JY', 'MJY']
   * energy: ['J', 'EV', 'ERG', 'RY']

**Sky definitions**

If a coordinate follows a sky definition it is parsed as a world coordinate.
A sky definition is either a case insensitive minimal match from the list::

  'EQUATORIAL','ECLIPTIC','GALACTIC','SUPERGALACTIC'

or it is a definition between curly brackets which can contain
the sky system, the reference system, equinox and epoch of observation.
The documentation for sky definitions is found in module :mod:`celestial`.
Epochs are described in :mod:`celestial.epochs`.

An empty string between curly brackets (e.g. {}) followed by a number
implies a world coordinate in the native sky system. 

Examples:

>>> pos = "{eq} 178.7792  {} 53.655"                      # As a sky definition between curly brackets
>>> pos = "{} 178.7792 {} 53.655"                         # A world coordinate in the native sky system
>>> pos = "{eq,B1950,fk4} 178.12830409  {} 53.93322241"   # With sky system, reference system and equinox
>>> pos = "{fk4} 178.12830409  {} 53.93322241"            # With reference system only.
>>> pos = "{eq, B1950,fk4, J1983.5} 178.1283  {} 53.933"  # With epoch of observation (FK4 only)
>>> pos = "{eq B1950 fk4 J1983.5} 178.1283  {} 53.933"    # With space as separator
>>> pos = "ga 140.52382927 ga 61.50745891"                # Galactic coordinates
>>> pos = "ga 140.52382927 {} 61.50745891"                # Second definition copies from first
>>> pos = "su 61.4767412, su 4.0520188"                   # Supergalactic
>>> pos = "ec 150.73844942 ec 47.22071243"                # Ecliptic
>>> pos = "{} 178.7792 6.0"                               # Mix world- and pixel coordinate
>>> pos = "5.0, {} 53.655"                                # Mix with world coordinate in native system

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

>>> pos = "{} 53.655 1.415418199417E+09 hz"    # Spatial and spectral world coordinate
>>> pos = "{} 53.655 1.415418199417E+03 Mhz"   # Change Hz to MHz
>>> pos = "53.655 deg 1.415418199417 Ghz"      # to GHz
>>> pos = "{} 53.655 vopt 1.05000000e+06"      # Use spectral translation to enter optical velocity
>>> pos = "{} 53.655 , vopt 1050 km/s"         # Change units
>>> pos = "10.0 , vopt 1050000 m/s"            # Combine with a pixel position
>>> pos = "{} 53.655 vrad 1.05000000e+06"      # Radio velocity
>>> pos = "{} 53.655 vrad 1.05000000e+03 km/s" # Radio velocity with different unit
>>> pos = "{} 53.655 FREQ 1.41541820e+09"      # A Frequency
>>> pos = "{} 53.655 wave 21.2 cm"             # A wave length with alternative unit
>>> pos = "{} 53.655 vopt c_/285.51662         # Use speed of light constant to get number in m/s


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
   is invalid, ``10.0 , vopt 1050000 m/s`` is ok but
   also ``{} 53.655 FREQ 1.415418199417e+09"`` is correct.
   
**Sexagesimal notation**

Read the documentation at :func:`parsehmsdms` for the details.
Here are some examples:

>>> pos = "11h55m07.008s 53d39m18.0s"
>>> pos = "{B1983.5} 11h55m07.008s {} -53d39m18.0s"


Reading from file with function *READCOL()*, *READHMS()* and *READDMS()*
..........................................................................

Coordinates can also be read from file. Either we want to be able to read just
one column, or we want to combine three columns into one if the columns represent
hours/degrees, minutes and seconds.
The arguments of these functions are derived from those in
:func:`tabarray.readColumns` with the exception that
argument *cols=* is replaced by *col=* for function *READCOL()* and
for *READHMS()* and *READDMS()* the *cols=* argument is replaced by
arguments *col1=, col2=, col3=*.
Filenames can be specified either with or without double quotes.

Columns start with 0.
Lines with 1.

Some examples:

Assume we have a file on disk called 'lasfootprint' which stores two sets
of positions separated by an empty line. If we want to read
the positions given by column 0 and 1 of the second segment (starting with line 66)
and column 1 is given in decimal hours, then you need the command:
   
>>> pos=  'readcol("lasfootprint", 0,66,0) HMShour readcol("lasfootprint", 1,66,0) deg'

The coordinate is followed by a unit (deg) so it is a world coordinate.
It was also possible to prepend the second coordinate with {} as in:

>>> pos=  'readcol("lasfootprint", 0,1,64) HMShour {} readcol("lasfootprint", 1,0,64)'

If columns 2 and 3 are galactic longitudes and latitudes, but
our basemap is equatorial, then we could have read the positions
with an alternative sky system as in:

>>> pos=  '{ga} readcol("lasfootprint", 2,1,64)  {} readcol("lasfootprint", 3,0,64)'

The second sky definition is empty which implies a copy of the first
definition (i.e. {ga}).

Now assume you have an ASCII file on disk with 6 columns representing
sexagesimal coordinates. Assume file *hmsdms.txt* contains equatorial
coordinates in 'hours minutes seconds degrees minutes seconds' format,
then read this data with:

>>> pos= '{} readhms(hmsdms.txt,0,1,2) {} readdms(hmsdms.txt,3,4,5)'

Or with explicit lines:

>>> pos= '{} readhms(hmsdms.txt,0,1,2,1,64) {} readdms(hmsdms.txt,3,4,5,1,64)'

What if the format is 'd m s d m s' and the coordinates are galactic.
Then we should enter;
   
>>> pos= 'ga readdms(hmsdms.txt,0,1,2) ga readdms(hmsdms.txt,3,4,5)'

if your current sky system is galactic then it also possible to enter:

>>> pos= 'readdms(hmsdms.txt,0,1,2) deg  readdms(hmsdms.txt,3,4,5) deg'

If the columns are not in the required order use the keyword names:

>>> pos= 'readdms(hmsdms.txt,col3=0,col2=1,col3=2) deg  readdms(hmsdms.txt,3,4,5) deg'

**syntax**

>>> readcol(filename, col=0, fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=0.0,
            rowslice=(None, ), colslice=(None, )

>>> readhms(filename, col1=0, col2=1, col3=2,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=0.0,
            rowslice=(None, ), colslice=(None, )):

Which has the same syntax as *READDMS()*.

   

    * filename - a string with the name of a text file containing the table.
    * fromline - Start line to be read from file (first is 1).
    * toline - Last line to be read from file. If not specified, the end of the file is assumed.
    * comment - a string with characters which are used to designate comments in the input file. The occurrence of any of these characters on a line causes the rest of the line to be ignored. Empty lines and lines containing only a comment are also ignored.
    * col - a scalar with one column number.
    * sepchar - a string containing the column separation characters to be used. Columns are separated by any combination of these characters.
    * rows - a tuple or list containing the row numbers to be extracted.
    * bad - a number to be substituted for any field which cannot be decoded as a number.
    * rowslice - a tuple containing a Python slice indicating which rows should be selected. If this argument is used in combination with the argument rows, the latter should be expressed in terms of the new row numbers after slicing. Example: rowslice=(10, None) selects all rows, beginning with the eleventh (the first row has number 0) and rowslice=(10, 13) selects row numbers 10, 11 and 12.
    * colslice - a tuple containing a Python slice indicating which columns should be selected. If this argument is used in combination with the argument cols, the latter should be expressed in terms of the new column numbers after slicing. Selection is analogous to rowslice.


There is a difference between the *rows=* and the *fromline=* , *endline=*
keywords. The first reads the specified rows from parsed contents of the
file, while the line keywords specify which lines you want to read from file.
Usually comment characters '#' and '!' are used. If you expect another comment
character then change this keyword.
Keyword *sepchar=* sets the separation character and *bad=* is the value
that is substituted for values that could not be parsed.


You can run 'main' (i.e. execute the module) to experiment with positions.
First it will display
a couple of examples before it prompts for user input. Then your are prompted
to enter a string (no need to enclose it with quotes).
 
>>> python position.py

We assumed you copied the module from the site packages directory to you current
working directory.

Note that if the module is used in the GIPSY context then you can use
GIPSY's functions and constants. Python parsing then, is done with the *eval()*
command. For reading data from files you can either use GIPSY's *file()* command
or command *readcol()*.

Using Python's evaluation function with *EVAL()*
.................................................

One can always force the parser to use Python's expression evaluation.
We defined function *eval()* for this. The argument is one expression
or a sequence of expressions separated by a comma. This allows you to
use spaces in an expression because a space is not a separator symbol
in the context of *EVAL()*.
For example:
      
>>> pos="eval(atan2(3, 2) , 11 /2) eval([10,20])"

works while

>>> pos="atan2(3, 2),  11 /2 10,20"

does not. Also one should note that the *EVAL()* function groups data.
So in ``eval(atan2(3, 2) , 11 /2)`` the two values represent the X-coordinate
and not a position!


Reading header items with function *HEADER()*
..............................................

Command *header* reads an item from the header that was used to create the Projection
object. The header item must represent a number.

>>> pos= "header('crpix1') header('crpix2')"


Structure of output
....................

In a previous example we processed the output as follows::
      
   w,p,u,e = positions.str2pos('5, 6', pr)
   if e == '':
      print "pixels:", p
      print "world coordinates:", w, u

The function :func:`str2pos` returns a tuple with four items:

      * *w*: an array with positions in world coordinates
      * *p*: an array with positions in pixel coordinates
      * *u*: an array with the units of the world coordinates
        These units are derived from the projection object with
        an optional alternative sky system and/or an optional
        spectral translation.
      * *e*: an error message. If the length of this string is not 0, then
        it represents an error message and the arrays *w* and *p* are empty.

Errors:
........

The position parser is flexible but there are some rules. If the input
cannot be transformed into coordinates then an appropriate message will be
returned. In some cases the error message seems not related to the problem
but that seems inherent to parsers. If a number is wrong, the parser tries
to parse it as a sky system or a unit. If it fails, it will complain about
the sky system or the unit and not about the number.

Functions
---------

.. autofunction:: str2pos
.. autofunction:: parsehmsdms

"""

# Imports
from __future__ import division
from math import *                               # For Pythons 'eval()' function
from types import ListType, TupleType
from re import split as re_split
from re import findall as re_findall
from string import whitespace, ascii_uppercase, join
from string import upper as string_upper
from types import StringType
from numpy import nan as unknown
from numpy import asarray, zeros, floor, array2string
from numpy import ndarray, sign
from kapteyn import wcs                          # The WCSLIB binding
from kapteyn.celestial import skyparser
from kapteyn.tabarray import readColumns
try:
   from gipsy import evalexpr, GipsyError
   gipsymod = True
except:
   gipsymod = False

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


def readcol(filename, col=0, fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=0.0,
            rowslice=(None, ), colslice=(None, )):
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
   column = [column]
   if rows != None:
      if not issequence(rows):
         rows = [rows]
   lines = None
   if not fromline is None or not toline is None:
      lines = (fromline, toline)
   c = readColumns(filename=filename, comment=comment, cols=column, sepchar=sepchar,
               rows=rows, lines=lines, bad=bad, rowslice=rowslice, colslice=colslice)
   return list(c.flatten())


def readhms(filename, col1=0, col2=1, col3=2,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=0.0,
            rowslice=(None, ), colslice=(None, )):
   #-------------------------------------------------------------
   """
   Utility to prepare a call to tabarray's readColumns() function
   We created a default for the 'comment' argument and changed the
   column argument to accept only one column.
   """
   #-------------------------------------------------------------
   column = [col1, col2, col3]
   if rows != None:
      if not issequence(rows):
         rows = [rows]
   lines = None
   if not fromline is None or not toline is None:
      lines = (fromline, toline)
   c = readColumns(filename=filename, comment=comment, cols=column, sepchar=sepchar,
               rows=rows, lines=lines, bad=bad, rowslice=rowslice, colslice=colslice)
   h = c[0]; m = c[1]; s = c[2]
   vals = (h+m/60.0+s/3600.0)*15.0
   return vals


def readdms(filename, col1=0, col2=1, col3=2,
            fromline=None, toline=None, rows=None, comment="!#",
            sepchar=', t', bad=0.0,
            rowslice=(None, ), colslice=(None, )):
   #-------------------------------------------------------------
   """
   Utility to prepare a call to tabarray's readColumns() function
   We created a default for the 'comment' argument and changed the
   column argument to accept only one column.
   """
   #-------------------------------------------------------------
   column = [col1, col2, col3]
   if rows != None:
      if not issequence(rows):
         rows = [rows]
   lines = None
   if not fromline is None or not toline is None:
      lines = (fromline, toline)
   c = readColumns(filename=filename, comment=comment, cols=column, sepchar=sepchar,
               rows=rows, lines=lines, bad=bad, rowslice=rowslice, colslice=colslice)
   d = c[0]; m = c[1]; s = c[2]
   # Take care of negative declinations
   vals = sign(d)*(abs(d)+abs(m)/60.0+abs(s)/3600.0)
   return vals



def minmatch(userstr, mylist, case=0):
   """--------------------------------------------------------------
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
   ------------------------------------------------------------------"""
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
   Return the factor for a conversion of numbers between two units.
   """
#-----------------------------------------------------------------------------------
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
            'RY' :          (10,                       2.179872e-18)
           }
   mylist = units.keys()
   i = minmatch(unitfrom, mylist)
   errmes = ''
   if i != None:
      if i >= 0:
         key = units.keys()[i]
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
         key = units.keys()[i]
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
   except ValueError, message:
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
      >>> hmsdms = '20h34m60-7.7003'   # Expression allowed because Pythons eval() is used
      >>> hmsdms = '-51.28208458d0m'   # Negative value for latitude

      * The 's' for seconds is optional
      * Expressions in numbers are allowed
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

   Notes:         Parenthesis are used for functions e.g. FILE()
                  Curly brackets are used to identify sky definitions
                  Square brackets allow the use of GIPSY lists e.g. [90:70:-10]
                  Quotes group characters into one token
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
      if ch in ws and not (sqbdepth or brackdepth or pardepth or quote) :
         if tokens[-1] != '' : tokens.append('')
      else :
         tokens[-1] += ch

   return tokens



class Coordparser(object):
   """--------------------------------------------------------------------
   Purpose: Parse a string wich represents position(s). Return an object
   with the sorted input grids and world coordinates.
   
   First a pre parser finds the tokens in the string. Tokens are
   separated by a comma or whitespace.
   A group of characters enclosed by single or double quotes form one token.
   This enables a user to group numbers (with a sky system, a spectral
   translation and/or a unit)
   Characters enclosed by (), {} or [] are transferred unparsed. This allows
   a user to enter:
   1) parameters for GIPSY functions, e.g. POS=atan2(x,y)
   2) group parameters of a sky system, e.g. POS={eq, J1983.5}
   3) GIPSY lists, e.g. POS=[0::2 3:5]
   4) expressions for Python's eval() parser

   strings between single quotes are parsed unalterd. Except the quotes themselves.
   They are removed.
   Strings between double quotes are parsed unalterd. This includes the double quotes.
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
                naxis, translations, source, maxpos=100000, usegipsy=False):
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
          maxpos- A maximum for the total number of coordinates that can be
                  entered in one number specification for GIPSY's
                  expression evaluation. By default the maximum is set to
                  100000. If this method is used in the GIPSY environment
                  then this parameter could be important if you expect
                  many coordinates.

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
      if maxpos < 1:
         maxpos = 1
      self.maxpos = maxpos
      self.usegipsy = usegipsy and gipsymod      # User wants GIPSY and module is available
      if translations:
         self.strans, self.sunits = zip(*translations)
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
      # Examples for GIPSY (For other examples see test in main)
      # POS=file(asciipos.txt,1,1:4) 0::3 0 0     # 3 numbers (one comment) from file, three times grid 0
      #                                           # followed by two grids 0
      # POS='0 1 4'  '242 243 244' km/s           # Grouping of 3 grids and 3 world coordinates with unit
      # POS= 0::2 '-243 244' km/s                 # Equivalent to next statement
      # POS= 0 -243 km/s 0 -244 km/s
      #-------------------------------------------------------------------
      tryother = False
      currenttoken = self.tokens[tpos]
      number = None
      if currenttoken.startswith('{'):   # Fast way out. Cannot be a number
         return None, '', 0

      # First parse special functions
      readcolfie = "READCOL"
      headfie = "HEADER"
      # Column from file
      doreadcol = currenttoken.upper().startswith("READCOL")
      doreadhms = currenttoken.upper().startswith("READHMS")
      doreaddms = currenttoken.upper().startswith("READDMS")
      if doreadcol or doreadhms or doreaddms:
         # Then evaluate the local version readcol() which forces
         # reading only one column and returns a list instead of
         # a numpy array
         try:
            # We want to allow a user to enter the file name argument
            # without quotes as in:
            # readcol(lasfootprint.txt, 0,1,64)
            # But the function itself needs it as a string
            argstr = currenttoken[len(readcolfie)+1:-1]
            if argstr.count('"') == 0:
               args = argstr.split(',', 1) # Split only until first comma
               argstr = '"'+args[0]+'",'+args[1]
            #pstr = "readcol(%s)"%currenttoken[len(readcolfie)+1:-1]
            if doreadcol:
               pstr = "readcol(%s)"%argstr
            elif doreadhms:
               pstr = "readhms(%s)"%argstr
            elif doreaddms:
               pstr = "readdms(%s)"%argstr
            number = eval(pstr)
         except Exception, message:
            self.errmes = usermessage(currenttoken, message)
            return None, 'x', 0                  # 'x' means hopeless
      # Number from header item
      elif currenttoken.upper().startswith(headfie):
         try:
            t = currenttoken[len(headfie)+1:-1].upper()
            # A string between double quotes is copied with quotes included
            # Then remove these double quotes
            if t.startswith('"'):
               t = t[1:-1]
            # We expect only one floating point number. Make list of result
            number = [float(self.source[t])]
         except Exception, message:
            self.errmes = usermessage(currenttoken, message)
            return None, 'x', 0
      # Is it a string to be parsed with Pythons eval() function?
      elif currenttoken.upper().startswith('EVAL'):
         # A token that starts with eval is processed by Pythons 'eval()' function
         # It results in one or more numbers. The mathematical routines are those
         # available in module math.h
         # Examples:
         # Use Python's eval() function and list comprehension to generate
         # 10 numbers: POS=EVAL([sin(x) for x in range(10)])
         # The power of this exceeds that of the GIPSY parsing based on
         # herinp.c. The reason that we allow numbers to be entered using
         # GIPSY syntax is that we require downwards compatibility, i.e. if
         # the GIPSY flag is set in the constructor.
         try:
            x = eval(currenttoken[5:-1])      # Remove eval and parentheses
            if type(x) is TupleType:
               x = list(x)
            if type(x) is not ListType:       # These two types cannot be combined. x = list(x) will raise except.
               x = [x]
            number = x
         except Exception, message:
            self.errmes = usermessage(currenttoken, message)
            # Obviously a mistake in an eval expression. This cannot
            # imply something else. So return.
            return None, 'x', 0
      else:
         # Not one of the known functions
         # If the context is GIPSY (Groningen Image Processing SYstem), then
         # numbers are evaluated by GIPSY's parser. Python evaluations are
         # parsed with EVAL()
         if self.usegipsy:
            try:
               number = evalexpr(currenttoken, self.maxpos)
               if self.maxpos == 1:
                  number = [number]
            except GipsyError, message:
               self.errmes = usermessage(currenttoken, message)
               tryother = True
         else:
            # Try it as argument for Python's eval()
            try:
               x = eval(currenttoken)
               if type(x) is TupleType:
                  x = list(x)
               if type(x) is not ListType:       # These two types cannot be combined. x = list(x) will raise except.
                  x = [x]
               number = x
            except Exception, message:
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
            candidate = (m_ind >= 0 and m_ind > h_ind and m_ind > d_ind)
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
            # environment (GIPSY) expects the input was a grid. In the
            # GIPSY routine where the coordinate transformation takes place
            # the grids are transformed to pixels for WCSLIB
            pc = self.crpix[coordindx]
            if self.usegipsy:
               # Go from FITS pixel to grid
               pc -= nint(self.crpix[coordindx])
            return [pc], 'g', 1
         elif currenttoken.upper() == 'AC':
            # Next code is compatible to code in cotrans.c only we made the expression
            # simpler by rewriting the formula so that cotrans' offset is not necessary.
            n = self.naxis[coordindx]
            ac = 0.5 * (n+1)
            if self.usegipsy:
               # Go from FITS pixel to grid. See also comment at 'PC'
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
            siunit = self.siunits[coordindx]
         unitfact = None
         unitfact, message = unitfactor(nexttoken, siunit)
         if unitfact == None:
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
            skydef = sk         # Copy PARSED sky definition!
      except Exception, message:
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


def dotrans(parsedpositions, subproj, subdim, mixpix=None, gipsy=False):
   #-------------------------------------------------------------------
   """
   """
   #-------------------------------------------------------------------
   skyout_orig = subproj.skyout            # Store and restore before return
   errmes = ''                             # Init error message to no error 
   r_world = []
   r_pixels = []
   subsetunits = None
   if gipsy:
      # First we determine the -integer- offsets to transform grids
      # into 1-based FITS pixels
      offset = [0.0]*subdim
      for i in range(subdim):
          offset[i] = nint(subproj.crpix[i])
          
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
         numbers = asarray(p[i][0])        # Contents of coordinate number 'i' (can be a list with numbers)
         # Numbers here is always a LIST with 1 or more numbers. Make a NumPy
         # array of this list to facilitate grid to pixel conversions
         if numbers.shape == ():
            N = 1
         else:
            N = numbers.shape[0]
         if p[i][1] == 'g':
            # Convert from grid to pixel
            gcoord[i] = numbers
            if gipsy:
               gcoord[i] += offset[i]
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
      except wcs.WCSerror, message:
         errmes = str(message[1])  # element 0 is error number
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


def str2pos(postxt, subproj, mixpix=None):
   #-------------------------------------------------------------------
   """
   This method accepts a string that represents a position in the
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
   :type mixpix:    Integer

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
   if type(postxt) != StringType:
      raise TypeError, "str2pos(): parameter postxt must be a String"
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
                                 subproj.source) # Get access to header

   if parsedpositions.errmes:
      if postxt != '':
         return [], [], [], parsedpositions.errmes
   else:
      wor, pix, subsetunits, errmes = dotrans(parsedpositions.positions,
                                              subproj,
                                              subdim,
                                              mixpix,
                                              gipsy=False)
      if errmes != '':
         return [], [], [], errmes

   return wor, pix, subsetunits, ''


def dotest():
   def printpos(postxt, pos):
      # Print the position information
      world, pixels, units, errmes = pos
      print    "Expression        : %s"%postxt
      if errmes == '':
         print "World coordinates :", world, units
         print "Pixel coordinates :", pixels
      else:
         print errmes
      print ""
      
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
              'CRVAL3' : 1.415418199417E+09,
              'CTYPE1' : 'RA---NCP',
              'CTYPE2' : 'DEC--NCP',
              #'CTYPE3' : 'FREQ-OHEL',
              'CTYPE3' : 'FREQ',
              'CUNIT1' : 'DEGREE',
              'CUNIT2' : 'DEGREE',
              'CUNIT3' : 'HZ',
              'DRVAL3' : 1.050000000000E+03,
              'DUNIT3' : 'KM/S',
              'FREQ0'  : 1415418199.417,
              'INSTRUME' : 'WSRT',
              'NAXIS1' : 10,
              'NAXIS2' : 10,
              'NAXIS3' : 10
            }

   origproj = wcs.Projection(header)

   print "-------------- Examples of numbers and constants, missing spatial--------------\n"
   proj = origproj.sub((1,3,2))
   mixpix = 6
   userpos = ["(3*4)-5 1/5*(7-2)",
              "abs(-10), sqrt(3)",
              "sin(radians(30)), degrees(asin(0.5))",
              "cos(radians(60)), degrees(acos(0.5))",
              "pi, tan(radians(45))-0.5, 3*4,0",
              "[sin(x) for x in range(10)], range(10)",
              "atan2(2,3), 192",
              "atan2(2,3)  192",
              "[pi]*3, [e**2]*3",
              "eval(atan2(3,2),11/2) eval([10,20])",
              "c_/299792458.0,  G_/6.67428e-11"
             ]
   for postxt in userpos:
      wp = str2pos(postxt, proj, mixpix=mixpix)
      printpos(postxt, wp)
   print ''

   print "-------------- Examples of units, spectral translations and grouping -----------\n"
   proj = origproj.sub((3,))
   userpos = ["7",
              "1.415418199417E+09 hz",
              "1.415418199417E+03 Mhz",
              "1.415418199417 Ghz",
              "vopt 1.05000000e+06",
              "vopt 1050 km/s",
              "vopt 0",
              "vrad 1.05000000e+06",
              "FREQ 1.415418199417E+09",
              "0 7 10 20",
              "'1.41, 1.415418199417, 1.42, 1.43' Ghz",
              "[1.41, 1.415418199417, 1.42, 1.43] Ghz"
              ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''

   print "--------- Output of previous coordinates in terms of VOPT:----------\n"
   userpos = ["7",
              "freq 1.415418199417E+09 hz",
              "fr 1.415418199417E+03 Mhz",
              "fr 1.415418199417 Ghz",
              "vopt 1.05000000e+06",
              "vopt 1050 km/s",
              "vopt 0",
              "vrad 1.05000000e+06",
              "FREQ 1.415418199417E+09",
              "0 7 10 20 70.233164383215",
              "FREQ '1.41, 1.415418199417, 1.42, 1.43' Ghz",
              "FR [1.41, 1.415418199417, 1.42, 1.43] Ghz"
              ]
   proj2 = proj.spectra('VOPT-???')
   for postxt in userpos:
      wp = str2pos(postxt, proj2)
      printpos(postxt, wp)
   print ''

   print "--------- Sky systems and AC&PC ----------\n"
   proj = origproj.sub((1,2))
   userpos = ["0 0",
              "5,6 0 0 3,1",
              "eq 178.7792  eq 53.655",          # e 10 will not work because e is not a symbol and an ambiguous sky system`
              "{eq} 178.7792  {} 53.655",
              "178.7792 deg  53.655 deg",
              "11h55m07.008s 53d39m18.0s",
              "{eq, B1950,fk4} 178.7792  {} 53.655",
              "{eq, B1950,fk4} 178.12830409  {} 53.93322241",
              "{fk4} 178.12830409  {} 53.93322241",
              "{B1983.5} 11h55m07.008s {} 53d39m18.0s",
              "{eq, B1950,fk4, J1983.5} 178.12830409  {} 53.93322241",
              "ga 140.52382927 ga  61.50745891",
              "su 61.4767412, su 4.0520188",
              "ec 150.73844942 ec 47.22071243",
              "{} 178.7792 6.0",
              "5.0, {} 53.655",
              "{eq} '178.779200, 178.778200, 178.777200' {} '53.655000, 53.656000, 53.657000'",
              "PC",
              "ac"]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''

   print "--------- Same as previous but in terms of Galactic coordinates ----------\n"
   sky_old = proj.skyout
   proj.skyout = "ga"
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''
   proj.skyout = sky_old

   print "--------- XV map: one spatial and one spectral axis ----------\n"
   proj = origproj.sub((2,3,1))
   mixpix = 5
   print "Spectral translations: ", proj.altspec
   userpos = ["{} 53.655 1.415418199417E+09 hz",
              "{} 53.655 1.415418199417E+03 Mhz",
              "53.655 deg 1.415418199417 Ghz",
              "{} 53.655 vopt 1.05000000e+06",
              "{} 53.655 , vopt 1050000 m/s",
              "0.0 , vopt 1050000 m/s",
              "10.0 , vopt 1050000 m/s",
              "{} 53.655 vrad 1.05000000e+06",
              "{} 53.655 FREQ 1.41541819941e+09",
              "{} 53.655 wave 21.2 cm",
              "{} 53.655 vopt c_/300 m/s"]
   for postxt in userpos:
      wp = str2pos(postxt, proj, mixpix=mixpix)
      printpos(postxt, wp)
   print ''

   # Create an Ascii file with dummy data for testing the READCOL command
   # The data in the Ascii file is composed of a fixed sequence of grids
   # that are transformed to their corresponding galactic coordinates. 
   asciifile = "test123positions.txt"
   f = open(asciifile, 'w')
   s = "! Test file for Ascii data and the FILE command\n"
   f.write(s)
   for i in range(10):
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

   print "--------- Reading from file ----------\n"
   proj = origproj.sub((1,2))
   userpos = [ 'readcol(test123positions.txt, col=0) readcol(test123positions.txt, col=2)',
               '{} readcol(test123positions.txt, col=0) {} readcol(test123positions.txt, col=2)',
               'ga readcol(test123positions.txt, col=0) ga readcol(test123positions.txt, col=2)',
               'readcol(test123positions.txt, col=0) deg readcol(test123positions.txt, col=2) deg',
               '{} readhms(hmsdms.txt,0,1,2) {} readdms(hmsdms.txt,3,4,5)',
               '{} readhms(hmsdms.txt,col1=0, col3=1, col2=2) {} readdms(hmsdms.txt,3,4,5)',
             ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''

   print "--------- Reading from header ----------\n"
   proj = origproj.sub((1,2))
   userpos = [ "{} header('crval1') {} header('crval2')",
               "header('crpix1') header('crpix2')" ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''

   print "--------- Problem strings and error messages ----------\n"
   proj = origproj.sub((1,2))
   userpos = ["33",
              "1 2 3",
              "eq 178, ga 40",
              "22 {}",
              "10, 53 heg",
              'readcol("test123wcsRADECFREQ.txt, 0) readcol("test123wcsRADECFREQ.txt", 2)',
              'readcol("test123wcsRADECFREQ.txt", 0, range(1:4)) 3:5',
              'readcol("test123wcsRADECFREQ.txt", 2, rows=[0,1,2,3])',
              'readcol("test123wcsRADECFREQ.txt", 0, rowsslice(5,None)) readcol("test123wcsRADECFREQ.txt", 2, rowslice=(5,None))',
              'readcol("test123wcsRADECFREQ.txt", 0, row=2) readcol("test123wcsRADECFREQ.txt", 2, row=2)',
              '{ga} readcol("test123wcsRADECFREQ.txt", 1) {} readcol("test123wcsRADECFREQ".txt, 3)',
              '{ga} readcol("test123positions.txt", col=0) {} readcol("test123positions.txt", col=2)'
            ]
   for postxt in userpos:
      wp = str2pos(postxt, proj)
      printpos(postxt, wp)
   print ''


   import readline
   upos = 'xxx'
   proj = origproj.sub((1,2)); mixpix = None
   while upos != '':
      upos = raw_input("Enter position(s) ..... [quit loop]: ")
      readline.add_history(upos) 
      if upos != '':
         wp = str2pos(upos, proj, mixpix=mixpix)
         printpos(upos, wp)


if __name__ == "__main__":
      dotest() 