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

This module enables a user/programmer to specify positions in either
pixel- or world coordinates. Its functionality is provided by a parser
which converts strings with position information into pixel coordinates
and world coordinates. It recognizes alternative sky- and reference systems,
epochs and spectral systems.
It does a minimal match on units and it can convert between compatible units.

Introduction
------------

Physical quantities in a datastructure that represents a measurement are usually
measurements at fixed positions in the sky or at spectral positions such as
Doppler shifts, frequencies or velocities. Both positions are examples of so called
**World Coordinates**. In these data structures the quantities are identified by their
pixel coordinates. Following the rules for FITS files, a pixel coordinate starts with
1 and runs to *NAXISn* which is a header item that sets the length of the n-th axis
in the structure.

Assume you have a data structure representing an optical image of a part of the sky
you need to mark a certain feature in the image or need to retrieve the intensity
of a certain pixel. Then usually it is easy to identify the pixel using
pixel coordinates. But sometimes you have positions (e.g. from external sources like
catalogs) given in world coordinates and then it would be convenient if you could specify
positions in those coordinates. Module :mod:`wcs` wcs provides methods for conversions between
pixel coordinates and world coordinates given a description of the world coordinate
system (as defined in a header). Module :mod:`celestial` converts world coordinates
between different sky- and reference systems and/or epochs.
Note that a description of a world coordinate system can be either a FITS header or
a Python dictionary with FITS keywords.

In this module we combined the functionality of :mod:`wcs` and :mod:`celestial`
to write a position parser.

How to use this module
----------------------

This module is included in other modules of the Kapteyn Package, but
it can be imported in your own scripts also so that you are able to convert
positions in a string to pixel- and world coordinates.
It is also possible to use it as a test application (*python positions.py*) where you
can add your own strings for conversion.

At the end of the source code of this module you find a practical example how to use
the conversion method :meth:`str2pos`.

Position syntax
---------------

Pixel coordinates
.................

All numbers which are not recognized as world coordinates are returned as pixel
coordinates. The first pixel has coordinate 1. Header value *CRPIX* sets the
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
>>> pos = "[pi]\*3, [e**2]*3"

Constants
..........

A number of global constants are defined and these can be used in the
expressions for positions:

These constants are::

      c_ = 299792458.0             # Speed of light in m/s
      h_ = 6.62606896e-34          # Planck constant in J.s
      k_ = 1.3806504e-23           # Boltzmann in J.K^-1
      G_ = 6.67428e-11             # Gravitation in m^3. kg^-1.s^-2
      s_ = 5.6704e-8               # Stefan- Boltzmann in J.s^-1.m^-2.K^-4
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
     units of the world coordinate is given in the (FITS) header in keyword *CUNIT*.
   * a coordinate prepended by a definition for a sky system or a spectral system.
   * a coordinate entered in sexagisimal notation. (hms/dms)

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
A sky definition is either a minimal match from the list::

  'EQUATORIAL','ECLIPTIC','GALACTIC','SUPERGALACTIC'

or it is a definition between curly brackets. The definition can contain
the sky system, the reference system, Equinox and epoch of observation.
The documentation for sky definitions is found in module :mod:`celestial`.

Examples:

   >>> pos = "{eq} 178.7792  {} 53.655"                      # As a sky definition between curly brackets
   >>> pos = "{} 178.7792 {} 53.655"                         # A world coordinate in the native sky system      
   >>> pos = "{eq,B1950,fk4} 178.12830409  {} 53.93322241"   # With sky system, reference system and equinox
   >>> pos = "{fk4} 178.12830409  {} 53.93322241"            # With reference system only.
   >>> pos = "{eq, B1950,fk4, J1983.5} 178.1283  {} 53.933"  # With epoch of observation (FK4 only)
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
in a compatible spectral system. 

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

   For positions in data structure with one spatial axis, the other
   (missing) spatial axis is identified by a pixel coordinate.
   This restricts the spatial world coordinates to their native wcs.
   We define a world coordinate in its native sky system
   with *{}* 


**Sexagisimal notation**

Read the documentation at :func:`parsehmsdms` for the details.
Here are some examples:

   >>> pos = "11h55m07.008s 53d39m18.0s"
   >>> pos = "{B1983.5} 11h55m07.008s {} -53d39m18.0s"

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
from re import compile as re_compile
from re import VERBOSE, IGNORECASE
from string import whitespace, ascii_uppercase, join
from string import upper as string_upper
from numpy import nan as unknown
from numpy import asarray, zeros, floor, array2string
from kapteyn import wcs                          # The WCSLIB binding
from kapteyn.celestial import skyparser 



# Define constants for use in eval()
c_ = 299792458.0    # Speed of light in m/s
h_ = 6.62606896e-34 # Planck constant in J.s
k_ = 1.3806504e-23  # Boltzmann in J.K^-1
G_ = 6.67428e-11    # Gravitation in m^3. kg^-1.s^-2
s_ = 5.6704e-8      # Stefan- Boltzmann in J.s^-1.m^-2.K^-4
M_ = 1.9891e+30     # Mass of Sun in kg
P_ = 3.08567758066631e+16 # Parsec in m


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
   as if it was a spatial world coordinate eiher in
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
      *None* and an error message the the parsing failed.

   :Notes:

      A distinction has been made between longitude axes and
      latitude axes. The hms format can only be used on longitude
      axes. However there is no restriction on the sky system.
      The input is flexible (see examples) but

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
         number = eval(p)                        # Everything that Python can parse in a number
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
               return None, "Invalid syntax for sexagisimal numbers"
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
         ch = '' # Copy quotes or not
      if ch in ws and not (sqbdepth or brackdepth or pardepth or quote) :
         if tokens[-1] != '' : tokens.append('')
      else :
         tokens[-1] += ch

   return tokens



class Coordparser(object):
   """--------------------------------------------------------------------
   Start parsing string with position(s)
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

   Description of the token parser:

   token END:    "#"
   token FILE"   "A file on disk"
   token NUM:    "is a plain number"
   token SNUM    "is a sexagisimal number
   token UNIT:   "is a unit"
   token WORLD   "NUM followed by UNIT"
   token SKY     "One of EQ, EC, GA, SG or [SKY,parameters]"
   token SPECTR  "A compatible spectral translation"

   goal:                positions END
   positions:           N (coordinates)*3 or datafromfile
   coordinate:          a grid or a world coordinate or sequence from file
   grid:                NUM: valid result of evalexpr() or result of Pythons eval() function
   unit:                UNIT: valid result of unitfactor
   world:               SNUM or NUM UNIT or sky or spectral
   sky:                 SKY world or SKY NUM
   spectral:            SPECTR world or SPECTR NUM
   ---------------------------------------------------------------------"""
   def __init__(self, tokenstr, ncoords, siunits, types, crpix, naxis, translations, maxpos=100000):
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
                  entered in one number specification.

      Returns:    It instantiates an object from class 'Coordparser'. The 
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
      if maxpos < 1:
         maxpos = 1
      self.maxpos = maxpos
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
      # We need ncoords coordinates to get one position
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
            self.errmes = "Different number elements in group [%d, %d]" % (numval, lval)
            return None , tpos
         tpos += tposdelta
         numcoords += 1
         p.append((val, typ, sky, spec))
      if numcoords != self.ncoords:
         self.errmes = "Not enough coordinates for a position"
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
            self.errmes = "Not a grid nor world coord. sky or spectral parameter"
      return None, '', '', '', 0
      


   def getnumber(self, tpos, coordindx, unit=None):
      #-------------------------------------------------------------------
      # Allow a different unit if the unit is changed by a spectral translation
      #
      # POS=file(asciipos.txt,1,1:4) 0::3 0 0     # 3 numbers (one comment) from file, three times grid 0
      #                                             # followed by two grids 0
      # POS='0 1 4'  '242 243 244' km/s             # Grouping of 3 grids and 3 world coordinates with unit
      # POS= 0::2 '-243 244' km/s                   # Equivalent to nexttoken
      # POS= 0 -243 km/s 0 -244 km/s
      #-------------------------------------------------------------------
      currenttoken = self.tokens[tpos]
      number = None
      if currenttoken.startswith('{'):
         return None, '', 0

      try:
         # Python's eval() can be abused to import and execute potentially dangerous
         # functions. If needed we can restrict its namespace.
         # Function eval() can also process a number of strings. The strings
         # need to be separated with a comma. So first we have to split the
         # strings at comma's and whitespace. The we join the
         # splitted strings with only a comma. The effect is that
         # '2  3 atan2(3,  4)' is cleaned up and parsed to: '2,3,atan2(3,4)'
         # which will be evaluated as [2, 3, 0.64350110879328437]
         #tokens = re_split('[,\s]+', currenttoken) 
         #comsep = join(tokens, ',')
         #print "comsep=", comsep
         #x = eval(comsep)
         x = eval(currenttoken)
         if type(x) is TupleType:
            x = list(x)
         if type(x) is not ListType:       # These two types cannot be combined. x = list(x) will raise except.
            x = [x]
         number = x
      except:
         # Not a number or numbers from a file. Perhaps a sexagisimal number
         # candidate = re_findall('[hmsHMSdD]', currenttoken)
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
               self.errmes = "(%s) %s" % (currenttoken, errmes)
               return None, '', 0
         elif currenttoken.upper() == 'PC':
            # 'PC' represents the projection center for spatial axes
            # but more general, it is the position of the reference pixel.
            # In GIPSY grids, this position is located somewhere in grid 0.
            # GIPSY: pc = self.crpix[coordindx] - nint(self.crpix[coordindx])
            pc = self.crpix[coordindx]
            return [pc], 'g', 1
         elif currenttoken.upper() == 'AC':
            # Next code is compatible to code in cotrans.c only we made the expression
            # simpler by rewriting the formula so that cotrans' offset is not necessary.
            n = self.naxis[coordindx]
            # GIPSY:
              #shift = nint(self.crpix[coordindx])
              #ac = 0.5 * (n+1) - shift
            ac = 0.5 * (n+1)
            return [ac], 'g', 1

         else:
            # No number nor a sexagisimal number
            return None, '', 0

      if number == None:
         return None, '', 0
      
      # A number is parsed. The number could be modified if a unit follows
      nexttoken = self.tokens[tpos+1]
      if nexttoken != self.END:
         if unit != None:
            siunit = unit
         else:
            siunit = self.siunits[coordindx]
         unitfact = None
         unitfact, message = unitfactor(nexttoken, siunit)
         if unitfact == None:
             self.errmes = message
             
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
      # POS=ga 140d30m ga 62d10m    ; Use sexagisimal numbers
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
      skylist = ['EQUATORIAL','ECLIPTIC','GALACTIC','SUPERGALACTIC']
      currenttoken = self.tokens[tpos]
      skydef = currenttoken.startswith('{')      # Start of a sky definition
      
      
      indx = -1
      if not skydef:
         if len(currenttoken) > 1:               # If its only one character then probably it is a constant
            indx = minmatch(currenttoken, skylist, 0)
      
      if indx >= 0 or skydef:                    # Found either a sky system or sky definition ('[')
         if skydef:
            skydef = currenttoken[1:-1]          # Get rid of braces {} (curly brackets)
            if skydef != '':
               dumskydef, errmes = self.parsesky(skydef)
               if skydef == None:
                  self.errmes = errmes
                  return None, '', 0
         else:
            skydef = indx
         nexttoken = self.tokens[tpos+1]
         if nexttoken != self.END:
            number, typ, tposdelta = self.getnumber(tpos+1, coordindx)
            if number != None:
               return number, skydef, tposdelta+1
            else:
               # No number no world coordinate
               self.errmes = "(%s) is a sky system but not followed by grid or world coord." % currenttoken
               return None, '', 0
         else:
            # A sky but nothing to parse after this token
            self.errmes = "(%s) is a sky system but not followed by grid or world coord." % currenttoken
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
      # POS= 0 beta -243000/c      beta = v/c
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
               self.errmes = "(%s) is a spectral trans. but without grid or world c." %  currenttoken
               return None, '', 0
         else:
            # A spectral translation but nothing to parse after this token
            self.errmes = "(%s) is a spectral trans. but without grid or world c." %  currenttoken
            return None, '', 0
      else:
         # Not a spectral translation:
         return None, '', 0



   def parsesky(self, skydef):
      #-------------------------------------------------------------------
      # This is the parser for a sky definition. That is a specification
      # between curly brackets.
      #-------------------------------------------------------------------
      sky, errmes = parseskysystem(skydef)
      if sky == None:
         return None, errmes
      if len(sky) == 0:
         # Try to copy from a previous sky.
         if self.prevsky != None:
            sky = self.prevsky
         if len(sky) == 0:
            #errmes = "Nothing in sky definition"
            #return None, errmes
            sky = ''

      self.prevsky = sky                         # Store this definition for the sky and copy it with {}
      return sky, ''



def str2pos(postxt, subproj, mixpix=None, maxpos=100000):
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
   :param maxpos:   The maximum number of positions that should be
                    returned. The default value is usually sufficient.
   :type maxpos:    Integer

   :Returns:

   This method returns a tuple with four elements:

   * a NumPy array with the parsed world coordinates
   * a NumPy array with the parsed pixel coordinates
   * A tuple with the units that corresponds to the axes
     in your world coordinate system.
   * An error message when a position could not be parsed


   :Examples:
      ::
         
         from kapteyn import wcs, positions

         header = {  'NAXIS'  : 2,
                     'BUNIT'  : 'w.u.',
                     'CDELT1' : -1.200000000000E-03,
                     'CDELT2' : 1.497160000000E-03,
                     'CRPIX1' : 5,
                     'CRPIX2' : 6,
                     'CRVAL1' : 1.787792000000E+02,
                     'CRVAL2' : 5.365500000000E+01,
                     'CTYPE1' : 'RA---NCP',
                     'CTYPE2' : 'DEC--NCP',
                     'CUNIT1' : 'DEGREE',
                     'CUNIT2' : 'DEGREE'}
         
         proj = wcs.Projection(header)
         
         position = []
         position.append("0 0")
         position.append("eq 178.7792  eq 53.655")
         position.append("{eq} 178.7792  {} 53.655")
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
            poswp = positions.str2pos(pos, proj, maxpos=1)
            if poswp[3] != "":
               raise Exception, poswp[3]
            world = poswp[0][0]
            pixel = poswp[1][0]
            units = poswp[2]
         print pos, "=", pixel, '->',  world , units

   """
   #-------------------------------------------------------------------
   subdim = len(subproj.types)
   if mixpix != None:
      subdim -= 1

   parsedpositions = Coordparser(postxt,         # Text containing positions as entered by user
                                 subdim,         # The number of coordinates in 1 position
                                 subproj.units,  # Units (for conversions) in order of subset axes
                                 subproj.types,  # Axis types to distinguish spatial and spectral coords.
                                 subproj.crpix,  # Crpix values for 'PC' (projection center)
                                 subproj.naxis,  # Axis lengths for center 'AC'
                                 subproj.altspec,# List with allowed spectral translations
                                 maxpos)         # A maximum for the number evaluator
   if parsedpositions.errmes:
      valid = False
      if postxt != '':
         return [], [], [], parsedpositions.errmes       
   else:
      r_grids = []
      r_world = []
      r_pixels = []
      subsetunits = None
      # First we determine the -integer- offsets to transform grids
      # into 1-based FITS pixels
      offset = [0.0]*subdim
      for i in range(subdim):
         offset[i] = nint(subproj.crpix[i])
      for p in parsedpositions.positions:
         wcoord = [unknown]*subdim            # A list with tuples with a number and a conversion factor
         gcoord = [unknown]*subdim
         empty  = [unknown]*subdim
         grids  = [unknown]*subdim
         # Reset sky system to original.
         subproj.skyout = None
         # p[i][0]: A list with one or more numbers
         # p[i][1]: the mode ('g'rid or 'w'orld)
         # p[i][2]: the sky definition
         # p[i][3]: the spectral definition
         skyout = ''                          # Each position can have its own sky system
         for i in range(subdim):              # A position has 'subdim' coordinates
            numbers = asarray(p[i][0])        # Contents of coordinate number 'i' (can be a list with numbers)
            # Numbers here is always a list with 1 or more numbers. Make a NumPy
            # array of this list to facilitate grid to pixel conversions
            if numbers.shape == ():
               N = 1
            else:
               N = numbers.shape[0]
            if p[i][1] == 'g':
               # Convert from grid to pixel
               gcoord[i] = numbers      # + asarray(offset[i])
               wcoord[i] = asarray([unknown]*N)
            else:
               gcoord[i] = asarray([unknown]*N)
               wcoord[i] = numbers
            empty[i] = asarray([unknown]*N)
            if not skyout and p[i][2] != '':  # Only the first sky definition counts
               skyout = p[i][2]

         # Add the crpix to convert grid to pixel and repeat for
         # all numbers in one coordinate.
         #if mixax != None:
            #mixpix = self.mixgrid + crpix_mixax
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
         # can be a list or a numpy array.
         #print "wor,pix=", wcoord,gcoord, newproj.types
         try:
            wor, pix = newproj.mixed(tuple(wcoord), tuple(gcoord))
         except Exception, errmes:
            return [], [], [], errmes

         # Now we have the pixels and want the world coordinates in the original
         # system. Then first reset the skyout attribute.
         subproj.skyout = None

         # Then calculate the native world coordinates
         wor = subproj.toworld(tuple(pix))
         # At this moment the status is that one entered a position either in grids,
         # in world coordinates or in a mix of both. This position then is
         # processed to calculate missing grids and missing world coordinates.
         # Then the output is a list with grids and world coordinates in the native
         # projection system. But what is the native projection system?
         # A Set object can have attributes that set the system in a certain output
         # mode, i.e. an output sky system and spectral translation.
         # Assume we have a spectral axis with primary type FREQ and have a
         # velocity from the literature VRAD, but we want to plot a marker for
         # this value in a plot with axis VOPT, then we have to achieve this
         # into two steps (i.e. first with spectral translation VRAD and the second
         # with VOPT).

         """
         if finalsky == None:
            finalsky = subproj.skysys
         outproj = None
         if finalspec != None:
            outproj = subproj.spectra(finalspec)
         if outproj == None:               # We have to create a new one
            outproj = subproj
            outproj.skyout = finalsky      # Could also be None to calculate world coords in native system
            #if userepobs != None:
            #   outproj.epobs = userepobs
            #print "EPOBS=",outproj.epobs
            # At this stage, the flag 'usedate' is set to True if an observation epoch was found
            #print "USEDATE OUTPROJ=", outproj.usedate


         if outproj != None:
            #print "wor, empty, pix", wor
            #print empty
            #print pix
            # We need only new world coordinates, not pixels. Then use toworld() instead of mixed()
            wor = outproj.toworld(tuple(pix))
            #print "TOWORLD ---->NA: wor2, pix2", wor, pix

            #wor, pix = outproj.mixed(tuple(empty), tuple(pix))
            #print "NA: wor, pix", wor, pix
            subsetunits = outproj.cunit
         else:
            subsetunits = self.subproj.cunit
         """
         
         subsetunits = newproj.cunit
         """
         for i in range(subdim):
            # For all elements in each coordinate, subtract crpix
            grids[i] = pix[i] - offset[i]

         if mixax != None:               # Convert pixel of missing spatial axis to a grid
            grids.append(pix[-1] - crpix_mixax)
         """
         #print "grid, wor", grids, wor
         # grids is a tuple with 'subdim' coordinates. But note: each coordinate
         # can be an array with one or more numbers.
         # Make a NumPy array of this tuple and transpose the array
         # to get one position (i.e. subdim coordinates) in one row.
         #gt = asarray(grids).T
         wt = asarray(wor).T
         pt = asarray(pix).T
         # Append to the results list. Note that list appending is more flexible than
         # NumPy array concatenation.
         for w, p in zip(wt, pt):
            #r_grids.append(g)
            r_world.append(w)
            r_pixels.append(p)
      valid = True
      # return asarray(r_grids), asarray(r_world), asarray(r_pixels), subsetunits, ''
      return asarray(r_world), asarray(r_pixels), subsetunits, ''


def dotest():
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
              'CTYPE3' : 'FREQ-OHEL',
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
              "[pi]*3, [e**2]*3"
             ]
   for postxt in userpos:
      pw = str2pos(postxt, proj, mixpix=mixpix)
      print postxt, '=', pw[0], pw[1], pw[2]
   
   proj = origproj.sub((3,))
   userpos = ["1.415418199417E+09 hz",
              "1.415418199417E+03 Mhz",
              "1.415418199417 Ghz",
              "vopt 1.05000000e+06",
              "vrad 1.05000000e+06",
              "FREQ 1.41541820e+09",
              "0 10 20",
              "'1.41, 1.42, 1.43' Ghz",
              "[1.41, 1.42, 1.43] Ghz"
              ]
   for postxt in userpos:
      pw = str2pos(postxt, proj)
      print postxt, '=', pw[0], pw[1]

   proj = origproj.sub((1,2))
   userpos = ["0 0",
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
              "PC",
              "ac"]
   for postxt in userpos:
      pw = str2pos(postxt, proj)
      print postxt, '=', pw[0], pw[1]

   proj = origproj.sub((2,3,1))
   print "Spectral translations: ", proj.altspec
   userpos = ["{} 53.655 1.415418199417E+09 hz",
              "{} 53.655 1.415418199417E+03 Mhz",
              "53.655 deg 1.415418199417 Ghz",
              "{} 53.655 vopt 1.05000000e+06",
              "{} 53.655 , vopt 1050000 m/s",
              "0.0 , vopt 1050000 m/s",
              "10.0 , vopt 1050000 m/s",
              "{} 53.655 vrad 1.05000000e+06",
              "{} 53.655 FREQ 1.41541820e+09",
              "{} 53.655 wave 21.2 cm",
              "{} 53.655 vopt c_/300 m/s"]
   for postxt in userpos:
      pw = str2pos(postxt, proj, mixpix=mixpix)
      print postxt, '=', pw[0], pw[1]

   import readline
   upos = 'xxx'
   proj = origproj.sub((1,2)); mixpix = None
   while upos != '':
      upos = raw_input("Enter position(s) ..... [quit loop]: ")
      readline.add_history(upos) 
      if upos != '':
         pw = str2pos(upos, proj, mixpix=mixpix)
         if pw[3] != '':
            print pw[3]
         else:
            print upos, '=', pw[0], pw[1]

if __name__ == "__main__":
      dotest()