#!/usr/bin/env python
#----------------------------------------------------------------------
# FILE:    celestial.py
# PURPOSE: Build a matrix for conversions between sky systems and or
#          celestial reference systems and epochs.
#          In the calling environment one is supposed to use the function 
#         'skymatrix' only. The other functions are helper functions.
# AUTHOR:  M.G.R. Vogelaar, University of Groningen, The Netherlands
# DATE:    December 12, 2007
# UPDATE:  April 17, 2008
#          June 29, 2009: Changed docstrings for Sphinx
# VERSION: 1.0
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
# E: gipsy@astro.rug.nl
#----------------------------------------------------------------------
"""
Module Celestial
================

This document describes functions from the Python module *celestial*
(celestial.py) which provides a programmer with a basic set of
routines to transform a world coordinate in a given sky system
into a world coordinate of another system assuming
zero proper motion, parallax, and recessional velocity.

The most important function
builds a matrix for conversions of positions between sky systems,
celestial reference systems and epochs of the equinox.
This function is called :func:`skymatrix` and it can be used in the following
contexts:

   * Implicit, in module *wcs*, using the *Transformation* class as in::

       world_eq = (192.25, 27.4)   # FK4 coordinates of galactic pole
       tran = wcs.Transformation("equatorial fk4_no_e B1950.0", "galactic")
       print tran(world_eq)

   * As stand alone utility in scripts or in an interactive Python
     session. Usually one uses function :func:`sky2sky` to transform
     longitudes and latitudes::

       M = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                              (0,0,1.0), (10,20,20) )

   * Hidden in the *topixel()* and *toworld()* methods in module *wcs*.
     There the sky system is read from a (FITS) header and the 
     sky system for which we want the transformed coordinates
     is set with attribute *skyout* of the projection object. 
   
.. index::
   single: Tutorial; Celestial
   module: celestial
   
.. seealso:: Tutorial material:
   
     * :doc:`celestialbackground` which contains many examples with source code.


.. _celestial-skydefinitions:
   
Sky definitions
---------------

A sky definition can consist of a *sky system*,
a *reference system*, an *equinox* and an *epoch of
observation*. It is either a string or it is a tuple with one or more elements.
It can also be a single element.
The elements in a tuple representing a sky- or reference system are symbols
from the table below. For a string, the parts of the string representing a
sky- or reference system are minimal matched against the strings in the table below.
The match is case insensitive.


.. _celestial-skysystems:

Sky systems
...........

======================= ============= =============================================
Symbol                  String        Description
======================= ============= =============================================
*eq*, *equatorial*      EQUATORIAL    Equatorial coordinates (\u03B1, \u03B4),
                                      See also next table with reference systems
*ecl*, *ecliptic*       ECLIPTIC      Ecliptic coordinates (\u03BB, \u03B2)
                                      referred to the ecliptic and mean equinox
*gal*, *galactic*       GALACTIC      Galactic coordinates (lII, bII)
*sgal*, *supergalactic* SUPERGALACTIC De Vaucouleurs Supergalactic
                                      coordinates (sgl, sgb)
======================= ============= =============================================


.. _celestial-refsystems:
   
Reference systems
.................

.. tabularcolumns:: |p{20mm}|p{20mm}|p{110mm}|

======================= ============= =============================================
Symbol                  String        Description
======================= ============= =============================================
*fk4*                   FK4           Mean place pre-IAU 1976 system. FK4 is the
                                      old barycentric (i.e. w.r.t. the common
                                      center of mass) equatorial coordinate
                                      system, which should be qualified by an
                                      Equinox value.
                                      For accurate work FK4
                                      coordinate systems should also be qualified
                                      by an Epoch value. This is the *epoch of
                                      observation*.
*fk4_no_e*              FK4_NO_E,     The old FK4 (barycentric) equatorial system
                        FK4-NO-E      but without the *E-terms of aberration*.
                                      This coordinate system should also be
                                      qualified by both an Equinox and an Epoch
                                      value.
*fk5*                   FK5           Mean place post IAU 1976 system.
                                      Also a barycentric equatorial coordinate
                                      system.
                                      This should be qualified by an
                                      Equinox value (only).
*icrs*                  ICRS          The International Celestial Reference
                                      System, for optical data realized through
                                      the Hipparcos catalog.
                                      By definition, ICRS
                                      is not an equatorial system, but it is
                                      very close to the FK5 (J2000) system.
                                      No Equinox value is required.
*j2000*, *dynj2000*     DYNJ2000      This is an equatorial coordinate system
                                      based on the mean dynamical equator and
                                      equinox at epoch J2000.
                                      The dynamical equator and equinox differ
                                      slightly compared to the equator and equinox
                                      of FK5 at J2000 and the ICRS system.
                                      This system need not be qualified by an
                                      Equinox value
======================= ============= =============================================


.. note::
   Reference systems are stored in FITS headers under keyword *RADESYS=*.

.. note::
   Standard in FITS: RADESYS defaults to IRCS unless EQUINOX is given alone,
   in which case it defaults to FK4 prior to 1984 and FK5 after 1984.

   EQUINOX defaults to 2000 unless RADESYS is FK4, in which case it defaults
   to 1950.

.. note::
   In routines dealing with sky definitions tne names are minimal matched against
   a list with full names.
   
.. _celestial-epochs:
   
Epochs for the equinox and epoch of observation
...............................................

An epoch can be set in various ways. The options are distinguished
by a prefix. Only the 'B' and 'J' epochs can be negative.

.. tabularcolumns:: |p{15mm}|p{135mm}|

====== ===============================================================
Prefix Epoch
====== ===============================================================
B      Besselian epoch.
       Example: ``'B 1950'``, ``'b1950'``, ``'B1983.5'``, ``'-B1100'``
J      Julian epoch.
       Example: ``'j2000.7'``, ``'J 2000'``, ``'-j100.0'``
JD     Julian date. This number of days (with decimals)
       that have elapsed since the initial epoch defined
       as noon Universal Time (UT) Monday, January 1, 4713 BC
       in the proleptic Julian calendar
       Example: ``'JD2450123.7'``
MJD    The Modified Julian Day (MJD) is the number of days
       that have elapsed since midnight at the beginning of
       Wednesday November 17, 1858. In terms of the Julian day:
       MJD = JD - 2400000.5
       Example: ``'mJD 24034'``, ``'MJD50123.2'``
RJD    The Reduced Julian Day (RJD): Julian date counted from
       nearly the same day as the MJD,
       but lacks the additional offset of 12 hours that MJD has.
       It therefore starts from the previous noon UT or TT,
       on Tuesday November 16, 1858. It is defined as:
       RJD = JD - 2400000
       Example:  ``'rJD50123.2'``, ``'Rjd 23433'``
F      Various FITS formats:

       * DD/MM/YY  Old FITS format.
         Example: ``'F29/11/57'``
       * YYYY-MM-DD  FITS format.
         Example: ``'F2000-01-01'``
       * YYYY-MM-DDTHH:MM:SS FITS format with date and time.
         Example: ``'F2002-04-04T09:42:42.1'``

====== ===============================================================

**Epoch of observation**.

Reference system FK4 is not an inertial system. It is slowly rotating
and positions are further away from the true mean places if the date of observation
is greater than B1950. FK5 is an inertial system. If we convert coordinates
from FK4 to FK5, the accuracy of the FK5 position can be improved
if we know the date of the observation. So in all transformations where
a conversion between FK4 and  FK5 is involved, an epoch of observation can
be part of the sky definition. Note that this also involves a conversion between
galactic coordinates and equatorial, FK5 coordinates because that conversion
is done in steps and one step involves FK4.

To be able to distinguish an equinox from an epoch of observation, an epoch of
observation is followed by an underscore character and some arbitrary characters
to indicate that it is a special epoch (e.q. "B1960_OBS"). Only the underscore is
obligatory.

.. note::
   If a sky definition is entered as a string, there cannot be a space
   between the prefix and the epoch, because a space is a separator
   for the parser in :func:`celestial.skyparser`. 

.. note::
   An *epoch of observation* is either the second epoch in your input or
   or the epoch string has a suffix '_' which may be followed by arbitrary
   characters (e.g. "B1963.5_OBS").

Input Examples
..............

.. tabularcolumns:: |p{35mm}|p{25mm}|p{90mm}|

========================== ========================== =====================================
Input string               Description                Remarks
========================== ========================== =====================================
"eq"                       Equatorial, ICRS           ICRS because no reference system
                                                      and no equinox is given. 
"Eclip"                    Ecliptic, ICRS             Ecliptic coordinates
"ecl fk5"                  Ecliptic, FK5              Ecliptic coordinates with a non
                                                      default reference system
"GALACtic"                 Galactic II                Minimal match is case insensitive
"s"                        Supergalactic              Shortest string to identify system.
"fk4"                      Equatorial, FK4            Only a reference system is entered.
                                                      Sky system is assumed to be
                                                      equatorial
"B1960"                    Equatorial, FK4            Only an equinox is given. This is
                                                      a date before 1984 so FK4 is
                                                      assumed. Therefore the sky system
                                                      is equatorial
"EQ, fk4_no_e, B1960"      Equatorial, FK4 no e-terms Sky system, reference system,
                                                      and an equinox
"EQ, fk4-no-e, B1960"      Equatorial, FK4 no e-terms Same as above but underscores
                                                      replaced by hyphens.
"fk4,J1983.5_OBS"          Equatorial, FK4 + epobs    FK4 with an epoch of observation.
                                                      Note that only the underscore
                                                      is important.
"J1983.5_OBS"              Equatorial, FK4 + epobs    Only a date of observation. Then
                                                      reference system FK4 is assumed.
"EQ,fk4,B1960, B1983.5_O"  Equatorial, FK4 + epobs    A complete description of an
                                                      equatorial system.
"B1983.5_O fk4 B1960,eq"   Equatorial, FK4 + epobs    The same as above, showing that
                                                      the order of the elements are
                                                      unimportant.
========================== ========================== =====================================

Code examples
.............

To show that one can use both the tuple and the string representation of a system,
we use both for the same system and compare a transformed position.
The result should be 0 for both coordinates.

>>> world_eq = numpy.array([192.25, 27.4])     # FK4 coordinates of galactic pole
>>> tran1 = wcs.Transformation("equatorial fk4_no_e B1950.0", "galactic")
>>> tran2 = wcs.Transformation((wcs.equatorial, wcs.fk4_no_e, 'B1950.0'), wcs.galactic)
>>> print tran1(world_eq)-tran2(world_eq)
[ 0.  0.]


Module level data
-----------------


:data:`skyrefsystems`
   An object from class :class:`skyrefset` which is a container
   with a list with systems and two dictionaries with systems.

   >>> for s in skyrefsystems.skyrefs_list:
   >>>    print s.fullname, s.description, s.idnum

For programmers who need to access the id's of the sky and reference systems: 
External modules can set their own variables.
Here are some examples how one can do this.

Example with copy of celestial's variables:
  
  * ``eq = celestial.eq``
  * ``ec = celestial.ecl``
  * ``ga = celestial.gal``  etc.

Example with minimal match:
      
 * ``eq = celestial.skyrefsystems.minmatch2skyref('EQUA')[0].idnum``
 * ``ec = celestial.skyrefsystems.minmatch2skyref('ecli')[0].idnum``

Read this as: get the object for which a minimal match
is found. Item [0] is the object (the other is the number of times
a match is found). The 'idnum' is the integer for which we can
identify a system.

Or use the equivalent with method :meth:`skyrefset.minmatch2id`:
      
 * ``eq = celestial.skyrefsystems.minmatch2id('EQUA')``
 * ``ec = celestial.skyrefsystems.minmatch2id('ecli')``

Example with full name (case sensitive!):
      
 * ``eq = celestial.skyrefsystems.fullname2id('EQUATORIAL')``
 * ``ec = celestial.skyrefsystems.fullname2id('ECLIPTIC')``



Classes
-------

.. autoclass:: skyrefsys
.. autoclass:: skyrefset


Core Functions
--------------

.. index:: Input syntax for sky definitions
.. autofunction:: skyparser
.. autofunction:: skymatrix
.. autofunction:: sky2sky
.. index:: Epoch conversions
.. autofunction:: epochs

Utility functions
-----------------

.. index:: Julian day number
.. autofunction:: JD
.. index:: Label formatting
.. autofunction:: lon2hms
.. autofunction:: lat2dms
.. autofunction:: lon2dms
.. index:: Besselian epochs
.. autofunction:: JD2epochBessel
.. autofunction:: epochBessel2JD
.. index:: Julian epochs
.. autofunction:: JD2epochJulian
.. autofunction:: epochJulian2JD
.. index:: Obliquity
.. autofunction:: obliquity1980
.. autofunction:: obliquity2000
.. index:: Precession angles
.. autofunction:: IAU2006precangles
.. autofunction:: Lieskeprecangles
.. autofunction:: Newcombprecangles


.. index:: Rotation matrices

Rotation matrices
-----------------

.. autofunction:: MatrixEqB19502Gal
.. autofunction:: MatrixGal2Sgal
.. autofunction:: MatrixEq2Ecl
.. autofunction:: FK42FK5Matrix
.. autofunction:: ICRS2FK5Matrix
.. autofunction:: ICRS2J2000Matrix
.. autofunction:: JMatrixEpoch12Epoch2
.. autofunction:: BMatrixEpoch12Epoch2
.. autofunction:: IAU2006MatrixEpoch12Epoch2
.. autofunction:: MatrixEpoch12Epoch2

.. index:: Elliptic terms of aberration

Functions related to E-terms
----------------------------

.. autofunction:: getEterms
.. autofunction:: addEterms
.. autofunction:: removeEterms

"""
import numpy as n
import types
from re import split as re_split
import six


class skyrefsys(object):
#----------------------------------------------------------------------
   """
Class creates an object that describes a sky- or reference system.
This module initializes a set of systems. They are accessible
through methods in class :class:`celestial.skyrefset`

:param fullname:
   Complete name to identify the system, e.g. *"EQUATORIAL"*
:type fullname:
   String
:param idnum:
   A unique integer to identify the system
:type idnum:
   Integer
:param description:
   A short description of the system
:type description:
   String
:param refsystem:
   Is this system a reference system?
:type refsystem:
   Boolean


**Attributes:**
   
.. attribute:: fullname

      A string to identify a system, e.g. "EQUATORIAL".
      
.. attribute:: idnum

      A unique integer to identify the system.

.. attribute:: description

      A string to describe the system.
      
.. attribute:: refsystem

      If *True* then this system is a reference system.
      Else it is a sky system.
   
   """
#----------------------------------------------------------------------
   def __init__(self, fullname, idnum, description, refsystem):
      self.fullname = fullname
      self.idnum = idnum
      self.description = description
      self.refsystem = refsystem           # Boolean



class skyrefset(object):
#----------------------------------------------------------------------
   """
A container with sky- and reference system objects from class
:class:`celestial.skyrefsys`. It is used to initialize variables
that can be used as identifiers for sky- or reference systems.
Applications can use its methods to retrieve information given
an integer identifier or (part of) a string.

For example when we want a list with all the supported systems
then type: 

>>> for s in skyrefsystems.skyrefs_list:
>>>    print s.fullname, s.description, s.idnum

.. automethod:: append
.. automethod:: minmatch2skyref
.. automethod:: minmatch2id
.. automethod:: fullname2id
.. automethod:: id2skyref
.. automethod:: id2fullname
.. automethod:: id2description

**Attributes:**

   .. attribute:: skyrefs_list
   
         The list with systems
   
   .. attribute:: skyrefs_id
         
         A dictionary with the systems and with id's as keys
   
   .. attribute:: skyrefs_fullname
   
         A dictionary with the systems and with full names as keys

:Examples: Next short script shows how to get a list with
   sky systems and how to use methods of this class to get data for
   a system if an (integer) id is found:: 

      from kapteyn.celestial import skyrefsystems
      
      for s in skyrefsystems.skyrefs_list:
         print s.fullname, s.description, s.idnum
         i = s.idnum
         print "Full name using id2fullname:", skyrefsystems.id2fullname(i)
         print "Description using id2description:", skyrefsystems.id2description(i)
         print "id of %s with minimal match: %d" % \\
               (s.fullname[:3], skyrefsystems.minmatch2skyref(s.fullname[:3])[0].idnum)
         print "id of %s with minimal match, alternative: %d" % \\
               (s.fullname[:3], skyrefsystems.minmatch2id(s.fullname[:3]))
         print "id of %s with full name: %d" % \\
               (s.fullname[:3], skyrefsystems.fullname2id(s.fullname))

   """
#----------------------------------------------------------------------
   def __init__(self):
      self.skyrefs_list = []                     # The list with systems
      self.skyrefs_id = {}                       # A dict. version with id's as keys
      self.skyrefs_fullname = {}                 # A dict. version with names as keys
      
   def append(self, skyrefsys):
      """
      :param skyrefsys:
         Append this system to the list with supported systems
      :type skyrefsys:
         Instance of class :class:`skyrefsys`

      :Returns:
         A unique integer id which can be used to identify a system.
      """
      self.skyrefs_list.append(skyrefsys)
      self.skyrefs_id[skyrefsys.idnum] = skyrefsys
      self.skyrefs_fullname[skyrefsys.fullname] = skyrefsys.idnum
      return skyrefsys.idnum
   
   def minmatch2skyref(self, s):
      """
      Return the relevant skyrefsys object with the number of times
      it is matched or return None if nothing was found.

      :param s:
         Part of the string name of a system
      :type s:
         String

      :Returns:
         Instance of class :class:`skyrefsys` and the number of times
         that the input string gives a match.
      """
      s = s.upper()
      if s.startswith("FK4"):       # Allow also FK4-NO-E. Replace hyphen by underscore
         s = s.replace('-','_')
      found = 0
      found_sk = None
      for sk in self.skyrefs_list:
         foundone = False
         if s == sk.fullname:
            found += 1
            found_sk = sk
            return found_sk, found         # Exact match !
         else:
            i = sk.fullname.find(s, 0, len(s))
         if i == 0:
            found += 1
            found_sk = sk
      return found_sk, found
         
   def minmatch2id(self, s):
      """
      From the found skyrefsys object corresponding to string *s*,
      return the idnum attribute. Case insensitive minimal match
      is used to find the sky- or reference system.
      Return None if there was no match or more than one match.

      :param s:
         Part of the string name of a system
      :type s:
         String

      :Returns:
         Instance of class :class:`skyrefsys` or None if there was not
         a match or more than one match.
      """
      s = s.upper()
      found = 0
      found_sk = None
      for sk in self.skyrefs_list:
         foundone = False
         if s == sk.fullname:
            found += 1
            found_sk = sk
            return found_sk.idnum         # Exact match !
         else:
            i = sk.fullname.find(s, 0, len(s))
         if i == 0:
            found += 1
            found_sk = sk
      if found == 1:
         return found_sk.idnum
      return None
         
   def fullname2id(self, fullname):
      """
      This is the fastest method to get an integer id from a
      string which represents a sky system or a reference system.
      Note that the routine is case sensitive because it uses
      the full names as keys in a dictionary.
      The parameter *fullname* therefore must be in in capitals!

      :param fullname:
         The full descriptive name of a system e.g. "EQUATORIAL"
      :type fullname:
         String

      :Returns:
          Integer id of the found system or *None* if nothing was found.
      """
      try:
         idnum = self.skyrefs_fullname[fullname]
      except:
         idnum = None
      return idnum
      
   def id2skyref(self, idnum):
      """
      Given an integer id of a system, return the corresponding system
      as an instance of class :class:`skyrefsys`.
      Usually the calling environment will deal with the attributes of
      this object, for instance to write a short description of the system.

      :param idnum:
         Integer id of a system
      :type idnum:
         Integer

      :Returns:
         Instance of class :class:`skyrefsys` or None if there was not
         a corresponding system.
      """
      try:
         sys = self.skyrefs_id[idnum]
      except:
         sys = None
      return sys
      
   def id2fullname(self, idnum):
      """
      Given an integer id of a system, return the full name
      of the corresponding system.

      :param idnum:
         Integer id of a system
      :type idnum:
         Integer

      :Returns:
         Full name (e.g. "EQUATORIAL") of the 
         corresponding system or an empty string if nothing was found.
      """
      try:
         fullname = self.skyrefs_id[idnum].fullname
      except:
         fullname = ''
      return fullname
      
   def id2description(self, idnum):
      """
      Given an integer id of a system, return the description
      of the corresponding system.

      :param idnum:
         Integer id of a system
      :type idnum:
         Integer

      :Returns:
         A short description of the 
         corresponding system or an empty string if nothing was found.
      """
      try:
         descr = self.skyrefs_id[idnum].description
      except:
         descr = ''
      return descr


# Create a collection of sky systems and reference systems.
# The integer is an identifier and the last parameter tells you
# whether the system is a reference system or not.
# Also a set of global integer variable is created to facilitate
# parsers in this module.
skyrefsystems = skyrefset()
eq       = skyrefsystems.append(skyrefsys('EQUATORIAL', 0, "Equatorial", False))
ecl      = skyrefsystems.append(skyrefsys('ECLIPTIC', 1, "Ecliptic", False))
gal      = skyrefsystems.append(skyrefsys('GALACTIC', 2, "Galactic II", False))
sgal     = skyrefsystems.append(skyrefsys('SUPERGALACTIC', 3, "Supergalactic", False))
fk4      = skyrefsystems.append(skyrefsys('FK4', 4, "Fourth Fundamental Catalogue", True))
fk4_no_e = skyrefsystems.append(skyrefsys('FK4_NO_E', 5, "FK4 without E-terms", True))
fk5      = skyrefsystems.append(skyrefsys('FK5', 6, "Fifth Fundamental Catalogue ", True))
icrs     = skyrefsystems.append(skyrefsys('ICRS', 7, "International Celestial Reference System", True))
j2000    = skyrefsystems.append(skyrefsys('DYNJ2000', 8, "Dynamic J2000", True))

# Some aliases
equatorial = eq; ecliptic = ecl; galactic = gal; supergalactic = sgal; dynj2000 = j2000 


#for s in skyrefsystems.skyrefs_list:
#   print s.fullname, s.description, s.idnum
# Tests:
# print "EQ, EC:", eq, ecl, gal, sgal, fk4, fk4_no_e, fk5, icrs, j2000
#
#for i in range(10):
#   s = skyrefsystems.id2skyref(i)
#   if s != None:
#      print s.fullname, s.description, s.idnum
#      print "Full name using id2fullname:", skyrefsystems.id2fullname(i)
#      print "Description using id2description:", skyrefsystems.id2description(i)
#      print "id of %s with minimal match: %d" % (s.fullname[:3], skyrefsystems.minmatch2skyref(s.fullname[:3])[0].idnum)
#      print "id of %s with full name: %d" % (s.fullname[:3], skyrefsystems.fullname2id(s.fullname))
#      print "id of %s with minimal match 2: %d" % (s.fullname[:3], skyrefsystems.minmatch2id(s.fullname[:3]))



# Conversion factors deg <-> rad
convd2r = n.pi/180.0
convr2d = 180.0/n.pi


#----------------------------------------------------------------------
# Some utility routines
#---------------------------------------------------------------------
def d2r(degs):
   return degs * convd2r

def r2d(rads):
   return rads * convr2d

def I():
   return n.identity(3, dtype='d')


def JD(year, month, day):
#----------------------------------------------------------------------
   """
Calculate Julian day number (Julian date)

:param year:
   Year (nnnn)
:type year:
   Integer
:param month:
   Month (nn)
:type month:
   Integer
:param day:
   Day (nn.n...)
:type day:
   Floating point number
   
   
:Returns:
   Julian day number *jd*.

:Reference:
   Meeus, Astronomical formula for Calculators, 2nd ed, 1982

:Notes:
   Months start at 1. Days start at 1. The Julian day begins at
   Greenwich mean noon, i.e. at 12h. So Jan 1, 1984 at 0h is
   entered as *JD(1984,1,1)* and Jan 1, 1984 at 12h is entered
   as *JD(1984,1,1.5)*

   There is a jump at *JD(1582,10,15)* caused by a change of
   calendars. For dates after 1582-10-15 one enters a date
   from the Julian calendar and before this date you enter a
   date from the Gregorian calendar.

:Examples:
   * Julian date of JD reference:
     ``print celestial.JD(-4712,1,1.5) ==> 0.0``
   * The first day of 1 B.C.:
     ``print celestial.JD(0,1,1) ==> 1721057.5``
   * Last day before Gregorian reform:
     ``print celestial.JD(1582,10,4) ==> 2299159.5``
   * First day of Gregorian reform:
     ``print celestial.JD(1582,10,15) ==> 2299170.5``
   * Half a day later:
     ``print celestial.JD(1582,10,15.5) ==> 2299161.0``
   * Unix reference:
     ``print celestial.JD(1970,1,1) ==> 2440587.5``

   """
#----------------------------------------------------------------------   
   if (month > 2):
      y = year
      m = month
   elif (month == 1 or month == 2):
      y = year - 1
      m = month + 12

   calday = year + month/100.0 + day / 10000.0

   if (calday > 1582.1015):
      A = int(y/100.0)
      B = 2 - A + int(A/4.0)
   else:
      B = 0

   if (calday > 0.0229):                  # Dates after 29 February year 0
      jd = int(365.25*y) + int(30.6001*(m+1)) + day + 1720994.50 + B
   else:
      jd = int(365.25*y-0.75) + int(30.6001*(m+1)) + day + 1720994.50 + B

   return jd



def longlat2xyz(longlat):
   """
-----------------------------------------------------------------------
Purpose:   Given two angles in longitude and latitude return 
           corresponding Cartesian coordinates x,y,z
Input:     Sequence of positions e.g. ((a1,d1),(a2,d2), ...)
Returns:   Corresponding values of x,y,z in same order as input
Reference: -
Notes:     The three coordinate axes x, y and z, the set of 
           right-handed Cartesian axes that correspond to the
           usual celestial spherical coordinate system. 
           The xy-plane is the equator, the z-axis 
           points toward the north celestial pole, and the 
           x-axis points toward the origin of right ascension. 
-----------------------------------------------------------------------
   """
   lon = d2r( n.asarray(longlat[:,0],'d').flatten(1) )
   lat = d2r( n.asarray(longlat[:,1],'d').flatten(1) )
   x = n.cos(lon)*n.cos(lat)
   y = n.sin(lon)*n.cos(lat)
   z = n.sin(lat)
   return n.mat((x,y,z))



def xyz2longlat(xyz):
   """
-----------------------------------------------------------------------
Purpose:   Given Cartesian x,y,z return corresponding longitude and 
           latitude in degrees.
Input:     Sequence of tuples with values for x,y,z
Returns:   The same number of positions (longitude, latitude and in the
           same order as the input.
Reference: -
Notes:     Note that one can expect strange behavior for the values 
           of the longitudes very close to the pole. In fact, at the 
           poles itself, the longitudes are meaningless.
-----------------------------------------------------------------------
   """
   x = n.asarray(xyz[0],'d').flatten(1)
   y = n.asarray(xyz[1],'d').flatten(1)
   z = n.asarray(xyz[2],'d').flatten(1)

   lat = r2d( n.arctan2(z, n.sqrt(x*x+y*y)) )
   lon = r2d( n.arctan2(y, x) )
#   eps = n.array(0.00000001, 'd')
#   lon = n.where( ((abs(lat) > 89.9999) & (abs(x) < eps) & (abs(y) < eps)),\
#                  0.0, r2d( n.arctan2(y, x)))
   lon = n.where(lon < 0.0, lon+360.0, lon)
   return n.mat([lon,lat]).T




def lon2hms(a, prec=1, delta=None, tex=False):
#----------------------------------------------------------------------
   """
Convert an angle in degrees to **hours, minutes, seconds** format.

:param a:
   Angle (in degrees) for which we want to create a formatted text label.
:type a:
   Floating point number
:param prec:
   The required number of decimals in the seconds part of output.
   If a value is omitted, then the default is 1.
:type prec:
   Integer
:param delta:
   If one labels world coordinates along an axis then the default labels
   are in hours, minutes and seconds with some decimal number. This is probably
   not want you want if the step size between subsequent positions is
   for example an integer number of degrees or minutes.
   Then you want labels showing only hours or hours and minutes.
   This function tries to find out whether this is the case (given a value
   for *delta*) or not. If so, a minimum length label is returned.
:type delta:
   *None* or a floating point number
:param tex:
   The default is *False*. If set to *True*, the string is formatted
   in LaTeX. Such labels can be plotted in, for example, Matplotlib.
:type tex:
   Boolean

:Returns:
   Formatted string representing the input angle.
   
:Notes:
   Longitudes are forced into the range, 360 deg. and then
   converted to hours, minutes and seconds.

:Examples:
   Format a position in hms and dms:

          >>> ra = 359.9999
          >>> dec = 0.0000123
          >>> print celestial.lon2hms(ra),  celestial.lat2dms(dec)
              00h 00m  0.0s +00d 00m  0.0s
          >>> print celestial.lon2hms(ra, 2),  celestial.lat2dms(dec, 2)
              23h 59m 59.98s +00d 00m  0.04s
          >>> print celestial.lon2hms(ra, 4),  celestial.lat2dms(dec, 4)
              23h 59m 59.9760s +00d 00m  0.0443s
   """
#----------------------------------------------------------------------
   degs = n.fmod(a, 360.0)  # Now in range -360, 360
   if degs < 0.0:
      degs += 360.0	
   if prec < 0:
      prec = 0
   # How many seconds is this. Round to 'prec'
   sec = n.round(degs*240.0, prec)
   sec = n.fmod(sec, 360.0*240.0)     # Rounding can result in 360 deg again, so correct
   Isec = n.int(sec)     # Integer seconds
   Fsec = sec - Isec     # Fractional remainder
   hours = Isec / 3600.0
   Ihours = n.int(hours)
   secleft = Isec - Ihours*3600.0
   Imin = int(secleft / 60.0)
   secleft = secleft - Imin*60.0
   # print "\n prec Ideg, Imin, secleft, Fsec", prec, Ideg, Imin, secleft, Fsec
   if tex:
      if prec > 0:
         hms = "%d^h%.2d^m%.2d^s" % (Ihours, Imin, secleft)
         fsec = "%*.*d" % (prec, prec, int(round(Fsec*10.0**prec,0)))
         s = r"$" + hms + fsec + "$"
      else:
         s1 = r"$%d^h%.2d^m%.2d^s$" % (Ihours, Imin, secleft)
         if delta == None:
            s = s1
         else:
            if (delta*3600.0) % (15.0*3600) == 0.0: # Only hours
               s = r"$%d^h$" % Ihours
            elif (delta*3600.0) % (15.0*60) == 0.0: # Only hours and minutes
               s = r"$%d^h%.2d^m$" % (Ihours, Imin)
            else:
               s = s1
   else:
      if prec > 0:
         s = "%.2dh%.2dm%0*.*fs" % (Ihours, Imin, prec+3, prec, secleft+Fsec)
      else:
         s = "%.2dh%.2dm%2ds" % (Ihours, Imin, secleft)
   return s



def lat2dms(a, prec=1, delta=None, tex=False):
#----------------------------------------------------------------------
   """
Convert an angle in degrees into the **degrees, minutes, seconds** 
format assuming it was a latitude. Its value should be in
the range -90 to 90 degrees


:param a:
   Angle (in degrees) for which we want to create a formatted text label.
:type a:
   Floating point number
:param prec:
   The required number of decimals in the seconds part of output.
   If a value is omitted, then the default is 1.
:type prec:
   Integer
:param delta:
   If one labels world coordinates along an axis then the default labels
   are in degrees, minutes and seconds with some decimal number. This is probably
   not want you want if the step size between subsequent positions is
   for example an integer number of degrees or minutes.
   Then you want labels showing only degrees or degrees and minutes.
   This function tries to find out whether this is the case (given a value
   for *delta*) or not. If so, a minimum length label is returned.
:type delta:
   *None* or a floating point number
:param tex:
   The default is *False*. If set to *True*, the string is formatted
   in LaTeX. Such labels can be plotted in, for example, Matplotlib.
:type tex:
   Boolean

:Returns:
   Formatted string representing the input angle or a string
   with '#' characters indicating that the input was out of range.

:Notes:
   The HMS and DMS format 
   should be treated differently because their ranges in world
   coordinates are different.
   Longitudes should be in range of (0,360)
   degrees. So -10 deg is in fact 350 deg. and 370 deg is in
   fact 10 deg. Latitudes range from -90 to 90 degrees. Then 91
   degrees is in fact 89 degrees but at a longitude that is
   separated 180 deg. from the stated longitude. But we don't
   have control over the longitudes here so the only thing we
   can do is reject the value and return a dummy string.

   """
#----------------------------------------------------------------------

   if a > 90.0 or a < -90.0:
      return "##d##m##s";
   sign = 1;
   si = ' ' # one space
   if a < 0.0:
      sign = -1
      si = '-'
   degs = sign * a  # Make positive
   if prec < 0:
     prec = 0
   # How many seconds is this. Round to 'prec'
   sec = n.round(degs*3600.0, prec)
   Isec = n.int(sec)     # Integer seconds
   Fsec = sec - Isec     # Fractional remainder
   degs = Isec / 3600.0
   Ideg = n.int(degs)
   secleft = Isec - Ideg*3600.0
   Imin = int(secleft / 60.0)
   secleft = secleft - Imin*60.0
   if tex:
      if prec > 0:
         dms = r"%c%d^{\circ}%.2d^{\prime}%.2d^{\prime\prime}" % (si, Ideg, Imin, secleft)
         fsec = ".%*.*d" % (prec, prec, int(round(Fsec*10.0**prec,0)))
         s = r"$" + dms + fsec + "$"
      else:
         s1 = r"$%c%d^{\circ}%.2d^{\prime}%.2d^{\prime\prime}$" % (si, Ideg, Imin, secleft)
         if delta == None:
            s = s1
         else:
            if (delta*3600.0) % 3600 == 0.0: # Only degrees
               s = r"$%c%d^{\circ}$" % (si,Ideg)
            elif (delta*3600.0) % 60 == 0.0:  # Only degrees and minutes
               s = r"$%c%d^{\circ}%.2d^{\prime}$" % (si, Ideg, Imin)
            else:
               s = s1
   else:
      if prec > 0:
         s = "%c%.2dd%.2dm%0*.*fs" % (si, Ideg, Imin, prec+3, prec, secleft+Fsec)
      else:
         s = "%c%.2dd%.2dm%2ds" % (si, Ideg, Imin, secleft)
   return s



def lon2dms(a, prec=1, delta=None, tex=False):
#----------------------------------------------------------------------
   """
Convert an angle in degrees to **degrees, minutes, seconds** format,
assuming the input is a longitude but not associated with an equatorial
system.

:param a:
   Angle (in degrees) for which we want to create a formatted text label
:type a:
   Floating point number
:param prec:
   The required number of decimals in the seconds part of output
   If a value is omitted, then the default is 1.
:type prec:
   Integer
:param delta:
   If one labels world coordinates along an axis then the default labels
   are in hours, minutes and seconds with some decimal number. This is probably
   not want you want if the step size between subsequent positions is
   for example an integer number of degrees or minutes.
   Then you want labels showing only degrees or degrees and minutes.
   This function tries to find out whether this is the case (given a value
   for *delta*) or not. If so, a minimum length label is returned.
:type delta:
   *None* or a floating point number
:param tex:
   The default is *False*. If set to *True*, the string is formatted
   in LaTeX. Such labels can be plotted in, for example, Matplotlib.
:type tex:
   Boolean

:Returns:
   Formatted string representing the input angle.
   
:Notes:
   Longitudes are forced into the range 0, 360 deg. and then
   converted to hours, minutes and seconds.

:Examples:
   Format a longitude to dms:

      >>> print celestial.lon2dms(167.342, 4)
         167d 20m 31.2000s
      >>> print celestial.lon2dms(-10, 4)
         350d  0m  0.0000s

   """
#----------------------------------------------------------------------
   degs = n.fmod(a, 360.0)  # Now in range -360, 360
   if (a < 0.0):
      degs += 360.0         # In range 0, 360 circle-wise
   if prec < 0:
     prec = 0
   # How many seconds is this. Round to 'prec'
   sec = n.round(degs*3600.0, prec)
   Isec = n.int(sec)     # Integer seconds
   Fsec = sec - Isec     # Fractional remainder
   degs = Isec / 3600.0
   Ideg = n.int(degs)
   secleft = Isec - Ideg*3600.0
   Imin = int(secleft / 60.0)
   secleft = secleft - Imin*60.0
   if tex:
      if prec > 0:
         dms = r"%d^{\circ}%.2d^{\prime}%.2d^{\prime\prime}" % (Ideg, Imin, secleft)         
         fsec = ".%*.*d" % (prec, prec, int(round(Fsec*10.0**prec,0)))
         s = r"$" + dms + fsec + "$"
      else:
         s1 = r"$%d^{\circ}%.2d^{\prime}%.2d^{\prime\prime}$" % (Ideg, Imin, secleft)
         if delta == None:
            s = s1
         else:
            if (delta*3600.0) % 3600 == 0.0: # Only degrees
               s = r"$%d^{\circ}$" % Ideg
            elif (delta*3600.0) % 60 == 0.0:  # Only degrees and minutes
               s = r"$%d^{\circ}%.2d^{\prime}$" % (Ideg, Imin)
            else:
               s = s1
   else:
      if prec > 0:
         s = "%4dd%2dm%0*.*fs" % (Ideg, Imin, prec+3, prec, secleft+Fsec)
      else:
         s = "%4dd%2dm%2ds" % (Ideg, Imin, secleft)
   return s



def JD2epochBessel(JD):
#----------------------------------------------------------------------
   """
Convert a Julian date to a Besselian epoch.

:param JD:
   Julian date (e.g. 2445700.5)
:type JD:
   Floating point number
   
:Returns:
   Besselian epoch (e.g. 1983.9)

:Reference:
   Standards Of Fundamental Astronomy,
    
   http://www.iau-sofa.rl.ac.uk/2003_0429/sofa/epb.html

:Notes:
   e.g. 2445700.5 -> 1983.99956681

   One *Tropical Year* is 365.242198781 days and
   JD(1900) = 2415020.31352
   
   If we know the JD then the Besselian epoch can be
   calculated with:
   
   ``BE = B[1900 + (JD - 2415020.31352)/365.242198781]``

   Expression corresponds to the IAU SOFA expression in the reference
   with:
   ``2451545-36524.68648 = 2415020.31352``

   """
#---------------------------------------------------------------------- 
   return 1900.0 + (JD-2415020.31352)/365.242198781



def epochBessel2JD(Bepoch):
#----------------------------------------------------------------------
   """
Convert a Besselian epoch to a Julian date

:param Bepoch:
   Besselian epoch in format nnnn.nn
:type Bepoch:
   Floating point number
   
:Returns:
   Julian date

:Reference:
   See: :func:`JD2epochBessel`
   
:Notes:
   e.g. 1983.99956681 converts into 2445700.5
   It's the inverse of :func:`JD2epochBessel`

   """
#----------------------------------------------------------------------
   return (Bepoch-1900.0)*365.242198781 + 2415020.31352



def JD2epochJulian(JD):
#---------------------------------------------------------------------
   """
Convert a Julian date to a Julian epoch

:param JD:
   Julian date
:type JD:
   Floating point number
   
:Returns:
   Julian epoch

:Reference:
   Standards Of Fundamental Astronomy,
   
   http://www.iau-sofa.rl.ac.uk/2003_0429/sofa/epj.html

:Notes:
   e.g. ``2445700.5 converts into 1983.99863107``
   Assuming years of exactly 365.25 days, we can
   calculate a Julian epoch from a Julian date.
   Expression corresponds to IAU SOFA routine 'epj'

   """
#----------------------------------------------------------------------
   return 2000.0 + (JD - 2451545.0)/365.25



def epochJulian2JD(Jepoch):
#----------------------------------------------------------------------
   """
Convert a Julian epoch to a Julian date

:param Jepoch:
   Julian epoch (in format nnnn.nn)
:type Jepoch:
   Floating point number
   
:Returns:
   Julian date

:Reference:
   See :func:`JD2epochJulian`
   
:Notes:
   e.g. ``1983.99863107 converts into 2445700.5``
   It's the inverse of function JD2epochJulian

   """
#----------------------------------------------------------------------
   return (Jepoch-2000.0)*365.25 + 2451545.0



def obliquity1980(jd):
#----------------------------------------------------------------------
   """
What is the obliquity of the ecliptic at this Julian date? (IAU 1980 model)

:param jd:
   Julian date
:type jd:
   Floating point number
   
:Returns:
   Mean obliquity in degrees

:Reference:
   Explanatory Supplement to the Astronomical Almanac,
   P. Kenneth Seidelmann (ed), University Science Books (1992),
   Expression 3.222-1 (p114).

:Notes:
   The epoch is entered in Julian date and the time is calculated 
   w.r.t. J2000.
   
   The obliquity is the angle between the mean equator and
   ecliptic, or, between the ecliptic pole and mean celestial
   pole of date

   """
#----------------------------------------------------------------------
   # T = (Date - 1 jan, 2000, 12h noon)
   T = (jd-2451545.0)/36525.0
   eps = (84381.448+(-46.8150+(-0.00059+0.001813*T)*T)*T) / 3600.0
   return eps



def obliquity2000(jd):
#----------------------------------------------------------------------
   """
What is the obliquity of the ecliptic at this Julian date?
(IAU model 2000)

:param jd:
   Julian date
:type jd:
   Floating point number
   
:Returns:
   Mean obliquity in degrees

:Reference:
   Fukushima, T. 2003, AJ, 126,1
   Kaplan, H., 2005, The IAU Resolutions
   on Astronomical Reference Systems,
   Time Scales, and Earth Rotation Models,
   United States Naval Observatory circular no. 179,
   http://aa.usno.navy.mil/publications/docs/Circular_179.pdf
   (page 44)
   
:Notes:
   The epoch is entered in Julian date and the time is calculated
   w.r.t. J2000.
   
   The obliquity is the angle between the mean equator and
   ecliptic, or, between the ecliptic pole and mean celestial
   pole of date.
           
   """
#----------------------------------------------------------------------
   # T = (Date - 1 jan, 2000, 12h noon)
   T = (jd-2451545.0)/36525.0

   eps = (84381.406       +
         (  -46.836769    +
         (   -0.0001831   +
         (    0.00200340  +
         (   -0.000000576 +
         (   -0.0000000434 )*T)*T)*T)*T)*T) / 3600.0
   return eps



def IAU2006precangles(epoch):
#----------------------------------------------------------------------
   """
Calculate IAU 2000 precession angles for precession from
input epoch to J2000.

:param epoch:
   Julian epoch of observation.
:type epoch:
   Floating point number
   
:Returns:
   Angles \u03B6 (zeta), z, \u03B8 (theta) in degrees to setup a rotation matrix
   to transform from J2000 to input epoch.
   
:Reference:
   Capitaine N. et al., IAU 2000 precession A&A 412, 567-586 (2003)

:Notes:
   Input are Julian epochs!
   ``T = (jd-2451545.0)/36525.0``
   Combined with ``jd = Jepoch-2000.0)*365.25 + 2451545.0`` gives:
   (see module code at function *epochJulian2JD(epoch)*)
   ``T = (epoch-2000.0)/100.0``

   This function should be updated as soon as there are IAU2006 adopted
   angles to replace the angles used in this function.

   """
#----------------------------------------------------------------------
   # T = (Current epoch - 1 jan, 2000, 12h noon)
   T = (epoch-2000.0)/100.0
   d0 = 2.5976176
   d1 = 2306.0809506
   d2 = 0.3019015
   d3 = 0.0179663
   d4 = -0.0000327
   d5 = -0.0000002
   zeta_a = T*(d1+T*(d2+T*(d3+T*(d4+T*(d5)))))+d0
   d0 = -2.5976176
   d1 = 2306.0803226
   d2 = 1.0947790
   d3 = 0.0182273
   d4 = 0.0000470
   d5 = -0.0000003
   z_a = T*(d1+T*(d2+T*(d3+T*(d4+T*(d5)))))+d0
   d0 = 0.0
   d1 = 2004.1917476
   d2 = -0.4269353
   d3 = -0.0418251
   d4 = -0.0000601
   d5 = -0.0000001
   theta_a = T*(d1+T*(d2+T*(d3+T*(d4+T*(d5)))))+d0
   # Return values in degrees
   return zeta_a/3600.0, z_a/3600.0, theta_a/3600.0



def Lieskeprecangles(jd1, jd2):
#----------------------------------------------------------------------
   """
Calculate IAU 1976 precession angles for a precession
of epoch corresponding to Julian date jd1 to epoch corresponds
to Julian date jd2.

:param jd1:
   Julian date for start epoch
:type jd1:
   Floating point number 
:param jd2:
   Julian date for end epoch
:type jd2:
   Floating point number 
   
:Returns:
   Angles \u03B6 (zeta), z, \u03B8 (theta) degrees

:Reference:
   Lieske,J.H., 1979. Astron.Astrophys.,73,282.
   equations (6) & (7), p283.

:Notes:
   The ES (Explanatory Supplement to the Astronomical Almanac)
   lists for a IAU1976 precession from 1984, January 1d0h to J2000
   the angles in **arcsec**:  ``xi_a=368.9985, ze_a=369.0188 and th_a=320.7279``
   Using the functions in this module, this can be calculated
   by applying:

      >>> jd1 = celestial.JD(1984,1,1)
      >>> jd2 = celestial.JD(2000,1,1.5)
      >>> print celestial.Lieskeprecangles(jd1, jd2)
         (0.10249958598931658, 0.10250522534285664, 0.089091092843880629)
      >>> print [a*3600 for a in angles]
          [368.99850956153966, 369.01881123428387, 320.72793423797026]

   The function returns values in degrees, while literature values
   often are listed in seconds of arc.


   Lieske's fit belongs to the so called Quasi-Linear Types
   Below a table with the precision (according to IAU SOFA):

      * 1960AD to 2040AD: < 0.1"
      * 1640AD to 2360AD: < 1"
      * 500BC to 3000AD: < 3"
      * 1200BC to 3900AD: > 10"
      * < 4200BC or > 5600AD: > 100"
      * < 6800BC or > 8200AD: > 1000"
      
   """
#----------------------------------------------------------------------
   #   T = (Current epoch - 1 jan, 2000, 12h noon)
   T = (jd1-2451545.0)/36525.0
   t = (jd2-jd1)/36525.0

   d1 = 2306.2181
   d2 = 1.39656
   d3 = -0.000139
   d4 = 0.30188
   d5 = 0.000344
   d6 = 0.017998
   D1 = d1 + T*(d2+T*d3)
   zeta_a = t*(D1 + t*((d4+d5*T) + t*d6))
   # d1 = 2306.2181
   # d2 = 1.39656
   # d3 = -0.000139
   d4 = 1.09468
   d5 = -0.000066
   d6 = 0.018203
   z_a = t*(D1 + t*((d4+d5*T) + t*d6))
   d1 = 2004.3109
   d2 = -0.85330
   d3 = -0.000217
   d4 = -0.42665
   d5 = -0.000217
   d6 = -0.041833
   D1 = d1 + T*(d2+T*d3)
   theta_a = t*(D1 + t*((d4+d5*T) + t*d6))
   # Return values in degrees
   return zeta_a/3600.0, z_a/3600.0, theta_a/3600.0



def Newcombprecangles(epoch1, epoch2):
#----------------------------------------------------------------------
   """
Calculate precession angles for a precession in FK4, using
Newcomb's method (Woolard and Clemence angles)

:param epoch1:
   Besselian start epoch
:type epoch1:
   Floating point number
:param epoch2:
   Besselian end epoch
:type epoch2:
   Floating point number


:Returns:
   Angles \u03B6 (zeta), z, \u03B8 (theta) degrees
   
:Reference:
   ES 3.214 p.106
   
:Notes:
   Newcomb's precession angles for old catalogs (FK4),
   see ES 3.214 p.106.
   Input are **Besselian epochs**!
   Adopted accumulated precession angles from equator
   and equinox at B1950 to 1984 January 1d 0h according
   to ES (table 3.214.1, p 107) are:
   ``zeta=783.7092, z=783.8009 and theta=681.3883``
   The Woolard and Clemence angles (derived in this routine)
   are:
   ``zeta=783.70925, z=783.80093 and theta=681.38830``
   (see same ES table as above).
   
   This routine found (in seconds of arc):
   ``zeta,z,theta =  783.709246271 783.800934641 681.388298284``
   for ``t1 = 0.1`` and ``t2 = 0.133999566814``
   using the lines in the next example.

:Examples: From an interactive Python session:
      
            >>> b1 = 1950.0
            >>> b2 = celestial.epochs("F1984-01-01")[0]
            >>> print [x*3600 for x in celestial.Newcombprecangles(be1, be2)]
                [783.70924627097793, 783.80093464073127, 681.38829828393466]

   """
#----------------------------------------------------------------------
   t1 = (epoch1-1850.0)/1000.0    #1000 tropical years
   t2 = (epoch2-1850.0)/1000.0
   tau = t2 - t1

   d0 = 23035.545; d1 = 139.720; d2 = 0.060; d3 = 30.240; d4 = -0.27; d5 = 17.995
   a0 = d0 + t1*(d1+d2*t1); a1 = d3 + d4*t1; a2 = d5
   zeta_a = tau*(a0+tau*(a1+tau*a2))

   d0 = 23035.545; d1 = 139.720; d2 = 0.060; d3 = 109.480; d4 = 0.39; d5 = 18.325
   a0 = d0 + t1*(d1+d2*t1); a1 = d3 + d4*t1; a2 = d5
   z_a = tau*(a0+tau*(a1+tau*a2))

   d0 = 20051.12; d1 = -85.29; d2 = -0.37; d3 = -42.65; d4 = -0.37; d5 = -41.80
   a0 = d0 + t1*(d1+d2*t1); a1 = d3 + d4*t1; a2 = d5
   theta_a = tau*(a0+tau*(a1+tau*a2))
   # Return values in degrees
   return zeta_a/3600.0, z_a/3600.0, theta_a/3600.0



def rotX(angle):
   """
-----------------------------------------------------------------------
Purpose:    Calculate the matrix that represents a 3d rotation
            around the X axis.
Input:      Rotation angle in degrees
Returns:    A 3x3 matrix representing the rotation about angle around 
            X axis. 
Reference:  Diebel, J. 2006, Stanford University, Representing Attitude:
            Euler angles, Unit Quaternions and Rotation Vectors.
            http://ai.stanford.edu/~diebel/attitude.html

Notes:      Return the rotation matrix for a rotation around the X axis.
            This is a rotation in the YZ plane. Note that we construct
            a new vector with: xnew = R1.x
            In the literature, this rotation is usually called R1
-----------------------------------------------------------------------
   """
   a = d2r(angle)
   v = n.asmatrix(n.zeros((3,3), 'd'))
   cosa = n.cos(a)
   sina = n.sin(a)
   v[0,0] =  1.0;    v[0,1] =  0.0;    v[0,2] =  0.0;
   v[1,0] =  0.0;    v[1,1] =  cosa;   v[1,2] =  sina;
   v[2,0] =  0.0;    v[2,1] = -sina;   v[2,2] =  cosa;
   return v



def rotY(angle):
   """
-----------------------------------------------------------------------
Documentation in 'rotX'
Return rot. mat. for rot. around Y axis
-----------------------------------------------------------------------
   """
   a = d2r(angle)
   v = n.asmatrix(n.zeros((3,3), 'd'))
   cosa = n.cos(a)
   sina = n.sin(a)
   v[0,0] =  cosa;   v[0,1] =  0.0;     v[0,2] = -sina;
   v[1,0] =  0.0;    v[1,1] =  1.0;     v[1,2] =  0.0;
   v[2,0] =  sina;   v[2,1] =  0.0;     v[2,2] =  cosa;
   return v



def rotZ(angle):
   """
-----------------------------------------------------------------------
Documentation in 'rotX'
Return rot. mat. for rot. around Z axis
-----------------------------------------------------------------------
   """
   a = d2r(angle)
   v = n.asmatrix(n.zeros((3,3), 'd'))
   cosa = n.cos(a)
   sina = n.sin(a)
   v[0,0] =  cosa;    v[0,1] =  sina;   v[0,2] =  0.0;
   v[1,0] = -sina;    v[1,1] =  cosa;   v[1,2] =  0.0;
   v[2,0] =  0.0;     v[2,1] =  0.0;    v[2,2] =  1.0;
   return v



def fitsdate(date):
   """
-----------------------------------------------------------------------
Purpose:   Given a string from a FITS file, try to parse it and 
           convert the string into three parts: an integer year, an
           integer month and a fractional day.
Input:     A string, representing a date in FITS format
Returns:   Integer year, integer month, fractional day.
Reference: -
Notes:     Process the FITS dates as part of the 'epochs' function. 
           It processes the following formats:
           DD/MM/YY or DD/MM/19YY
           YYYY-MM-DD
           YYYY-MM-DDTHH:MM:SS
-----------------------------------------------------------------------
   """
   parts = date.split('/')
   if len(parts)==3:
      return ((int(parts[2])%1900)+1900, int(parts[1]), float(parts[0]))

   parts = date.split('T')
   if len(parts)==2:
      date = parts[0]
      parts = parts[1].split(':')
      facts = (3600.0, 60.0, 1.0)
      time = 0.0
      for i in range(len(parts)):
         time += float(parts[i])*facts[i]     
   else:
      time = 0.0
   parts = date.split('-')
   return (int(parts[0]), int(parts[1]), float(parts[2])+time/86400.0)



def epochs(spec):
#-----------------------------------------------------------------------
   """
Flexible epoch parser. The functions in this module have different
input parameters (Julian epoch, Besselian epochs, Julian dates) because
the algorithms came from different sources. What we needed was a routine
that could convert a string which represents a date in various formats,
to values for a Julian epoch, Besselian epochs and a Julian date.
This function returns these value for any valid input date.

For the epoch syntax read the documentation at :ref:`celestial-epochs`.
Note that an epoch of observation is either a second epoch in the string
(the first is always the equinox) or the epoch string has
a suffix '_' which may be follwed by arbitrary characters.

:param spec:
   An epoch specification (see below)
:type spec:
   String
   
:Returns:
   Calculated corresponding **Besselian epoch**, **Julian epoch** and **Julian date**.
   Return in order: *B, J, JD*
   
:Reference:
    Various sources listing Julian dates.
    
:Notes:

:Examples: Some checks:

   >>> celestial.epochs('F2008-03-31T8:09')  # should return:
       (2008.2474210134737, 2008.2459673739454, 2454556.8395833336)
   >>> celestial.epochs('F2007-01-14T13:18:59.9')
       (2007.0378545262108, 2007.0364267212976, 2454115.0548599539)
   >>> celestial.epochs("j2007.0364267212976")
       (2007.0378545262108, 2007.0364267212976, 2454115.0548599539)
   >>> celestial.epochs("b2007.0378545262108")
       (2007.0378545262108, 2007.0364267212976, 2454115.0548599539)
       
   """
#-----------------------------------------------------------------------

   if not spec:
      mes = "No epoch in string"
      raise Exception(mes)

   b = j = jd = None

   i = spec.find('_')
   if i != -1:
      spec = spec[:i]

   parts = re_split(r'(\d.*)', spec, 1)

   try:
      prefix = (parts[0].strip().upper())
      if prefix == 'B' or prefix == '-B':
         b = float(parts[1])
         if prefix == '-B':
            b *= -1.0
         jd = epochBessel2JD(b)
         j  = JD2epochJulian(jd)
      elif prefix == 'J' or prefix == '-J':
         j = float(parts[1])
         if prefix == '-J':
            j *= -1.0
         jd = epochJulian2JD(j)
         b  = JD2epochBessel(jd)
      elif prefix == 'JD':
         jd = float(parts[1])
         b  = JD2epochBessel(jd)
         j  = JD2epochJulian(jd)
      elif prefix == 'MJD':
         mjd = float(parts[1])
         # MJD = JD - 2400000.5
         jd = mjd + 2400000.5
         b  = JD2epochBessel(jd)
         j  = JD2epochJulian(jd)
      elif prefix == 'RJD':
         rjd = float(parts[1])
         # RJD = JD - 2400000
         jd = rjd + 2400000
         b  = JD2epochBessel(jd)
         j  = JD2epochJulian(jd)
      elif prefix == 'F':
         epoch = parts[1];
         fd = fitsdate(parts[1])
         jd = JD(fd[0], fd[1], fd[2])
         b  = JD2epochBessel(jd)
         j  = JD2epochJulian(jd)
      else:
         raise Exception("Unknown prefix for epoch")
   except:
      mes = "No prefix or cannot convert epoch to a number"
      raise Exception(mes)

   return (b, j, jd)



def MatrixEqJ20002Gal():
#-----------------------------------------------------------------------
   """
Purpose:   (Experimental) Return the rotation matrix for a transformation
           between equatorial (FK5, J2000) and galactic IAU 1958 
	   coordinate systems. This function is not used because it could
           be composed of two fundamental transformations.
Input:     -
Returns:   Matrix M as in: XYZgal = M * XYZj2000
Reference:-Murray, C.A. The Transformation of coordinates between the 
           systems B1950.0 and J2000.0, and the principal galactic axes
	   referred to J2000.0, 
           Astronomy and Astrophysics (ISSN 0004-6361), vol. 218, no. 1-2, 
	   July 1989, p. 325-329.
          -Blaauw, A., Gum C.S., Pawsey, J.L., Westerhout, G.: 1958, 
	   Monthly Notices Roy. Astron. Soc. 121, 123
Notes:     The position of the galactic pole is defined in the fk4, B1950
           system (without e-terms).
           For a position in fk5 J2000 one could consider to create a
           rotation matrix based on J2000 coordinates of the galactic pole.

           192.85948121     -RA of galactic north pole (mean b1950.0)
           27.12825118      -Dec of galactic north pole
           122.93191857     -Galactic longitude of celestial equator

           >>> print celestial.sky2sky(celestial.fk4_no_e, celestial.fk5,192.25,27.4)
               [[ 192.85948121   27.12825118]]
           >>> print celestial.sky2sky( celestial.fk5, celestial.gal, 0,90)
               [[ 122.93191857   27.12825118]]

           According to the Hipparcos explanatory supplement the angles
           in J2000 are:
           192.85948               Right Ascension of Galactic North Pole
           27.12825                Declination of Galactic North Pole
           32.93192                Galactic longitude of celestial equator  

           HOWEVER:
           Murray (1989) however objects against the transformation of 
           these principal directions because in the J2000 system the 
           axes are not orthogonal, which is unacceptable for a transformation.
           Therefore the
           transformation from fk5 to galactic is calculated in two steps. 
           First a position is transformed to fk4 (no e-terms) and then 
           to a galactic coordinate (lII, bII) 
           The result matrix in celestial.py is calculated with: 
           skymatrix((eq,"J2000.0",fk5),gal)
           and produces the numbers:
           [[-0.054875539396 -0.873437104728 -0.48383499177 ]
            [ 0.494109453628 -0.444829594298  0.7469822487  ]
            [-0.867666135683 -0.198076389613  0.455983794521]]
           which are all consistent with equation (33) in Murray, 1989.

           If, on the other hand we calculate the rotation matrix for the J2000
           coordinates:
           >>> R = rotZ(180-122.93191857)*rotY(90-27.12825118)*rotZ(192.85948121)
           >>> print skymatrix((eq,"J2000.0",fk5),gal)[0] - R
               [[ -4.26766400e-11  -1.39604994e-11   3.00424130e-11]
                [ -9.72683045e-12   4.29156710e-12   8.98969787e-12]
                [ -2.84006152e-12   5.19224108e-11   1.71504477e-11]]

           then we cannot conclude that these different methods differ
           significantly.

           In the 2MASS All-Sky Data Release Explanatory Supplement:
           we read:
          'There is an ambiguity in the appropriate way to convert J2000
           ICRS coordinates to the galactic system. Galactic coordinates
           could be derived by precessing J2000.0 coordinates to B1950, 
           then using the rotation transformations into the lII,bII 
           system (as in MatrixEqB19502Gal(), VOG). 
           This transformation method produces galactic coordinates
           that can differ 
           by up to 0.4'' from those, e.g., produced using the direct
           J2000-to-galactic transformations, proposed by Murray 
           (1989, AsAp, 218, 325).'

           Murray's matrix however is composed of the transformation fk5
           to fk4 without e-terms and fk4 without e-terms to Galactic. 
           So the differences can only be explained by 
           wrongly adding e-terms in fk4 before transforming these to
           galactic coordinates.
   """
#-----------------------------------------------------------------------
   M1 = FK52FK4Matrix()
   M2 = MatrixEqB19502Gal()
   return M2*M1



def MatrixEqB19502Gal():
#-----------------------------------------------------------------------
   """
Create matrix to convert equatorial fk4 coordinates
(without e-terms) to IAU 1958 lII,bII system of
galactic coordinates.

:Parameters:
   None
   
:Results:
   3x3 Matrix M as in XYZgal = M * XYZb1950

:Reference:
   
   1. Blaauw, A., Gum C.S., Pawsey, J.L., Westerhout, G.: 1958,
   2. Monthly Notices Roy. Astron. Soc. 121, 123,
   3. Blaauw, A., 2007. Private communications.

:Notes:
   Original definitions from 1.:
            
   *  The new north galactic pole lies in the direction
      alpha = 12h49m (192.25 deg),
      delta=27.4 deg (equinox 1950.0).
   *  The new zero of longitude is the great semicircle
      originating at the new north galactic pole at the
      position angle theta = 123 deg with respect
      to the equatorial pole for 1950.0.
   *  Longitude increases from 0 to 360 deg. The sense is
      such that, on the galactic equator increasing galactic
      longitude corresponds to increasing Right Ascension.
      Latitude increases from -90 deg through 0 deg to 90 deg
      at the new galactic pole.

   Given the RA and Dec of the galactic pole, and using the
   Euler angles scheme::

      M = rotZ(a3).rotY(a2).rotZ(a1)

   We first rotate the spin vector of the XY plane about
   an angle a1 = ra_pole and then rotate the spin vector
   in the XZ plane (i.e. around the Y axis) with an angle
   a2=90-dec_pole to point it in the right declination.

   Now think of a circle with the galactic pole as its center.
   The radius is equal to the distance between this center
   and the equatorial pole. The zero point now is on the circle
   and opposite to this pole.
   
   We need to rotate along this circle (i.e. a rotation
   around the new Z-axis) in a way that the angle between the
   zero point and the equatorial pole is equal to 123 deg.
   So first we need to compensate for the 180 deg of the
   current zero longitude, opposite to the pole. Then we need
   to rotate about an angle 123 deg but in a way that increasing
   galactic longitude corresponds to increasing Right Ascension
   which is opposite to the standard rotation of this circle
   (note that we rotated the original X axis about 192.25 deg).
   The last rotation angle therefore is a3=+180-123::

     M = rotZ(180-123.0)*rotY(90-27.4)*rotZ(192.25)

   The composed rotation matrix is the same as in Slalib's 'ge50.f'
   and the matrix in eq. (32) of Murray (1989).
   """
#-----------------------------------------------------------------------

   return rotZ(180-123.0)*rotY(90-27.4)*rotZ(192.25)
   # Alternative: rotZ(-33.0)*rotX(62.6)*rotZ(90+192.25)




def MatrixGal2Sgal():
#-----------------------------------------------------------------------
   """
Transform galactic to supergalactic coordinates

:Parameters:      
   None
   
:Returns:   
   Matrix M as in XYZsgal = M * XYZgal

:Reference:  
   Lahav, O., The supergalactic plane revisited with the 
   Optical Redshift Survey
   Mon. Not. R. Astron. Soc. 312, 166-176 (2000)

:Notes:      
   The Supergalactic equator is conceptually defined by the 
   plane of the local (Virgo-Hydra-Centaurus) supercluster,
   and the origin of supergalactic longitude is at the
   intersection of the supergalactic and galactic planes. 
   (de Vaucouleurs) 
   
   North SG pole at l=47.37 deg, b=6.32 deg. 
   Node at l=137.37, sgl=0 (inclination 83.68 deg).

   Older references give for he position of the SG node 137.29
   which differs from 137.37 deg in the official definition.

   For the rotation matrix we chose the scheme *Rz.Ry.Rz*
   Then first we rotate about 47.37 degrees along the Z-axis
   followed by a rotation about 90-6.32 degrees is needed to
   set the pole to the right declination.
   The new plane intersects the old one at two positions.
   One of them is l=137.37, b=0 (in galactic coordinates).
   If we want this to be sgl=0 we have to rotate this plane along
   the new Z-axis about an angle of 90 degrees. So the composed
   rotation matrix is::
   
      M = Rotz(90)*Roty(90-6.32)*Rotz(47.37)
   """
#----------------------------------------------------------------------
   # Alternative rotX(90-6.32)*rotZ(90+47.37)
   return rotZ(90.0)*rotY(90-6.32)*rotZ(47.37)




def MatrixEq2Ecl(epoch, S1):
#----------------------------------------------------------------------
   """
Calculate a rotation matrix to convert equatorial 
coordinates to ecliptical coordinates

:param epoch:
   Epoch of the equator and equinox of date 
:type epoch: Floating point number
   
:param S1:
   equatorial system to determine if one entered epoch in
   B or J coordinates.
:type S1:
   Integer

:Returns:  
   3x3 Matrix M as in XYZecl = M * XYZeq

:Reference: 
   Representations of celestial coordinates in FITS, 
   Calabretta. M.R., & Greisen, E.W., (2002)
   Astronomy & Astrophysics,  395,  1077-1122.
   http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf

:Notes:     
   1. The origin for ecliptic longitude is the vernal equinox.
      Therefore the coordinates of a fixed object is subject to 
      shifts due to precession. The rotation matrix 
      uses the obliquity to do the conversion to the wanted ecliptic 
      coordinates.
      So we always need to enter an epoch. Usually this is J2000,
      but it can also be the epoch of date. The additional reference
      system indicates whether we need a Besselian or a Julian 
      epoch.

   2. In the FITS paper of Calabretta and Greisen (2002), one 
      observes the following relations to FITS:
   
      -Keyword RADESYSa sets the catalog system FK4, FK4-NO-E or FK5
      This applies to equatorial and ecliptical coordinates with 
      the exception of FK4-NO-E.
   
      -FK4 coordinates are not strictly spherical since they include 
      a contribution from the elliptic terms of aberration, the 
      so-called e-terms which amount to max. 343 milliarcsec. 
      FITS paper: *'Strictly speaking, therefore, a map obtained from, 
      say, a radio synthesis telescope, should be regarded
      as FK4-NO-E unless it has been appropriately re-sampled
      or a distortion correction provided.
      In common usage, however, CRVALia for such maps is usually 
      given in FK4 coordinates. In doing so, the e-terms are effectively
      corrected to first order only.'*. (See also ES, eq. 3.531-1 page 170.
   
      -Keyword EQUINOX sets the epoch of the mean equator and equinox.
   
      -Keyword EPOCH is often used in older FITS files. It is a deprecated keyword
      and should be replaced by EQUINOX.
      It does not require keyword RADESYS. From its value we derive
      whether the reference system is FK4 or FK5 (the marker value is 1984.0)
   
      -Ecliptic coordinates require the epoch of the equator and equinox
      of date.
      This will be taken as the time of observation rather than
      EQUINOX. 
      
      FITS paper: *'The time of observation may also be required for
      other astrometric purposes in addition to the usual astrophysical
      uses, for example, to specify when the mean place was
      correct in accounting for proper motion, including "fictitious"
      proper motions in the conversion between the FK4 and FK5 systems.
      The old *DATE-OBS* keyword may be used for this purpose.
      However, to provide a more convenient specification we
      here introduce the new keyword MJD-OBS'.*
      
      So MJD-OBS is the modified Julian Date (JD - 2400000.5) of the
      start of the observation.

   3. Equatorial to ecliptic transformations use the time dependent 
      obliquity of the equator (also known as the obliquity of the ecliptic).
      Again, start with::
      
         M = rotZ(0).rotX(eps).rotZ(0) = E.rotX(eps).E = rotX(eps)
      
      In fact this is only a rotation around the X axis
   """
#----------------------------------------------------------------------
   if (S1 == fk4):
      jd = epochBessel2JD(epoch)
   else:                 # For all other systems the epochs are Julian
      jd = epochJulian2JD(epoch)
   if (S1 == icrs or S1 == j2000):
      eps = obliquity2000(jd)
   else:
      eps = obliquity1980(jd)
   return rotX(eps)




def getEterms(epoch):
#----------------------------------------------------------------------
   """
Compute the E-terms (elliptic terms of aberration) for a given epoch.

:param epoch:
   A **Besselian** epoch
:type epoch:
   Floating point number
   
:Returns:   
   A tuple containing the e-terms vector 
   *(DeltaD,DeltaC,DeltaC.tan(e0))*

:Reference: 
   Seidelman, P.K.,  1992.  Explanatory Supplement to the Astronomical
   Almanac.  University Science Books, Mill Valley

:Notes:     
   The method is described on page 170/171 of the ES.
   One needs to process the e-terms for the appropriate
   epoch This routine returns the e-term vector for arbitrary epoch.
   """
#----------------------------------------------------------------------
   # Julian centuries since B1950
   T = (epoch-1950.0)*1.00002135903/100.0
   # Eccentricity of the Earth's orbit
   ec = 0.01673011-(0.00004193+0.000000126*T)*T
   # Mean obliquity of the ecliptic. Method is different compared to 
   # functions for the obliquity defined earlier. This function depends
   # on time wrt. epoch 1950 not epoch 2000.
   ob = (84404.836-(46.8495+(0.00319+0.00181*T)*T)*T)
   ob = d2r(ob/3600.0)
   # Mean longitude of perihelion of the solar orbit
   p = (1015489.951+(6190.67+(1.65+0.012*T)*T)*T)
   p = d2r(p/3600.0)
   # Calculate the E-terms vector
   ek = ec*d2r(20.49522/3600.0)         # 20.49552 is constant of aberration at J2000
   cp = n.cos(p)
   #       -DeltaD        DeltaC            DeltaC.tan(e0)
   return (ek*n.sin(p), -ek*cp*n.cos(ob), -ek*cp*n.sin(ob))



def addEterms(xyz, a=None):
#----------------------------------------------------------------------
   """
Add the elliptic component of annual aberration when the
result must be a catalogue fk4 position.

:param xyz:
   Cartesian position(s) converted from 
   lonlat = [ (a1,d1),(a2,d2), ..., (an,dn) ] -->
   xyz = [ (x1,y1,z1), (x2,y2,z2), ..., (xn,yn,zn) ]
:type xyz:
   NumPy (n,2) matrix
:param a:
   E-terms vector (as returned by getEterms())
   If input *a* is omitted (i.e. *a == None*), the e-terms for
   1950 will be substituted.
:type a:
   Tuple with 3 floating point numbers

:Result:
   **Apparent place**, NumPy (n,2) matrix 

:Reference: 
   * Seidelman, P.K.,  1992.  Explanatory Supplement to the Astronomical
     Almanac.  University Science Books, Mill Valley.
   * Yallop et al, Transformation of mean star places,
     AJ, 1989, vol 97, page 274
   * Stumpff, On the relation between Classical and Relativistic
     Theory of Stellar Aberration, 
     Astron, Astrophys, 84, 257-259 (1980)

:Notes:     
   There is a so called ecliptic component in the stellar aberration.
   This vector depends on the epoch at which we want to process
   these terms. It corresponds to the component of the earth's velocity
   perpendicular to the major axis of the ellipse in the ecliptic.
   The E-term corrections are as follows. A catalog FK4 position
   include corrections for elliptic terms of aberration. 
   These positions are apparent places. For precession and/or 
   rotations to other sky systems, one processes only mean places.
   So to get a mean place, one has to remove the E-terms vector.
   The ES suggests for the removal to use a decompositions of the
   E-term vector along the unit circle to get the approximate 
   new vector, which has almost the correct angle and has almost 
   length 1. The advantage is that when we add the E-term vector 
   to this new vector, we obtain a new vector with the original 
   angle, but with a length unequal to 1, which makes it suitable
   for closure tests.
   However, the procedure can be made more rigorous: 
   For the subtraction we subtract the E-term vector from the 
   start vector and normalize it afterwards. Then we have an
   exact new angle (opposed to the approximation in the ES).
   The procedure to go from a vector in the mean place system to 
   a vector in the system of apparent places is a bit more 
   complicated:
   Find a value for lambda so that the current vector is
   adjusted in length so that adding the e-term vector gives a new
   vector with length 1. This is by definition the new vector
   with the right angle. For more information, see the background
   information in :doc:`celestialbackground`.
   """
#----------------------------------------------------------------------

   xyzeterm = xyz.copy()
   if a == None:
      a = getEterms(1950.0)
   for i in range(xyz.shape[1]):         # Loop over all vectors
      x = xyz[0,i]; y = xyz[1,i]; z = xyz[2,i]
      # Normalize to get a vector of length 1. Our algorithm is based on that fact.
      d = n.sqrt(x*x + y*y + z*z)
      x /= d; y /= d; z /= d
      # Find the lambda to stretch the vector 
      w = 2.0 * (a[0]*x + a[1]*y + a[2]*z)
      p = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] - 1.0
      lambda1 = (-w + n.sqrt(w*w-4.0*p))/2.0     # Vector a is small. We want only the positive lambda 
      xyzeterm[0,i] = lambda1*x + a[0]
      xyzeterm[1,i] = lambda1*y + a[1] 
      xyzeterm[2,i] = lambda1*z + a[2] 

   return xyzeterm



def removeEterms(xyz, a=None):
#----------------------------------------------------------------------
   """
Remove the elliptic component of annual aberration when this
is included in a catalogue fk4 position.

:param xyz:
   Cartesian position(s) converted from 
   lonlat = [ (a1,d1),(a2,d2), ..., (an,dn) ] -->
   xyz = [ (x1,y1,z1), (x2,y2,z2), ..., (xn,yn,zn) ]
:type xyz:
   NumPy (n,2) matrix
:param a:
   E-terms vector (as returned by getEterms())
   If input a is omitted (== *None*), the e-terms for
   1950 will be substituted.
:type a:
   Tuple with 3 floating point numbers

:Result: 
   **Mean place**, NumPy (n,2) matrix

:Notes:
   Return a new position where the elliptic terms of aberration 
   are removed i.e. convert a apparent position from a catalog to
   a mean place.
   The effects of ecliptic aberration were included in the 
   catalog positions to facilitate telescope pointing.
   See also notes at 'addEterms'.

   """
#----------------------------------------------------------------------
   xyzeterm = xyz.copy()
   if a == None:
      a = getEterms(1950.0)
   # a(1950) should be:  = n.array([-1.62557e-6, -0.31919e-6, -0.13843e-6])
   for i in range(xyz.shape[1]):            # Loop over all vectors data
      x = xyz[0,i]; y = xyz[1,i]; z = xyz[2,i]
      x -= a[0]; y -= a[1]; z -= a[2]
      xyzeterm[0,i] = x
      xyzeterm[1,i] = y
      xyzeterm[2,i] = z

   return xyzeterm



def precessionmatrix(zeta, z, theta):
   """
---------------------------------------------------------------------- 
Purpose:   Given three precession angles, create the corresponding 
           rotation matrix
Input:     zeta, z, theta
Returns:   Rotation matrix M as in XYZepoch1 = M * XYZepoch2
Notes:     Return the precession matrix for the three precession angles 
           zeta, z and theta.
           Rotation matrix: R = rotZ(-z).rotY(th).rotZ(-zeta) (ES 3.21-7, p 103)
           Also allowed is the expression: rotZ(-90-z)*rotX(th)*rotZ(90-zeta)
---------------------------------------------------------------------- 
   """
   return rotZ(-z)*rotY(theta)*rotZ(-zeta)




def IAU2006MatrixEpoch12Epoch2(epoch1, epoch2):
#----------------------------------------------------------------------
   """
Create a rotation matrix for a precession based on 
IAU 2000/2006 expressions, see function :func:`IAU2006precangles`

:param epoch1:
   Julian start epoch
:type epoch1:
   Floating point number
:param epoch2:
    Julian epoch to precess to.
:type epoch2:
   Floating point number

:Returns:   
   Matrix to transform equatorial coordinates from epoch1 to 
   epoch2 as in XYZepoch2 = M * XYZepoch1

:Reference:
   Capitaine N. et al.: IAU 2000 precession A&A 412, 567-586 (2003)

:Notes:     
   Note that we apply this precession only to equatorial
   coordinates in the system of dynamical J2000 coordinates.
   When converting from ICRS coordinates this means applying 
   a frame bias. 
   Therefore the angles differ from the precession 
   Fukushima-Williams angles (IAU 2006)
   
   The precession matrix is::
    
      M = rotZ(-z).rotY(+theta).rotZ(-zeta)
   """ 
#----------------------------------------------------------------------
   if (epoch1 == epoch2):
      return I()
   if epoch1 == 2000.0:
      zeta, z, theta = IAU2006precangles(epoch2)
      return precessionmatrix(zeta, z, theta)
   elif epoch2 == 2000.0:
      zeta, z, theta = IAU2006precangles(epoch1)
      return (precessionmatrix(zeta, z, theta)).T
   else:
      # If both epochs are not J2000.0
      zeta, z, theta = IAU2006precangles(epoch1)
      M1 = (precessionmatrix(zeta, z, theta)).T
      zeta, z, theta = IAU2006precangles(epoch2)
      M2 = precessionmatrix(zeta, z, theta)
      return M2*M1



def BMatrixEpoch12Epoch2(Bepoch1, Bepoch2):
#----------------------------------------------------------------------
   """
Precession from one epoch to another in the fk4 system.
It uses :func:`Newcombprecangles` to calculate the 
precession angles.


:param Bepoch1:
   Besselian start epoch
:type Bepoch1:
   Floating point number
:param Bepoch2:
    Besselian epoch to precess to.
:type Bepoch2:
   Floating point number

:Returns:   
   3x3 rotation matrix M as in XYZepoch2 = M * XYZepoch1

:Reference: 
   Seidelman, P.K.,  1992.  Explanatory Supplement to the Astronomical
   Almanac.  University Science Books, Mill Valley. 3.214 p 106

:Notes:     
   The precession matrix is::
    
    M = rotZ(-z).rotY(+theta).rotZ(-zeta)
   
   """
#----------------------------------------------------------------------
   zeta, z, theta = Newcombprecangles(Bepoch1, Bepoch2)
   return precessionmatrix(zeta, z, theta)



def JMatrixEpoch12Epoch2(Jepoch1, Jepoch2):
#----------------------------------------------------------------------
   """
Precession from one epoch to another in the fk5 system.
It uses :func:`Lieskeprecangles` to calculate the 
precession angles.

:param Jepoch1:
   Julian start epoch
:type Jepoch1:
   Floating point number
:param Jepoch2:
    Julian epoch to precess to.
:type Jepoch2:
   Floating point number

:Returns:   
   3x3 rotation matrix M as in XYZepoch2 = M * XYZepoch1

:Reference: 
   Seidelman, P.K.,  1992.  Explanatory Supplement to the Astronomical
   Almanac.  University Science Books, Mill Valley. 3.214 p 106

:Notes:     
   The precession matrix is::
    
     M = rotZ(-z).rotY(+theta).rotZ(-zeta)

   """
#----------------------------------------------------------------------
   jd1 = epochJulian2JD(Jepoch1)
   jd2 = epochJulian2JD(Jepoch2)
   zeta, z, theta = Lieskeprecangles(jd1, jd2)
   return precessionmatrix(zeta, z, theta) 




def FK42FK5Matrix(t=None):
#----------------------------------------------------------------------
   """
Create a matrix to precess from B1950 in FK4 to J2000 in FK5 
following to Murray's (1989) procedure.

:param t:
   Besselian epoch as epoch of observation.
:type t:
   Floating point number
   
:Returns:   
   3x3 matrix M as in XYZfk5 = M * XYZfk4

:Reference: 
   * Murray, C.A. The Transformation of coordinates between the 
     systems B1950.0 and J2000.0, and the principal galactic axis 
     referred to J2000.0, 
     Astronomy and Astrophysics (ISSN 0004-6361), vol. 218, no. 1-2, 
     July 1989, p. 325-329.
   * Poppe P.C.R.,, Martin, V.A.F., Sobre as Bases de Referencia Celeste
     SitientibusSerie Ciencias Fisicas

:Notes:     
   Murray precesses from B1950 to J2000 using a precession matrix
   by Lieske. Then applies the equinox correction and ends up with a
   transformation matrix *X(0)* as given in this function.

   In Murray's article it is proven that using the procedure as
   described in the article,  ``r_fk5 = X(0).r_fk4`` for extra galactic
   sources where we assumed that the proper motion in FK5 is zero.
   This procedure is independent of the epoch of observation.
   Note that the matrix is not a rotation matrix.

   FK4 is not an inertial coordinate frame (because of the error
   in precession and the motion of the equinox. This has 
   consequences for the proper motions. e.g. a source with zero
   proper motion in FK5 has a fictitious proper motion in FK4.
   This affects the actual positions in a way that the correction
   is bigger if the epoch of observation is further away from 1950.0
   The focus of this library is on data of which we do not have
   information about the proper motions. So for positions of which
   we allow non zero proper motion in FK5 one needs to supply the
   epoch of observation.
   
:Examples: 
   Print the difference between the rotation matrix for 1970 and 
   1980:
   
      >>> M1 = celestial.FK42FK5Matrix(1970)
      >>> M2 = celestial.FK42FK5Matrix(1980)
      >>> M2 - M1
      matrix([[ -2.64546940e-10,  -1.15396722e-07,   2.11108953e-07],
              [  1.15403817e-07,  -1.29040234e-09,   2.36016437e-09],
              [ -2.11125281e-07,  -5.60232514e-10,   1.02585540e-09]])

           
   """
#----------------------------------------------------------------------
   r11 = 0.9999256794956877; r12 = -0.0111814832204662; r13 = -0.0048590038153592
   r21 = 0.0111814832391717; r22 =  0.9999374848933135; r23 = -0.0000271625947142
   r31 = 0.0048590037723143; r32 = -0.0000271702937440; r33 =  0.9999881946023742
  
   if t != None:  # i.e. we also assuming that v != 0 in FK5 !!
      jd = epochBessel2JD(t)
      T = (jd-2433282.423)/36525.0    # t-1950 in Julian centuries = F^-1.t1 from Murray (1989)
      r11 += -0.0026455262*T/1000000.0
      r12 += -1.1539918689*T/1000000.0
      r13 +=  2.1111346190*T/1000000.0
      r21 +=  1.1540628161*T/1000000.0
      r22 += -0.0129042997*T/1000000.0
      r23 +=  0.0236021478*T/1000000.0
      r31 += -2.1112979048*T/1000000.0
      r32 += -0.0056024448*T/1000000.0
      r33 +=  0.0102587734*T/1000000.0
   return n.matrix( ([r11,r12,r13],[r21,r22,r23],[r31,r32,r33]) )



def FK42FK5MatrixAOKI():
   """
----------------------------------------------------------------------
Experimental.
Create matrix to precess from B1950 in FK4 to J2000 in FK5
The method is described in section 3.59 of the ES. 
Proper motions are not taken into account. Parallax and radial velocity
are set to zero and not taken into account.
We do not repeat the procedures here, but copy part of the matrix from 
ES, 3.591-4, p 185
See also reference below:
Author(s): Aoki, S., Soma, M., Kinoshita, H., Inoue, K.
Title:	   Conversion matrix of epoch B 1950.0 FK4-based positions of 
           stars to epoch J 2000.0 positions in accordance with 
           the new IAU resolutions
Source:	   Astron. Astrophys. 128, 263-267
Year:	   1983

The matrix in the Yallop (1989) article has more digits than the
matrix from the ES.
Yallop, B.D. et al, 1989.  "Transformation of mean star places
from FK4 B1950.0 to FK5 J2000.0 using matrices in 6-space".
Astron.J. 97, 274.
----------------------------------------------------------------------
   """
   r0 = [0.999925678186902, -0.011182059642247, -0.004857946558960]
   r1 = [0.011182059571766, 0.999937478448132, -0.00002717441185]
   r3 = [0.004857946721186, -0.000027147426498, 0.999988199738770]
   return n.matrix( (r0,r1,r3) )



def FK42FK5MatrixLOWPREC():
   """
----------------------------------------------------------------------
Experimental.
Create matrix to precess from B1950 in FK4 to J2000 in FK5
The method is described in section 3.59 of the ES. 
Proper motions are not taken into account. Parallax and radial velocity
are set to zero and not taken into account.
We do not repeat the procedures here, but copy part of the matrix from 
ES, 3.591-4, p 185
See also reference below:
Author(s): Aoki, S., Soma, M., Kinoshita, H., Inoue, K.
Title:	   Conversion matrix of epoch B 1950.0 FK4-based positions of 
           stars to epoch J 2000.0 positions in accordance with 
           the new IAU resolutions
Source:	   Astron. Astrophys. 128, 263-267
Year:	   1983
----------------------------------------------------------------------
   """

   r0 = [0.9999256782, -0.0111820611, -0.0048579477]
   r1 = [0.0111820610, 0.9999374784, -0.0000271765]
   r3 = [0.0048579479, -0.0000271474, 0.9999881997]
   return n.matrix( (r0,r1,r3) )




def FK52FK4Matrix(t=None):
   """
----------------------------------------------------------------------
Purpose:   Create a matrix to convert a position in fk5 to fk4 using 
           the inverse matrix FK42FK5Matrix
Input:     Epoch of observation for those situations where we allow
           no-zero proper motion in fk4
Returns:   Rotation matrix M as in XYZfk5 = M * XYZfk4
Notes:     For this matrix we know that the inverse is not the
           transpose.
----------------------------------------------------------------------
   """
   return FK42FK5Matrix(t).I



def FK42FK5MatrixOLDATTEMPT():
   """
----------------------------------------------------------------------
Experimental.
Create matrix to precess from an epoch in FK4 to an epoch in FK5
So epoch1 is Besselian and epoch2 is Julian
1) Do an epoch transformation in FK4 from input epoch to 
   1984 January 1d 0h
2) Apply a zero point correction for the right ascension
   w.r.t. B1950. The formula is:
   E = E0 + E1*(jd-jd1950)/Cb
   E0 = 0.525;  E1 = 1.275 and Cb = the length of the tropical 
   century (ES 3.59 p 182) = 36524.21987817305
   For the correction at 1984,1,1 the ES lists 0.06390s which is
   0.06390*15=0.9585"
   This function calculated E = 0.958494476885" which agrees with the 
   literature.
3) Transform in FK5 from 1984 January 1d 0h to epoch2

Note that we do not use the adopted values for the precession angles, 
but use the Woolward and Clemence expressions to calculate the angles.
These are one digit more accurate than the adopted values.
----------------------------------------------------------------------
   """
   # Epoch transformation from B1950 to 1984, 1,1 in FK4
   jd = JD(1984,1,1)
   epoch1984 = JD2epochBessel(jd)
   M1 = BMatrixEpoch12Epoch2(1950.0, epoch1984)

   # Equinox correction to the right ascension
   jd1950 = epochBessel2JD(1950.0)
   E0 = 0.525;  E1 = 1.275
   Cb = 36524.21987817305          # In days = length of the tropical century
   E = E0 + E1*(jd-jd1950)/Cb

   E /= 3600.0                     # From seconds of arc to degree
   M2 = rotZ(-E)                   # The correction is positive so we have to rotate
                                   # around the z-axis in the negative direction.
   # Epoch transformation from 1984,1,1 to J2000
   epoch1984 = JD2epochJulian(jd)
   M3 = JMatrixEpoch12Epoch2(epoch1984, 2000.0)

   return M3*M2*M1



def addpropermotion(xyz):
   """
----------------------------------------------------------------------
Experimental.
Input is a Cartesian position xyz.
Return a new position where the input position is corrected for 
assumed proper motion in the FK4 system.
For convenience we assume the epoch of observation is 1950
----------------------------------------------------------------------
   """
   twopi=6.283185307179586476925287
   pmf = 100.0*60*60*360/twopi

   d = 1950.0
   mjd = 15019.81352 + (d-1900)*365.242198781   # Convert to Modified Julian date
   Julianepoch = 2000.0 + (mjd-51544.5)/365.25  # Convert this mjd to Julian epoch
   w = (Julianepoch-2000.0)/pmf                 # Correction factor

   xyzpm = xyz.copy()
   r0 = [-0.000551, +0.238514, -0.435623]         # Matrix from the ES.
   r1 = [-0.238565, -0.002667, +0.012254]
   r2 = [+0.435739, -0.008541,  +0.002117]
   M = n.matrix( (r0,r1,r2) )

   for i in range(xyz.shape[1]):            # Loop over all vectors
      p = n.array([xyz[0,i], xyz[1,i], xyz[2,i]]).T
      v = p.copy()
      v[0] = r0[0]*p[0]+r0[1]*p[1]+r0[2]*p[2]
      v[1] = r1[0]*p[0]+r1[1]*p[1]+r1[2]*p[2]
      v[2] = r2[0]*p[0]+r2[1]*p[1]+r2[2]*p[2]
      for j in range(2):
         xyzpm[j,i] = p[j] + w * v[j]
   return xyzpm



def EquinoxCorrection():
   """
----------------------------------------------------------------------
Experimental.
Purpose: Calculate the equinox correction according to Murray 
----------------------------------------------------------------------
   """
   F = 1.000021359027778       #Converts the rate of change of Newcomb's precession from tropical centuries to Julian centuries.
   jd1 = epochs('B1950.0')[2]
   jd2 = epochs('J2000.0')[2]
   juliancenturies = (jd2-jd1) / 36525.0  # 1 Julian century is 36525 days
   E0 = 0.525;  E1 = 1.275
   E = E0 + 0.0 * juliancenturies * F
   # print ", Juliancenturies, Juliancenturies-0.500002095577002 (Murray)", juliancenturies, juliancenturies-0.500002095577002
   E /= 3600.0                     # From seconds of arc to degree
   M = rotZ(-E)                    # The correction is positive so we have to rotate
                                   # around the z-axis in the negative direction.
   return M



def ICRS2FK5Matrix():
#----------------------------------------------------------------------
   """
Create a rotation matrix to convert a position from ICRS to fk5, J2000

:Parameters:
   None

:Returns:    
   3x3 rotation matrix M as in XYZfk5 = M * XYZicrs

:Reference:  
   Kaplan G.H., The IAU Resolutions on Astronomical Reference 
   systems, Time scales, and Earth Rotation Models, US Naval 
   Observatory, Circular No. 179

:Notes:      
   Return a matrix that converts a position vector in ICRS
   to FK5, J2000.
   We do not use the first or second order approximations
   given in the reference, but use the three rotation matrices
   from the same paper to obtain the exact result::
   
      M =  rotX(-eta0)*rotY(xi0)*rotZ(da0)

   eta0 = -19.9 mas, xi0 = 9.1 mas and da0 = -22.9 mas

   """
#----------------------------------------------------------------------
   eta0 = -19.9/(3600*1000)  # Convert mas to degree
   xi0 = 9.1/(3600*1000)
   da0 = -22.9/(3600*1000)
   return rotX(-eta0)*rotY(xi0)*rotZ(da0)
 


def ICRS2J2000Matrix():
#----------------------------------------------------------------------
   """
Return a rotation matrix for conversion of a position in the 
ICRS to the dynamical reference system based on the dynamical
mean equator and equinox of J2000.0 (called the dynamical
J2000 system) 

:Parameters:
   None
   
:Returns:   
   Rotation matrix to transform positions from ICRS to dyn J2000

:Reference: 
   * Hilton and Hohenkerk (2004), Astronomy and Astrophysics 
     413, 765-770
   * Kaplan G.H., The IAU Resolutions on Astronomical Reference
     systems, Time scales, and Earth Rotation Models, 
     US Naval Observatory, Circular No. 179

:Notes:     
   Return a matrix that converts a position vector in ICRS
   to Dyn. J2000. We do not use the first or second order
   approximations given in the reference, but use the three 
   rotation matrices to obtain the exact result::

      M = rotX(-eta0)*rotY(xi0)*rotZ(da0)

   eta0 = -6.8192 mas, xi0 = -16.617 mas and da0 = -14.6 mas

   """
#----------------------------------------------------------------------
   eta0 = -6.8192/(3600*1000)  # Convert mas to degree
   xi0  = -16.617/(3600*1000)
   da0  = -14.6/(3600*1000)
   return rotX(-eta0)*rotY(xi0)*rotZ(da0)



def MatrixEpoch12Epoch2(epoch1, epoch2, S1, S2, epobs=None):
#----------------------------------------------------------------------
   """
Helper function for :func:`skymatrix`. It handles precession and
the transformation between **equatorial** systems. This function
includes also conversions between reference systems.

:param epoch1:
   Epoch belonging to system S1 depending on the reference 
   system either Besselian or Julian.
:type epoch1:
   Floating point number
:param epoch2:
   Epoch belonging to system S2 depending on the reference 
   system either Besselian or Julian.
:param S1:
   Input reference system
:type S1:
   Integer
:param S2:
   Output rreferencesystem
:type S2:
   Integer
:param epobs:
   Epoch of observation. Only valid for conversions between
   FK4 and FK5.
:type epobs:
   Floating point number
   
:Returns:   
   Rotation matrix to transform a position in one of the 
   reference systems *S1* with *epoch1* to an equatorial system 
   with equator and equinox at *epoch2* in reference system *S2*.

:Notes:     
   Return matrix to transform equatorial coordinates from
   *epoch1* to *epoch2* in either reference system FK4 or FK5. 
   Or transform from epoch, FK4 or FK5 to ICRS or J2000 vice versa.
   Note that each transformation between FK4 and one of the
   other reference systems involves a conversion to
   FK5 and therefore the epoch of observation will be involved.
   
   Note that if no systems are entered and the one
   epoch is > 1984 and the other < 1984, then the
   transformation involves both sky reference systems FK4
   and FK5.

:Examples: 
   Calculate rotation matrix for a conversion between FK4, epoch 1940
   to FK5, epoch 1960, while the date of observation was 1950.
           
      >>> from kapteyn import celestial
      >>> celestial.MatrixEpoch12Epoch2(1940, 1960, celestial.fk4, celestial.fk5, 1950)
      matrix([[  9.99988107e-01,  -4.47301372e-03,  -1.94362889e-03],
              [  4.47301372e-03,   9.99989996e-01,  -4.34712255e-06],
              [  1.94362889e-03,  -4.34680782e-06,   9.99998111e-01]])

   """
#----------------------------------------------------------------------
   # note that if S1 or S2 is equal to ICRS, then corresponding epoch is irrelevant
   if (S1==fk5 and S2==fk5):
      return JMatrixEpoch12Epoch2(epoch1, epoch2)
   elif (S1==fk4 and S2==fk4):
      return BMatrixEpoch12Epoch2(epoch1, epoch2)
   elif (S1==fk4 and S2==fk5):
      M1 = BMatrixEpoch12Epoch2(epoch1, 1950.0)
      M2 = FK42FK5Matrix(epobs)
      M3 = JMatrixEpoch12Epoch2(2000.0, epoch2)
      return M3*M2*M1
   elif (S1==fk5 and S2==fk4):
      M1 = JMatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = FK52FK4Matrix(epobs)
      M3 = BMatrixEpoch12Epoch2(1950.0, epoch2)
      return M3*M2*M1
   elif (S1==icrs and S2==icrs):
      return I()
   elif (S1==icrs and S2==fk4):
      M1 = ICRS2FK5Matrix()
      M2 = FK52FK4Matrix(epobs)
      M3 = BMatrixEpoch12Epoch2(1950.0, epoch2)
      return M3*M2*M1
   elif (S1==icrs and S2==fk5):
      M1 = ICRS2FK5Matrix()
      M2 = JMatrixEpoch12Epoch2(2000.0, epoch2)
      return M2*M1
   elif (S1==fk5 and S2==icrs):
      M1 = JMatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = ICRS2FK5Matrix().T
      return M2*M1
   elif (S1==fk4 and S2==icrs):
      M1 = BMatrixEpoch12Epoch2(epoch1, 1950.0)
      M2 = FK42FK5Matrix(epobs)
      M3 = ICRS2FK5Matrix().T
      return M3*M2*M1
   elif (S1==j2000 and S2==j2000):
      M1 = IAU2006MatrixEpoch12Epoch2(epoch1, epoch2)
      return M1
   elif (S1==j2000 and S2==icrs):
      M1 = IAU2006MatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = ICRS2J2000Matrix().T
      return M2*M1
   elif (S1==j2000 and S2==fk5):
      M1 = IAU2006MatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = ICRS2J2000Matrix().T
      M3 = ICRS2FK5Matrix()
      M4 = JMatrixEpoch12Epoch2(2000.0, epoch2)
      return M4*M3*M2*M1
   elif (S1==j2000 and S2==fk4):
      M1 = IAU2006MatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = ICRS2J2000Matrix().T
      M3 = ICRS2FK5Matrix()
      M4 = FK52FK4Matrix(epobs)
      M5 = BMatrixEpoch12Epoch2(1950.0, epoch2)
      return M5*M4*M3*M2*M1
   elif (S1==icrs and S2==j2000):
      M1 = ICRS2J2000Matrix()
      M2 = IAU2006MatrixEpoch12Epoch2(2000.0, epoch2)
      return M2*M1
   elif (S1==fk5 and S2==j2000):
      M1 = JMatrixEpoch12Epoch2(epoch1, 2000.0)
      M2 = ICRS2FK5Matrix().T
      M3 = ICRS2J2000Matrix()
      M4 = IAU2006MatrixEpoch12Epoch2(2000.0, epoch2)
      return M4*M3*M2*M1
   elif (S1==fk4 and S2==j2000):
      M1 = BMatrixEpoch12Epoch2(epoch1, 1950.0)
      M2 = FK52FK4Matrix(epobs).T
      M3 = ICRS2FK5Matrix().T
      M4 = ICRS2J2000Matrix()
      M5 = IAU2006MatrixEpoch12Epoch2(2000.0, epoch2)
      return M5*M4*M3*M2*M1
   else:
      mes = "Unknown celestial reference system: %s or %s" % (S1, S2) 
      raise Exception(mes)



def rotmatrix(skyin, skyout, epoch1=2000.0, epoch2=2000.0, S1=fk5, S2=fk5, epobs=None):
   """
----------------------------------------------------------------------
Purpose:    Calculate and return the wanted rotation matrix.
Input:      A complete specification of input and output sky systems.
            The sky systems are equatorial, ecliptic, galactic, supergalactic
            which are represented by numbers 0,1,2 and 3
            The reference systems are fk4, fk4_no_e, fk5, icrs, j2000
            which are represented by the numbers.
Returns:    Transformation matrix as in XYZout = M * XYZin
Reference:  -
---------------------------------------------------------------------
   """
   if skyin == equatorial:
      if skyout == equatorial:
         M1 = MatrixEpoch12Epoch2(epoch1, epoch2, S1, S2, epobs)   # eq -> eq   epoch1 to epoch2
         return M1
      if skyout == ecliptic:
         M1 = MatrixEpoch12Epoch2(epoch1, epoch2, S1, S2)
         M2 = MatrixEq2Ecl(epoch2, S2)
         return M2*M1
      if skyout == galactic:                                # eq(epoch1) -> galactic
         M1 = MatrixEpoch12Epoch2(epoch1, 1950.0, S1, fk4)
         M2 = MatrixEqB19502Gal()
         return M2*M1
      if skyout == supergalactic:                           # eq(epoch1) -> super galactic
         M1 = MatrixEpoch12Epoch2(epoch1, 1950.0, S1, fk4)
         M2 = MatrixEqB19502Gal()
         M3 = MatrixGal2Sgal()
         return M3*M2*M1
      else:
         mes = "Unknown output sky system: %s" % (S2,)
         raise Exception(mes)

   elif skyin == ecliptic:
      if skyout == equatorial:
         M1 = MatrixEq2Ecl(epoch1, S1).T                    # S2 sets epoch to Besselian or Julian
         M2 = MatrixEpoch12Epoch2(epoch1, epoch2, S1, S2)
         return M2*M1
      if skyout == ecliptic:                                # ecl -> ecl   epoch1 to epoch2
         # This is an epoch transformation only
         M1 = MatrixEq2Ecl(epoch1, S1).T                    # to eq(epoch1)
         M2 = MatrixEpoch12Epoch2(epoch1, epoch2, S1, S2)   # epoch1 to epoch2
         M3 = MatrixEq2Ecl(epoch2, S2)                      # return to ecl(epoch2)
         return M3*M2*M1
      if skyout == galactic:                                # ecl(epoch1) -> galactic
         M1 = MatrixEq2Ecl(epoch1, S1).T                    # to eq(epoch1)
         M2 = MatrixEpoch12Epoch2(epoch1, 1950.0, S1, fk4)  # to eq(2000.0)
         M3 = MatrixEqB19502Gal()                           # eq(2000) to gal
         return M3*M2*M1
      if skyout == supergalactic:                           # ecl(epoch1) -> super galactic
         M1 = MatrixEq2Ecl(epoch1, S1).T                    # to eq(epoch1)
         M2 = MatrixEpoch12Epoch2(epoch1, 1950.0, S1, fk4)  # to eq(2000.0)
         M3 = MatrixEqB19502Gal()                           # eq(2000) to gal
         M4 = MatrixGal2Sgal()                              # gal to sgal
         return M4*M3*M2*M1
      else:
         mes = "Unknown output sky system: %s" % (S2,)
         raise Exception(mes)

   elif skyin == galactic:
      if skyout == equatorial:                              # gal -> eq, epoch2
         M1 = MatrixEqB19502Gal().T                         # gal to fk4 B1950
         M2 = MatrixEpoch12Epoch2(1950.0, epoch2, fk4, S2)  # fk4 B1950 to eq, epoch2
         return M2*M1
      if skyout == ecliptic:                                # gal -> ecl(epoch1)
         M1 = MatrixEqB19502Gal().T                         # gal to fk4 B1950
         M2 = MatrixEpoch12Epoch2(1950.0, epoch2, fk4, S2)  # fk4 B1950 to fk5 any epoch,equinox
         M3 = MatrixEq2Ecl(epoch2, S2)                      # eq(epoch2) to ecl(epoch2)
         return M3*M2*M1
      if skyout == galactic:                                # gal -> gal
         return I()
      if skyout == supergalactic:                           # gal -> sgal
         M1 = MatrixGal2Sgal()
         return M1
      else:
         mes = "Unknown output sky system: %s" % (S2,)
         raise Exception(mes)

   elif skyin == supergalactic:
      if skyout == equatorial:                              # sgal -> eq(epoch2)
         M1 = MatrixGal2Sgal().T                            # sgal to gal
         M2 = MatrixEqB19502Gal().T                         # gal to eq(2000)
         M3 = MatrixEpoch12Epoch2(1950.0, epoch2, fk4, S2)  # epoch 2000 to epoch2
         return M3*M2*M1
      if skyout == ecliptic:                                # sgal -> ecl(epoch2)
         M1 = MatrixGal2Sgal().T                            # sgal to gal
         M2 = MatrixEqB19502Gal().T                         # gal to eq(2000)
         M3 = MatrixEpoch12Epoch2(1950.0, epoch2, fk4, S2)  # 1950 to epoch2
         M4 = MatrixEq2Ecl(epoch2, S2)                      # eq(epoch2) to ecl(epoch2)
         return M4*M3*M2*M1
      if skyout == galactic:                                # sgal -> gal
         M1 = MatrixGal2Sgal().T
         return M1
      if skyout == supergalactic:                           # sgal -> sgal
         return I()
      else:
         mes = "Unknown output sky system: %s" % (S2,)
         raise Exception(mes)
   else:
      mes = "Unknown input sky system: %s" % (S1,)
      raise Exception(mes)



def skyparser(skyin):
#----------------------------------------------------------------------
   """
Parse a string, tuple or single integer that represents a sky definition.
A sky definition can consist of a *sky system*,
a *reference system*, an *equinox* and an *epoch of
observation*.
See also the description at :ref:`celestial-skydefinitions`.
The elements in the string are separated by
a comma or a space. The order of the elements is not important.
The string is converted to a tuple by :func:`celestial.parseskydefs`.

The parser is used in function :func:`celestial.skymatrix`
and :func:`celestial.sky2sky`. External applications can use this function
to check whether user input is valid.

Definitions in strings are usually used to define output sky definitions
in prompts or on command lines. Applications can use integer id's
for the sky- and reference systems. These integer id's are global constants
See also :ref:`celestial-skysystems` and :ref:`celestial-refsystems`.
          
The sky system and reference system strings are minimal matched
(case INsensitive) with the strings in the table
in the documentation at :ref:`celestial-skysystems` and :ref:`celestial-refsystems`.

For the epoch syntax read the documentation at :ref:`celestial-epochs`.
Note that an epoch of observation is either a second epoch in the string
(the first is always the equinox) or the epoch string has
a suffix '_' which may be follwed by arbitrary characters.

:param skyin:
   Represents a sky definition. See examples.
:type skyin:
   String, tuple or integer
      
:Returns:
   A tuple with the 'coded' system where strings for sky- and reference systems
   are replaced by integer id's. Missing values are filled in with defaults.

   If an error occurred then an exception will be raised. 

:raises:
   :exc:`ValueError`
      From :func:`celestial.parseskydefs`:
   
      *  *Empty string!*
      *  *Too many items for sky definition!*
      *  *... is ambiguous sky or reference system!*
      *  *... is not a valid epoch or sky/ref system!*

      From this function:

      * *Sky definition is not a string nor a tuple!*
      * *Too many elements in sky definition (max. 4)!*
      * *Two sky systems given!*
      * *Two reference systems given!*
      * *Invalid number for sky- or reference system!*
      * *Cannot determine the sky system!*
      * *Input contains an element that is not an integer or a string!*

:Examples: 
   
    >>> print celestial.skyparser("B1983.5_O fk4 B1960,eq")
    (0, 4, 1960.0, 1983.5)

    >>> print celestial.skyparser("su")
    (3, None, None, None)

    >>> print celestial.skyparser("supergal")
    (3, None, None, None)

 
      
:Notes:
   This is the parser for a sky definition.
   In this definition one can specify the sky system,
   the reference system, an equinox and an epoch of
   observation if the reference system is fk4.
   The order of these elements is not important.

   The rules for the defaults are:

   *   What if the sky system is not defined? If there is a reference
       system then we assume it is equatorial (could have been ecliptic).
   *   If there no sky system and no reference system but there is
       an equinox, assume sky system is equatorial (could have been ecliptic).
   *   If there no sky system and no reference system and no
       equinox but there is an epoch of observation,
       assume sky system is equatorial.
   *   Assume we have a sky system. What if there is no reference system?
       Standard in FITS: RADESYS (i.e our reference system) defaults to
       IRCS unless EQUINOX is given alone,
       in which case it defaults to FK4 prior to 1984 and FK5 after 1984.
   *   Assume we have a sky system and a reference system and the sky system was
       ecliptic or equatorial. What if we don't have an equinox?
       Standard in FITS: EQUINOX defaults to 2000 unless RADESYS is FK4,
       in which case it defaults to 1950.
   *   We have one item to address and that is the epoch of observation.
       This epoch of observation only applies to the reference systems FK4
       and FK4_NO_E.
       In 'Representations of celestial coordinates in FITS' (Calabretta & Greisen)
       we read that all reference systems are allowed for both equatorial- and
       ecliptic coordinates, except FK4-NO-E, which is only allowed for equatorial
       coordinates. If FK4-NO-E is given in combination with an ecliptic
       sky system then silently FK4 is assumed.
   """
#----------------------------------------------------------------------
   epochin    = None
   epochinset = None
   refin = None
   epobs = None
   sysin = None
   first = True
   if skyin == None:      # Nothing to parse
      return sysin, refin, epochin, epobs

   if type(skyin) not in [tuple, bytes]:
      try:
         skyin = tuple([skyin])
      except:
         raise ValueError("Sky definition is not a string nor a tuple or a scalar!")
   if type(skyin) == bytes:
      skyin = parseskydef(skyin)
      if skyin is None:   # e.g. input was '{}' then parseskydef returns None
         return None, None, None, None
   if len(skyin) > 4:
      raise ValueError("Too many elements in sky definition (max. 4)!")

   # Parse the tuple into a sky system, a reference system, equinox and obs epoch
   for element in skyin:
      if type(element) == int:
         s = skyrefsystems.id2skyref(element)
         if s != None:
            if s.refsystem:
               if refin == None:
                  refin = element
               else:
                  raise ValueError("Two sky systems given!")
            else:
               if sysin == None:
                  sysin = element
               else:
                  raise ValueError("Two reference systems given!")
         else:
            raise ValueError("Invalid number for sky- or reference system!")
      elif isinstance(element, six.string_types):
         if first and element.find('_') == -1:   # i.e. it is not an obs epoch
            epochinset = epochs(element)
            first = False
         else:
            # Could be obs. epoch if underscore in string or it is the second epoch
            epobs = epochs(element)[0]           # Always in Besselian data
      elif element != None:
         raise ValueError("Input contains an element that is not an integer or a string!")
   #------------------------------------------------------------
   # At this stage we have
   # sysin (sky system): integer or None
   # refin (ref. system): integer or None
   # epochinset (equinox): (B, J, JD) or None
   # epobs (epoch of observation): Besselian epoch or None
   #------------------------------------------------------------


   # Here we start to fill in the missing parts.
   # Most defaults are defined in the FITS standard. Others are the
   # most sensible. If essential parts are missing then an exception
   # will be raised.

   # What if the sky system is not defined? If there is a reference
   # system then we assume it is equatorial.
   if sysin == None:
      if refin != None:
         sysin = eq  # But this could also be ecliptic (except for fk4_no_e)
      else:
         if epochinset != None:  # No ref sys but an equinox: assume equatorial
            sysin = eq
         elif epobs != None:
            sysin = eq
         else:
            raise ValueError("Cannot determine the sky system!")

   # Now we have a sky system. What if there is no reference system?
   # Standard in FITS: RADESYS defaults to IRCS unless EQUINOX is given alone, 
   # in which case it defaults to FK4 prior to 1984 and FK5 after 1984.
   if sysin in [eq, ecl]:
      if refin == None:
         refin = icrs
         if epochinset != None:
            jd = epochinset[2]
            if jd < epochJulian2JD(1984.0):
               epochin = JD2epochBessel(jd)  # Always Besselian even if epoch was specified as Julian
            else:
               epochin = JD2epochJulian(jd)
            if epochin < 1984.0:
               refin = fk4  # Dangerous default. Could also be fk4_no_e for radio data
            else:
               refin = fk5
         elif sysin == eq and epobs != None:
            # If there is no reference system and there is no equinox
            # but there is an epoch of observation, then the reference is
            # fk4
            refin = fk4
   else:
      # Other sky systems do not have a reference system
      refin = None

   # We have a sky system and a reference system if the sky system was
   # ecliptic or equatorial. What if we don't have an equinox?
   # FITS: EQUINOX defaults to 2000 unless RADESYS is FK4, in which case
   # it defaults to 1950.
   if sysin in [eq, ecl]:
      if epochinset == None:
         if refin == fk4 or refin == fk4_no_e:  # The ref. system belongs to the fk4 family, 
            epochin = 1950.0
         else:
            epochin = 2000.0
      else:
         if refin == fk4 or refin == fk4_no_e:
            epochin = epochinset[0]          # Besselian epoch
         else:
            epochin = epochinset[1]          # Julian epoch

   # We have one item to address and that is the epoch of observation.
   # In 'Representations of celestial coordinates in FITS' (Calabretta & Greisen)
   # we read that all reference systems are allowed for both equatorial- and
   # ecliptic coordinates. Except FK4-NO-E which is only allowed for equatorial
   # coordinates!
   # This seems to contradict the fact that we must convert from fk4 to ecliptic
   # via fk4-no-e and therefore the actual reference system is fk4-no-e
   if not ((sysin == eq or sysin == ecl) and (refin == fk4 or refin == fk4_no_e)):
       epobs = None

   return sysin, refin, epochin, epobs



def parseskydef(skydef_in):
#----------------------------------------------------------------------
   """
Parse a string that represents a sky definition.
See documentation at function skyparser()
A tuple with values is returned. If the sky system was empty as
in {}, then return None
   """
#----------------------------------------------------------------------
   if skydef_in == '':
      raise Exception('Empty string!')

   bs = skydef_in.startswith('{')
   be = skydef_in.endswith('}')
   if bs and not be:
      raise ValueError("Definition starts with '{' but does not end with '}'")
   if be and not bs:
      raise ValueError("Definition ends with '}' but does not start with '{'")
   if bs and be:
      skydef = skydef_in[1:-1]     # Remove braces
      if len(skydef.strip()) == 0: # Empty sky def. {}
         return None 
   else:
      skydef = skydef_in

   tokens = re_split('[,\s]+', skydef.strip())           # Split on whitespace and comma
   if len(tokens) > 4:                                   # sky, ref, equinox, dateobs
      raise ValueError("Too many items for sky definition!")

   sky = []
   for t in tokens:
      t.strip()
      errmes = ''
      s, found = skyrefsystems.minmatch2skyref(t)        # 'skyrefs' is global list
      if s != None:
         if found > 1:
            errmes = "%s is ambiguous sky or reference system!" % t
            raise ValueError(errmes)
         else:
            sky.append(s.idnum)
      else:
         try:
            B, J, JD = epochs(t)
            sky.append(t)
         except:
            errmes = "%s is not a valid epoch or sky/ref system!" % t
            raise ValueError(errmes)
   return tuple(sky)


def isparsed(skytuple):
   #----------------------------------------------------------------------
   """
   A sky definition after parsing is a tuple with 4 elements.
   None of these is a string. The first one is a number and
   the others are either a number or are equal to None.
   """
   #----------------------------------------------------------------------
   if type(skytuple) == tuple and len(skytuple) == 4 and\
      type(skytuple[0]) == int and\
      type(skytuple[1]) != bytes and\
      type(skytuple[2]) != bytes and\
      type(skytuple[3]) != bytes:
      return True
   return False


def skymatrix(skyin, skyout):
#----------------------------------------------------------------------
   """
Create a transformation matrix to be used to transform a position from
one sky system to another (including epoch transformations).
For a description of the sky definitions see :ref:`celestial-skydefinitions`.

:param skyin:
   One of the supported sky systems or a tuple for equatorial systems
   which are identified with an equinox an reference system.
   This is the sky system from which you want to transform to
   another sky system (*skyout*).
:type skyin:
   Integer or tuple with one to four elements
:param skyout:
   The destination sky system
:type skyin:
   Integer or tuple with one to four elements
   

:Returns: Three elements:
   
   * The transformation matrix *M* for the transformation
     of positions in (x,y,z) as in *XYZskyout = M * XYZskyin*
   * followed by 'None' or a tuple with the e-term vector belonging
     input epoch.
   * followed by *None* or a tuple with the e-term vector belonging
     to the output epoch.
           
   See also notes below.
   
:Notes:
   The reference systems FK4 and FK4_NO_E are special. We
   consider FK4 as a catalog position where the **e-terms** are
   included. So besides a transformation matrix, this function
   should also return a flag for the addition or removal of
   e-terms. This flag is either *None* or the e-term vector
   which depends on the epoch.
   
   The structure of the output then is as follows:
   ``M, (A1,A2,A3), (A4,A5,A6)``
   where:
   
   * *M*: The 3x3 transformation matrix
   * *(A1,A2,A3)* or *None*: for adding or removing e-terms
     in the input sky system using this e-term vector *(A1,A2,A3)*.
   * *(A4,A5,A6)* or *None*: for adding or removing e-terms
     in the output sky system using this e-term vector *(A4,A5,A6)*.

   This function is the main function of this module.
   It calls function *skyparser()* for the parsing of the input and
   *rotmatrix()* to get the rotation matrix.
   There utility function *sky2sky()* transforms a sequence
   of longitudes and latitudes from one sky system to another.
   It is a valuable tool for experiments in an interactive Python
   session.
      
:Examples:
   Some examples of transformations between sky systems using either
   strings or tuples. We advise to use strings which is more safe
   then using variables from celestial (which can be accidentally
   replaced by other values). 
   Note that for transformations where FK4 is involved,
   the matrix is followed by a vector with e-terms.
   
      >>> from kapteyn import celestial
      >>> print skymatrix(celestial.gal,(celestial.eq,"j2000",celestial.fk5))
      (matrix([[-0.05487554,  0.49410945, -0.86766614],
               [-0.8734371 , -0.44482959, -0.19807639],
               [-0.48383499,  0.74698225,  0.45598379]]),
            None,
            None)
      
      >>> print skymatrix(celestial.fk4, celestial.fk5)
      (matrix([[  9.99925679e-01,  -1.11814832e-02,  -4.85900382e-03],
               [  1.11814832e-02,   9.99937485e-01,  -2.71625947e-05],
               [  4.85900377e-03,  -2.71702937e-05,   9.99988195e-01]]),
            (-1.6255503575995309e-06,
               -3.1918587795578522e-07,
               -1.3842701121066153e-07), None)
      
      >>> print skymatrix("eq,B1950.0,fk4_no_e","eq,B1950.0,fk4")
      (matrix([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]]),
            None,
            (-1.6255503575995309e-06,
               -3.1918587795578522e-07,
               -1.3842701121066153e-07))
      
      >>> print skymatrix("eq b1950 fk4 j1983.5", "eq J2000 fk5")
      (matrix([[  9.99925679e-01,  -1.11818698e-02,  -4.85829658e-03],
               [  1.11818699e-02,   9.99937481e-01,  -2.71546879e-05],
               [  4.85829648e-03,  -2.71721706e-05,   9.99988198e-01]]),
            (-1.6255503575995309e-06,
               -3.1918587795578522e-07,
               -1.3842701121066153e-07),
            None)
      
      >>> print skymatrix("eq J2000 fk4 F1984-1-1T0:30", "eq J2000 fk5")
      (matrix([[  1.00000000e+00,  -5.45185721e-06,  -3.39404820e-07],
               [  5.45185723e-06,   1.00000000e+00,   2.24950276e-08],
               [  3.39404701e-07,  -2.24971595e-08,   1.00000000e+00]]),
            (-1.6181121582090453e-06,
               -3.4112123324131958e-07,
               -1.4789407828956555e-07),
            None)


   See :ref:`celestial-epochs` for the possible epoch formats.
   """
#---------------------------------------------------------------------
   if isparsed(skyin):                           # Then no need to parse again
      sysin, refin, epochin, epobsin = skyin
   else:
      sysin, refin, epochin, epobsin = skyparser(skyin)
   if isparsed(skyout):
      sysout, refout, epochout, epobsout = skyout
   else:
      sysout, refout, epochout, epobsout = skyparser(skyout)

   # Take care of the e-terms
   Aep1 = None; Aep2 = None
   if sysin == eq:
      if refin == fk4:
         # This is a catalog value. We should remove e-terms before we transform anything
         Aep1 = getEterms(epochin)
      if refin == fk4_no_e:
         refin = fk4
   if sysout == eq:
      if refout == fk4:
         Aep2 = getEterms(epochout)
      if refout == fk4_no_e:
         refout = fk4
   # No e-terms for ecliptic coordinates in fk4
   # If fk4-no-e was selected then use fk4
   if sysin == ecl:
      if refin == fk4_no_e:
         refin = fk4
   if sysout == ecl:
      if refout == fk4_no_e:
         refout = fk4

   epobs = None
   if refin == fk4 and epobsin != None:
      epobs = epobsin
   if refout == fk4 and epobsout != None:
      epobs = epobsout

   return rotmatrix(sysin, sysout, epochin, epochout, refin, refout, epobs), Aep1, Aep2



def dotrans(skytuple, xyz):
   """
----------------------------------------------------------------------
Purpose:  Utility function that performs the rotation and adding or
          removing e-terms
Input:   -The tuple as produced by skymatrix
         -one or more positions in Cartesian coordinates (xyz)
Returns:  The transformed (Cartesian) coordinates
Notes:    Function skymatrix returns a tuple with the rotation matrix
          and e-terms if necessary. Tuple element 0 is the rotation
          matrix. Function dotrans() does the rotation for a vector 
          in Cartesian coordinates.
Examples: >>> lonlat = n.array( [(lon,lat)] )
          >>> xyz = longlat2xyz(lonlat)
          >>> M = skymatrix((eq,fk4,'j1950','b1995.0'), (eq,'J2000',fk5))
          >>> xyz2 = dotrans(M , xyz)
----------------------------------------------------------------------
   """
   M, A1, A2 = skytuple
   if A1:
      xyz2 = removeEterms(xyz, A1)
   else:
      xyz2 = xyz
   xyz3 = M*xyz2
   if A2:
      xyz3 = addEterms(xyz3, A2)
   return xyz3



def sky2sky(skyin, skyout, lons, lats):
#----------------------------------------------------------------------
   """
Utility function to facilitate command line use of skymatrix.

:param skyin:
   The input sky definition
:type skyin:
   See function :func:`skymatrix`
:param skyout:
   The output sky definition
:type skyout:
   See function :func:`skymatrix`
:param lons:
   Input longitude(s)
:type lons:
   Floating point number(s), scalar, list or tuple
:param lats:
   Input latitude(s)
:type lats:
   Floating point number(s), scalar, list or tuple

:Returns:
   Matrix. One position per row. See example below how to
   extract rows, columns and elements from this matrix.

:Example:
   Interactive Python session:

      >>> from kapteyn import celestial
      >>> M = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                                  (0,0,1.0), (10,20,20) )
      >>> M
      matrix([[ 102.6262244 ,  -50.83256452],
              [ 106.78021643,  -41.25289649],
              [ 107.9914125 ,  -41.49143448]])
      >>> M[2,0]
      107.99141249678289
      >>> M[0]         # Extract first transformed long, lat
      matrix([[ 102.6262244 ,  -50.83256452]])
      >>> M[:,1]       # Extract second column with latitudes
      matrix([[-50.83256452],
              [-41.25289649],
              [-41.49143448]])

:Notes:
   This function illustrates the core use of module *celestial*.
   First it converts the input of world coordinates into a matrix.
   This matrix is converted to spatial positions (X,Y,Z) with
   function *longlat2xyz()*. The function *dotrans()* transforms
   these positions (X,Y,Z) to positions (X2,Y2,Z2) in the output sky
   system. Then the function *xyz2longlat()* converts these positions
   into longitudes and latitudes and finally a matrix with these
   values is returned::
   
      lonlat = n.array( [(lons,lats)] )
      xyz = longlat2xyz(lonlat)
      xyz2 = dotrans(skymatrix(skyin, skyout), xyz)
      newlonlats = xyz2longlat(xyz2)
      return newlonlats

   """
#----------------------------------------------------------------------
   lonlat = n.array( [(lons,lats)] )
   xyz = longlat2xyz(lonlat)
   xyz2 = dotrans(skymatrix(skyin, skyout), xyz)
   newlonlats = xyz2longlat(xyz2)
   return newlonlats

