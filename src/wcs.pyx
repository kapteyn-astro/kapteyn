"""
==========
Module wcs
==========

.. author:: Hans Terlouw <gipsy@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5



Introduction
------------

.. index:: WCSLIB

This Python module interfaces to Mark Calabretta's
`WCSLIB <http://www.atnf.csiro.au/people/mcalabre/WCS/>`_ and also
provides a self-contained suite of celestial transformations.  The
WCSLIB routines "implement the FITS World Coordinate System (WCS)
standard which defines methods to be used for computing world
coordinates from image pixel coordinates, and vice versa." The celestial
transformations have been implemented in Python, using NumPy, and
support equatorial and ecliptic coordinates of any epoch and reference
systems FK4, FK4-NO-E, FK5, ICRS and dynamic J2000, and galactic and
supergalactic coordinates. 

.. index:: coordinate representation

.. _wcs-coordinates:

Coordinates
-----------
Coordinates can be represented in a number of different ways:

- as a tuple of scalars, e.g. (ra, dec).
- as a tuple of lists or NumPy arrays, e.g. ([ra_1, ra_2, ...], [dec_1, dec_2, ...], [vel_1, vel_2, ...]).
- as a NumPy matrix. The standard representation is a matrix with column vectors, but row vectors are also supported.
- as a NumPy array. This array can have any shape. The individual coordinate components are stored contiguously along the last axis.
- as a list of tuples. Every tuple represents one position, e.g. [(ra_1, dec_1), (ra_2, dec_2), ...].

Results delivered by the transformations done by the classes described
below will have the same representation as their inputs.
NumPy arrays and matrices will always be returned as type 'f8' (64 bit).

Class Projection
----------------
.. autoclass:: Projection(source[, rowvec=False, skyout=None, usedate=False, gridmode=False, alter='', minimal=False])

Class Transformation
--------------------
Celestial transformations are handled by objects of the class
Transformation.  These objects are callable.  Currently supported sky
systems are equatorial and ecliptic of any epoch and galactic and
supergalactic.

.. autoclass:: Transformation(sky_in, sky_out[, rowvec=False])

Functions
---------

Function coordmap
.................

.. autofunction:: coordmap(proj_src, proj_dst[, dst_shape=None, dst_offset=None, src_offset=None])

Utility functions
.................

The following are functions from the module :mod:`celestial` which have been
made available within the namespace of this :mod:`wcs` module:
For detailed information, refer to celestial's documentation.

.. function:: epochs(spec)

   Flexible epoch parser.

.. function:: lat2dms(a[, prec=1])

   Convert an angle in degrees into the degrees, minutes, seconds format
   assuming it was a latitude of which the value should be in the range
   -90 to 90 degrees. 

.. function:: lon2dms(a[, prec=1])

   Convert an angle in degrees to degrees, minutes, seconds.

.. function:: lon2hms(a[, prec=1])

   Convert an angle in degrees to hours, minutes, seconds format.

Constants
---------

**Sky systems** (imported from :mod:`celestial`)

.. data:: equatorial

.. data:: eq

.. data:: ecliptic

.. data:: ecl

.. data:: galactic

.. data:: gal

.. data:: supergalactic

.. data:: sgal

**Reference systems** (imported from :mod:`celestial`)

.. data:: fk4

.. data:: fk4_no_e

.. data:: fk5

.. data:: icrs

.. data:: dynj2000

.. data:: j2000

**Physical**

.. data:: c

   Velocity of light

Error handling
--------------

Errors are reported through the exception mechanism.  Two exception
classes have been defined: WCSerror for unrecoverable errors and
WCSinvalid for situations where a partial result may be available. 

.. rubric:: Footnotes

.. [#interpolation] For convenience, a slightly modified version of this
   module is also available in the Kapteyn Package as
   :mod:`kapteyn.interpolation`.  The modification replaces NaN values in
   the array to a finite value in case order>1, preventing the result
   becoming all NaN. 

"""

from c_wcs cimport wcsprm, wcsini, wcsset, wcsfree, wcsp2s, wcss2p, wcsmix, \
                   wcs_errmsg, wcssub, wcssptr, wcsprt, wcsutrn, celprm,\
                   unitfix, celfix, spcfix, wcsfix_errmsg, prj_categories,\
                   wcserr, wcserr_enable
from c_numpy cimport import_array, npy_intp, NPY_DOUBLE, PyArray_DATA, \
                     ndarray, PyArray_SimpleNewFromData, NPY_OWNDATA

import numpy, math, operator, types, os.path

from kapteyn.celestial import skymatrix, skyparser, \
                      eq, equatorial, ecl, ecliptic, gal, galactic, \
                      sgal, supergalactic, \
                      fk4, fk4_no_e, fk5, icrs, epochs, dynj2000, j2000, \
                      lon2hms, lon2dms, lat2dms
def issequence(obj):
   return isinstance(obj, (list, tuple, numpy.ndarray))

cdef extern from "math.h":
   cdef double floor(double x)

cdef extern from "xyz.h":
   cdef extern void to_xyz(double *world, double *xyz, int n, int ndims,
                           int lonindex, int latindex)
   cdef extern void from_xyz(double *world, double *xyz, int n, int ndims,
                           int lonindex, int latindex)
   cdef extern void flag_invalid(double *world, int n, int ndims,
                                 int *stat, float flag)

cdef extern from "eterms.h":
   cdef void eterms(double *xyz, int n, int direct,
                    double A0, double A1, double A2)

import_array()

if sizeof(long)<sizeof(long*):
   raise Exception, "cannot run on this architecture: pointer longer than long"

lontype  = 'longitude'
lattype  = 'latitude'
spectype = 'spectral'

skytab = { 'RA'   : equatorial,
           'ELON' : ecliptic,
           'GLON' : galactic,
           'SLON' : supergalactic
         }

reftab = { 'FK4'      : fk4,
           'FK4-NO-E' : fk4_no_e,
           'FK5'      : fk5,
           'ICRS'     : icrs
         }

c = 299792458.0                                          # velocity of light

TupleListFormat, ScalarTupleFormat, SequenceTupleFormat, ArrayFormat = range(4)
debug = False

# ==========================================================================
#                   Projection and Transformation classes
#                            (J.P. Terlouw)
# ==========================================================================

# ==========================================================================
#                             Declarations
# --------------------------------------------------------------------------
      
cdef extern from "stdlib.h":
   ctypedef int size_t
   void* malloc(size_t size)
   void* calloc(size_t nmemb, size_t size)
   void free(void* ptr)

cdef extern from "string.h":
   char *strncpy(char *dest, char *src, size_t n)

# ==========================================================================
#                             Functions
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
#                             void_ptr
# --------------------------------------------------------------------------
#   conversion from Python integer to pointer
#
cdef void *void_ptr(data):
   cdef long cdata
   cdata = data
   return <void*>cdata

# --------------------------------------------------------------------------
#                             world2world
# --------------------------------------------------------------------------
#   transform world coordinates
#
cdef world2world(skymat, double *world, n, ndims, lonindex, latindex):
   cdef double *c_xyz
   xyz = numpy.matrix(numpy.zeros(shape=(n,3), dtype='d')) # note: row vectors
   c_xyz = <double*>PyArray_DATA(xyz)
   to_xyz(world, c_xyz, n, ndims, lonindex, latindex)
   m_trans, e_prae, e_post = skymat
   if e_prae:
      eterms(c_xyz, n, -1, e_prae[0], e_prae[1], e_prae[2])
   xyz *= m_trans.T                                        # in place transform
   if e_post:
      eterms(c_xyz, n, +1, e_post[0], e_post[1], e_post[2])
   from_xyz(world, c_xyz, n, ndims, lonindex, latindex)

# --------------------------------------------------------------------------
#                             coordfix
# --------------------------------------------------------------------------
#   Frustrate WCSLIB's (version<4.4) test for all equal latitude or
#   longitude values.
#   This check can cause problems in case of |latitude| > 90 degrees.
#   Then the 180 degree adjustment for the longitude is incorrectly
#   applied to _all_ longitudes, also when |latitude| < 90 degrees.
#
cdef coordfix(double *world, n, ndims, lonindex, latindex):
   if n>1:
      if lonindex>=0 and lonindex<ndims:
         world[lonindex] += 1.0e-12
      if latindex>=0 and latindex<ndims:
         world[latindex] += 1.0e-12

# --------------------------------------------------------------------------
#                             pix2grd
# --------------------------------------------------------------------------
#   Convert pixel coordinates (dir=-1) to grid coordinates or grid coordinates
#   to pixel coordinates (dir=+1). In this context, grid coordinates are
#   simply CRPIX-relative pixel coordinates, where CRPIX is rounded to
#   the nearest integer.
#
cdef pix2grd(double *pixin, double *pixout, int n, wcsprm *param, int dir):
   cdef double *crpix = param.crpix
   cdef int naxis = param.naxis
   cdef int i
   
   for i in range(n*naxis):
      pixout[i] = pixin[i]+dir*floor(crpix[i%naxis]+0.5)

# --------------------------------------------------------------------------
#                             coordmap
# --------------------------------------------------------------------------
#   return a coordinate map to be used in calls to the function
#   scipy.ndimage.interpolation.map_coordinates()
#
def coordmap(proj_src, proj_dst, dst_shape=None, dst_offset=None,
                                                 src_offset=None):
   """
- *proj_src*, *proj_dst* -- the source- and destination projection objects.
- *dst_shape* -- the destination image's shape.  Must be compatible with the
  projections' dimensionality. The elements are in Python order, i.e.,
  the first element corresponds to the last FITS axis.
  If *dst_shape* is None (the default), the shape is derived from
  the *proj_dst.naxis* attribute.
- *dst_offset* -- the destination image's offset. If None, the offset for
  all axes will be zero. Otherwise it must be compatible with the
  projections' dimensionality. The elements are in Python order, i.e.,
  the first element corresponds to the last FITS axis.
- *src_offset* -- the source image's offset. If None, the offset for
  all axes will be zero. Otherwise it must be compatible with the
  projections' dimensionality. The elements are in Python order, i.e.,
  the first element corresponds to the last FITS axis.

This function returns a coordinate map which can be used as the
argument coordinates in calls to the function :func:`map_coordinates`
from the :mod:`scipy.ndimage.interpolation` module. [#interpolation]_
The resulting coordinate map can be
used for reprojecting an image into another image with a different
coordinate system.

Example::

   #!/bin/env python
   from kapteyn import wcs
   import numpy, pyfits
   from kapteyn.interpolation import map_coordinates
   
   hdulist = pyfits.open('ngc6946.fits')
   header = hdulist[0].header
   
   proj1 = wcs.Projection(header)                       # source projection
   trans = wcs.Transformation(proj1.skysys, skyout=wcs.galactic)
   
   header['CTYPE1'], header['CTYPE2'] = 'GLON-TAN', 'GLAT-TAN'
                                                        # new axis types
   header['CRVAL1'], header['CRVAL2'] = trans((header['CRVAL1'],header['CRVAL2']))
                                                        # new reference point
   
   proj2 = wcs.Projection(header)                       # destination projection
   
   coords = wcs.coordmap(proj1, proj2)
   
   image_in = hdulist[0].data
   image_out = map_coordinates(image_in, coords, order=1, cval=numpy.NaN)
   
   hdulist[0].data = image_out
   hdulist.writeto('ngc6946-gal.fits')

This example is a complete program and illustrates how a FITS file containing
an image with arbitrary coordinates can be reprojected into an
image with galactic coordinates.
The image can have two or more dimensions.
"""
   if dst_shape is None:
      dst_shape = proj_dst.naxis[-1::-1]

   naxis = len(dst_shape)

   if len(proj_src.types)!=naxis or len(proj_dst.types)!=naxis:
      raise TypeError, "incompatible projections and shape"

   if dst_offset is None:
      dst_offset = numpy.zeros((naxis,), dtype=numpy.int)
   else:
      dst_offset = numpy.flipud(dst_offset)

   if src_offset is None:
      src_offset = numpy.zeros((naxis,), dtype=numpy.int)
   else:
      src_offset = numpy.flipud(src_offset)

   gslices = []
   for ax in range(naxis):
      gslices.insert(0, slice(1,dst_shape[ax]+1))

   grids_dst = (numpy.mgrid[gslices]).T + dst_offset

   if proj_dst.specaxnum is not None and \
      proj_dst.ctype[proj_dst.specaxnum-1][:4] != \
      proj_src.ctype[proj_src.specaxnum-1][:4]:
      proj_dst = \
         proj_dst.spectra(proj_src.ctype[proj_src.specaxnum-1][:4]+'-???')
   else:
      proj_dst = proj_dst.copy()

   if proj_dst.skyout != proj_src.skyout:
      proj_dst.skyout = proj_src.skyout
   proj_dst.allow_invalid = True
   
   src_types = list(proj_src.types)
   src_perm = []                    # axis permutation relative to destination
   for axtype in proj_dst.types:
      iax = src_types.index(axtype)
      src_types[iax] = 'seen'       # do not visit a second time
      src_perm.append(iax+1)

   proj_src = proj_src.sub(src_perm)
   proj_src.allow_invalid = True

   grids_src = (proj_src.topixel(proj_dst.toworld(grids_dst))-1.0-src_offset).T

   result = numpy.zeros(grids_src.shape, dtype=numpy.float32)
   for iax in range(naxis):
      result[iax] = grids_src[src_perm[-iax-1]-1]

   return numpy.transpose(result, [0]+range(naxis, 0, -1))


# ==========================================================================
#                             WCS exceptions
# --------------------------------------------------------------------------
class WCSerror(Exception):
   pass

class WCSinvalid(WCSerror):
   pass

# --------------------------------------------------------------------------
#                             fmt_errmsg
# --------------------------------------------------------------------------
cdef fmt_errmsg(wcsprm *param):
   message = '%s. WCSLIB file %s, line %d. Function %s()' % \
   (param.err.msg, os.path.basename(param.err.file), param.err.line_no,
    param.err.function)
   return message

# ==========================================================================
#                             Class Coordinate
# --------------------------------------------------------------------------
#
#  Attributes:
#  -----------
#     dyn          - flag whether data array has been dynamically allocated
#                    and should be free'd afterwards. Applies to both source
#                    and result. Source array should be free'd by caller.
#     data         - pointer to double array, disguised as integer.
#     n            - number of coordinates.
#     ndims        - coordinates' dimensionality.
#     rowvec       - for matrix source and result: coordinates as row vectors.
#     __format     - TupleListFormat, ScalarTupleFormat, SequenceTupleFormat
#                    or ArrayFormat.
#     __matrix     - flag whether source is matrix.
#     __array      - for sequence tuples: elements are arrays
#     __shape      - for array source and result: its shape.
#
class Coordinate(object):

   def __init__(self, source, rowvec):
      cdef double *data
      if isinstance(source, list):
         if isinstance(source[0], tuple):
            self.__format = TupleListFormat
            self.n = len(source)
            self.ndims = len(source[0])
            data = <double*>malloc(self.n*self.ndims*sizeof(double))
            self.data = <long>data
            self.dyn = True
            i = 0
            for elem in source:
               for coord in elem:
                  data[i] = coord
                  i = i + 1
         else:
            self.dyn = False
            raise WCSerror, (-2, "unrecognized coordinate source")
      elif isinstance(source, tuple):
         if issequence(source[0]):
            self.__format = SequenceTupleFormat
            self.ndims = len(source)
            self.n     = len(source[0])
            self.__array = isinstance(source[0], numpy.ndarray)
            source = numpy.matrix(source, dtype='d').T.copy()
            data = <double*>PyArray_DATA(source)
            self.data = <long>data
            self.dyn = False
         else:
            self.__format = ScalarTupleFormat
            self.n = 1
            self.ndims = len(source)
            data = <double*>malloc(self.ndims*sizeof(double))
            self.data = <long>data
            self.dyn = True
            i = 0
            for coord in source:
               data[i] = coord
               i = i + 1
      elif isinstance(source, numpy.ndarray):
         self.__format = ArrayFormat
         self.__shape = source.shape
         if isinstance(source, numpy.matrix):
            self.rowvec = rowvec
            self.__matrix = True
            if not self.rowvec:
               source = source.T.copy()
         else:
            self.__matrix = False
         self.dyn = False
         self.ndims = source.shape[-1]
         n = 1
         for i in source.shape[:-1]:
            n = n*i
         self.n = n
         if not source.dtype=='d':
            source = source.astype('f8')
         if not source.flags.contiguous and source.flags.aligned:
            source = source.copy()
         data = <double*>PyArray_DATA(source)
         self.data = <long>data
      else:
         self.dyn = False
         raise WCSerror, (-2, "unrecognized coordinate source")
      self.source = source                  # prevent premature deallocation

   def __del__(self):
      cdef long data
      if self.dyn:
         data = self.data
         free(<void*>data)

   def result(self, pydata):
      cdef double *data
      cdef long   ldata
      cdef npy_intp c_nelem
      cdef ndarray result_c
      ldata = pydata
      data  = <double*>ldata
      if self.__format==TupleListFormat:
         i = 0
         result = []
         for elem in xrange(self.n):
            coord = []
            for j in range(self.ndims):
               coord.append(data[i])
               i = i + 1
            result.append(tuple(coord))
      elif self.__format==ScalarTupleFormat:
         i = 0
         coord = []
         for j in range(self.ndims):
            coord.append(data[i])
            i = i + 1
         result = tuple(coord)
      elif self.__format==ArrayFormat:
         # Note: *always* returns float64 array or matrix.
         c_nelem = self.n*self.ndims
         result = PyArray_SimpleNewFromData(1, &c_nelem, NPY_DOUBLE, data)
         result_c = <ndarray>result
         result_c.flags = result_c.flags | <int>NPY_OWNDATA
         result.shape = self.source.shape
         if self.__matrix:
            result = numpy.matrix(result, copy=False)
            if not self.rowvec:
               result = result.T.copy()
         result.shape = self.__shape
      elif self.__format==SequenceTupleFormat:
         c_nelem = self.n*self.ndims
         result = PyArray_SimpleNewFromData(1, &c_nelem, NPY_DOUBLE, data)
         result_c = <ndarray>result
         result_c.flags = result_c.flags | <int>NPY_OWNDATA
         result.shape = self.source.shape
         result = tuple(result.T)
         if not self.__array:
            lists = []
            for x in result:
               lists.append(list(x))
            result = tuple(lists)
      return result
            
# ==========================================================================
#                             Class WrappedHeader
# --------------------------------------------------------------------------
#
#  Simple dictionary subclass allowing to override selected items without
#  affecting the dictionary-like header object from which it is constructed.
#  Only the indexing operator is implemented, which will return any
#  redefined item if it exists, or otherwise the item from the original
#  header object.
#
class WrappedHeader(dict):

   optical =  ['FREQ-OHEL', 'FREQ-OLSR']
   radio   =  ['FREQ-RHEL', 'FREQ-RLSR']

   def __init__(self, header, alter):
      self.header = header
      if alter in [' ', None]:
         alter = ''
      self.alter = alter
      self.naxis = header['NAXIS']

   def freqtype(self):            # change special frequency types into 'FREQ'
      try:
         for i in range(1,self.naxis+1):
            key = 'CTYPE%d'%i + self.alter
            if self.header[key] in (self.optical+self.radio):
               self[key] = 'FREQ'
               break
      except KeyError:
         pass

   def wsrt_topo(self):           # WSRT file with topocentric frequencies?
      try:
         if self.header['INSTRUME'] != 'WSRT':
            return False
      except:
         return False
      try:
         refjd = epochs('F2006-07-03')[2]
         obsjd = epochs('F'+self.header['DATE-OBS'])[2]
         return obsjd<refjd
      except:
         return False

   def freqvalue(self):           # adjust special frequency values and types
      for i in range(1,self.naxis+1):
         suffix = '%d'%i + self.alter
         ctype = self.header['CTYPE' + suffix]
         if self.wsrt_topo() and ctype in (self.optical + self.radio):
            f  = self.header['CRVAL' + suffix]
            df = self.header['CDELT' + suffix]
            try:
               cunit = self.header['CUNIT' + suffix]
            except KeyError:
               cunit = 'Hz'
            for e, u in enumerate(['Hz', 'kHz', 'MHz', 'GHz', 'THz']):
               if u.lower()==cunit.lower():
                  fac = 10**(3*e)
                  f  *= fac
                  df *= fac
                  break
            else:
               raise WCSerror, (-13, 'invalid frequency unit: %s' % cunit)
            
            for key in ['RESTFRQ' + self.alter, 'RESTFREQ', 'FREQ0', 'FREQR']:
               try:
                  f0 = self.header[key]
                  break
               except KeyError:
                  pass
            else:
               raise WCSerror, (-11, "missing rest frequency")
            for key in ['DRVAL' + suffix, 'VELR']:
               try:
                  V = Z = self.header[key]
                  if key==('DRVAL%d'%i + self.alter):
                     try:
                        dtype = self.header['DTYPE' + suffix]
                     except:
                        dtype = 'VEL'
                     if dtype not in ['VELO', 'VEL']:
                        raise WCSerror, (-9, "invalid DTYPE: %s" % dtype)
                     try:
                        dunit = self.header['DUNIT' + suffix].lower()
                        if dunit=='km/s':
                           V = Z = 1000.0*Z
                        elif dunit!='m/s':
                           raise WCSerror, (-12, "invalid velocity unit: %s"
                                                 % dunit)
                     except KeyError:
                        raise WCSerror, (-14, "DRVALia missing")
                  elif key=='VELR':
                     pass
                  break
               except KeyError:
                  pass
            else:
               raise WCSerror, (-10, "missing reference velocity")
            if ctype in self.optical:
               fb = f0/(1.0+Z/c)               # optical velocity
            else:
               fb = f0*(1.0-V/c)               # radio velocity
            v = c * ((fb*fb-f*f)/(fb*fb+f*f))
            dfb = df*(c-v)/math.sqrt(c*c-v*v)
            self['CTYPE%d'%i + self.alter] = 'FREQ'
            self['CUNIT%d'%i + self.alter] = 'Hz'
            self['CRVAL%d'%i + self.alter] = fb
            self['CDELT%d'%i + self.alter] = dfb
            self['BARYFREQ']  = fb, dfb
            break

   def __getitem__(self, key):
      try:
         return super(WrappedHeader, self).__getitem__(key)
      except:
         return self.header[key]

# ========================================================================== 
#                             Function MinimalHeader
# --------------------------------------------------------------------------
#
#  Function returning a minimal representation of the header supplied
#  in the argument. Can be used to construct a minimal Projection object
#  when the current header contains errors which would prevent a
#  full Projection object to be made.
#
def MinimalHeader(header):
   result = {}
   naxis = header['NAXIS']
   result['NAXIS'] = naxis
   for i in range(1,naxis+1):
      keyword = 'NAXIS%d' % i
      result[keyword] = header[keyword]
   return result

# ==========================================================================
#                             Class Projection
# --------------------------------------------------------------------------
#
class Projection(object):
   """
:param source:
      a Python dictionary or dictionary-like object containing
      FITS-style keys and values, e.g. a header object from PyFITS.
:param rowvec:
      indicates whether input and output coordinates, when given as NumPy
      matrices, will be row vectors instead of the standard column vectors.
      True or False.
:param skyout:
      can be used to specify a system different from the sky
      system specified by the projection. This can be given as a string e.g.,
      ``"equatorial fk4_no_e B1950.0"`` or as a tuple:
      ``(equatorial fk4_no_e 'B1950.0')``.
      For a complete description see: :ref:`celestial-skydefinitions`.
:param usedate:
      indicates whether the date of observation is to be used for the
      appropriate celestial transformations. True or False.
:param gridmode:
      True or False. If True, the object will use grid coordinates instead
      of pixel coordinates. Grid coordinates are CRPIX-relative pixel
      coordinates, e.g. used in GIPSY. If CRPIX is not integer, the
      nearest integer is used as reference.
:param alter:
      an optional letter from 'A' through 'Z', indicating an alternative
      WCS axis description.
:param minimal:
      True or False. If True, the object will be constructed from only
      the NAXIS and NAXISi items in the source. All other items are ignored.
      In this way world- and pixel coordinates will have the same values.
      This can be useful when it is impossible to build an object from all
      items in the source, e.g., when there is an error in a FITS header.

**Methods:**

.. automethod:: toworld(pixel)
.. automethod:: topixel(world)
.. automethod:: toworld1d(pixel)
.. automethod:: topixel1d(world)
.. automethod:: mixed(world, pixel[, span=None, step=0.0, iter=7])
.. automethod:: sub([axes=None, nsub=None])
.. automethod:: spectra(ctype[, axindex=None])
.. automethod:: inside(coords, mode)
.. automethod:: pixel2grid(pixel)
.. automethod:: grid2pixel(grid)
.. automethod:: str2pos(postxt[, mixpix=None])

**WCSLIB-related attributes:**

The following attributes contain values which are parameters to
WCSLIB, *after* interpretation. So they can differ from the values
in the source object. These attributes should not be modified.

.. attribute:: category

   The projection category: one of the strings
   ``undefined``, ``zenithal``, ``cylindrical``, ``pseudocylindrical``,
   ``conventional``, ``conic``, ``polyconic``, ``quadcube``, ``HEALPix``.

.. attribute:: ctype

   A tuple with the axes' types in the axis order of the object.

.. attribute:: cunit

   A tuple with the axes' physical units in the axis order of the object.

.. attribute:: crval

   A tuple with the axes' reference values in the axis order of the object.

.. attribute:: cdelt

   A tuple with the axes' coordinate increments in the axis order of the
   object. 

.. attribute:: crpix

   A tuple with the axes' reference points in the axis order of the
   object. 

.. attribute:: crota

   A tuple with the axes' coordinate rotations, or None if no rotations
   have been specified.

.. attribute:: pc

   A NumPy matrix for the linear transformation between pixel axes
   and intermediate coordinate axes, or None if not specified.

.. attribute:: cd

   A NumPy matrix for the linear transformation (with scale) between pixel axes
   and intermediate coordinate axes, or None if not specified.
   
.. attribute:: pv

   A list with numeric coordinate parameters. Each list element is a tuple
   consisting of the world coordinate axis number
   `i`, the parameter number `m` and the parameter value.

.. attribute:: ps

   A list with character-valued coordinate parameters.
   Each list element is a tuple
   consisting of the world coordinate axis number
   `i`, the parameter number `m` and the parameter value.

.. attribute:: lonpole

   The native longitude of the celestial pole.

.. attribute:: latpole

   The native latitude of the celestial pole.

.. attribute:: euler

   A five-element list:
   Euler angles and associated intermediaries derived from the
   coordinate reference values.  The first three values are the Z-, X-,
   and Z'-Euler angles, and the remaining two are the cosine and sine
   of the X-Euler angle.

.. attribute:: equinox

   The equinox (formerly 'epoch') of the projection.

.. attribute:: restfrq

   Rest frequency in Hz.

.. attribute:: restwav

   Vacuum rest wavelength in m.

**Other Attributes:**

The attributes
*skyout*,
*allow_invalid*,
*rowvec*,
*epobs*,
*gridmode* and
*usedate*
can be modified at any time.
The others are read-only.



.. attribute:: skysys

   The projection's 'native' sky system.  E.g., ``(equatorial, fk5,
   'J2000.0')``. 

.. attribute:: skyout

   Alternative sky system.  Can be specified according to
   the rules of the module :mod:`celestial`.
   See: :ref:`celestial-skydefinitions`.
   For pixel-to-world
   transformations, the result in the projection's 'native' system is
   transformed to the specified one and for world-to-pixel transformations,
   the given coordinates are first transformed to the native system, then
   to pixels. 

.. attribute:: radesys

   Reference frame of equatorial or ecliptic coordinates: one of the
   (symbolic) values as defined in module :mod:`celestial`. E.g.
   ``icrs``, ``fk5`` or ``fk4``.

.. attribute:: epoch

   The projection's epoch string as derived from the attributes
   :attr:`equinox` and :attr:`radesys`. E.g., "B1950.0" or "J2000.0".

.. attribute:: dateobs

   The date of observation (string) as specified by the 'DATE-OBS' key
   in the source object or None if not present. 

.. attribute:: mjdobs

   The date of observation (floating point number) as specified by the
   'MJD-OBS' key in the source object or None if not present. 

.. attribute:: epobs

   The date of observation as specified by either the 'MJD-OBS' or the
   'DATE-OBS' key in the source object or None if both are absent.  This
   attribute is a string with the prefix 'MJD' or 'F' which can be parsed
   by the function epochs() in the module 'celestial' and consequently be
   part of the arguments *sky_in* and *sky_out* when creating a
   Transformation object. 

.. attribute:: gridmode

   True or False. If True, the object will use grid coordinates instead
   of pixel coordinates. Grid coordinates are CRPIX-relative pixel
   coordinates, e.g. used in GIPSY. If CRPIX is not integer, the
   nearest integer is used as reference.


.. attribute:: allow_invalid

   If set to True, no exception will be raised for invalid coordinates.
   Invalid coordinates will be indicated by ``numpy.NaN`` ('not a number')
   values.

.. attribute:: invalid

   True or False, indicating whether invalid coordinates were detected
   in the last transformation.  In the output, invalid coordinates are
   indicated by ``numpy.NaN`` ('not a number') values. 

.. attribute:: rowvec

   If set to True, input and output coordinates, when given as NumPy
   matrices, will be row vectors instead of the standard column vectors. 

.. attribute:: usedate

   Indicates whether the date of observation is to be used for the
   appropriate celestial transformations.  True or False. 

.. attribute:: types

   A tuple with the axes' coordinate types ('longitude', 'latitude',
   'spectral' or None) in the axis order of the object. 

.. attribute:: naxis

   A tuple with the axes' lengths in the axis order of the object. 
   (Convenience attribute not directly related to WCS.)

.. attribute:: lonaxnum

   Longitude axis number (1-relative). None if not defined.

.. attribute:: lataxnum

   Latitude axis number (1-relative). None if not defined.

.. attribute:: specaxnum

   Spectral axis number (1-relative). None if not defined.

.. attribute:: source

   Convenience attribute.  The object from which the Projection
   object was created.

.. attribute:: altspec

   A list of tuples with alternative spectral types and units.
   The first element of such a tuple is a string with an allowed alternative
   spectral type which can be used as the argument of method :meth:`spectra`
   and the second element is a string with the corresponding units.
   Example:
   ``[('FREQ', 'Hz'),  ('ENER', 'J'), ('VOPT-F2W', 'm/s'), ...,
   ('BETA-F2V', '')]``. If there is no spectral axis, the attribute will
   have the value None.

.. attribute:: altspecarg

   If the object was created with a call to :meth:`spectra`, the argument
   `ctype` as specified in that call. Otherwise None.

.. attribute:: minimal

   The object was created with the argument ``minimal=True``, using only
   the NAXIS and NAXISi items.

Example::

   #!/bin/env python
   from kapteyn import wcs
   import pyfits
   
   hdulist = pyfits.open('aurora.fits')      # open 3-dimensional FITS file
   
   proj3 = wcs.Projection(hdulist[0].header) # create Projection object
   
   pixel = ([51, 32], [17, 60], [11, 12])    # two 3-dimensional pixel coordinates
   world = proj3.toworld(pixel)              # transform pixel to world coordinates
   print world
   print proj3.topixel(world)                # back from world to pixel coordinates
   
   proj2 = proj3.sub([2,1])                  # subimage projection, axes 2 and 1
   
   pixel = ([1, 2, 4, 3], [7, 6, 8, 2])      # four 2-dimensional pixel coordinates
   world = proj2.toworld(pixel)              # transform pixel to world coordinates
   print world
   
   proj2.skyout = (wcs.equatorial, wcs.fk5,
                   'J2008')                  # specify alternative sky system
   
   world = proj2.toworld(pixel)              # transform to that sky system
   print world
   print proj2.topixel(world)                # back to pixel coordinates
"""



   def __init__(self, source=None, rowvec=False, skyout=None,
                      usedate=False, gridmode=False, alter='', minimal=False):

      wcserr_enable(1)
      dict_type, undef_type = range(2)
      source_type = undef_type
      self.debug = debug
      self.forward = None
      self.reverse = None
      self.world   = None
      self.pixel   = None
      self.dateobs = None
      self.mjdobs  = None
      self.altspecarg = None
      self.allow_invalid = False
      self.rowvec = rowvec
      self.gridmode = gridmode
      if alter in [' ', None]:
         alter = ''
      self.alter = alter
      self.minimal = minimal
      self.__dict__['usedate'] = False
      self.__dict__['epobs'] = None
      self.naxis = ()              # NAXISn - do not confuse with param.naxis!
      if source is None:
         return                    # empty object only
      try:
         source['NAXIS']           # is source a FITS dictionary?
         source_type = dict_type
      except:
         pass
      cdef wcsprm *param
      param = <wcsprm*>calloc(1, sizeof(wcsprm))
      param.flag = -1
      self.wcsprm = <long>param

      if source_type==dict_type:
         self.source = source
         if minimal:
            header = MinimalHeader(source)
         else:
            header = WrappedHeader(source, alter)
            header.freqtype()
         naxis = header['NAXIS']
         if wcsini(1, naxis, param):
            free(param)
            raise WCSerror, (-1, "Error allocating wcsprm struct")
         param.naxis = naxis
         param.altlin = 0
      
         # -------------------------
         #   RESTFRQ and/or RESTWAV
         # -------------------------
         param.restfrq = 0.0
         for key in ['RESTFRQ' + alter, 'RESTFREQ', 'FREQ0', 'FREQR']:
            try:
               param.restfrq = header[key]
               break
            except:
               pass
         try:
            param.restwav = header['RESTWAV' + alter]
         except:
            param.restwav = 0.0

         # ------------------------------------------
         #   CRVAL, CTYPE, CDELT, CRPIX, CROTA, NAXIS
         # ------------------------------------------
         encoding = "ascii"
         for i in range(naxis):
            iax = i+1
            try:
               param.crval[i] = header['CRVAL%d'%iax + alter]
            except:
               param.crval[i] = 0.0
            try:
               strncpy(param.ctype[i], header['CTYPE%d'%iax + alter].encode(encoding), 9)
            except:
               strncpy(param.ctype[i], ' ', 9)
            try:
               param.cdelt[i] = header['CDELT%d'%iax + alter]
            except:
               param.cdelt[i] = 1.0
            try:
               param.crpix[i] = header['CRPIX%d'%iax + alter]
            except:
               param.crpix[i] = 0.0
            try:
               param.crota[i] = header['CROTA%d'%iax + alter]
               param.altlin = param.altlin | 4
            except:
               param.crota[i] = 0.0
            try:
               strncpy(param.cunit[i], header['CUNIT%d'%iax + alter].encode(encoding), 9)
               wcsutrn(7, param.cunit[i])        # fix non-standard units
            except:
               strncpy(param.cunit[i], b'', 9)
            try:
               self.naxis += (header['NAXIS%d'%iax],)
            except:
               pass
      
         # --------------
         #    PC matrix
         # --------------
         for i in range(naxis):
            for j in range(naxis):
               k = i*naxis+j
               try:
                  param.pc[k] = header['PC%d_%d'%(i+1,j+1) + alter] # offical
                  param.altlin = param.altlin | 1
               except:
                  try:
                     param.pc[k] = header['PC%03d%03d'%(i+1,j+1) + alter] # obsolete
                     param.altlin = param.altlin | 1
                  except:
                     param.pc[k] = float(i==j)                     # absent

         # --------------
         #    CD matrix
         # --------------
         for i in range(naxis):
            for j in range(naxis):
               k = i*naxis+j
               try:
                  param.cd[k] = header['CD%d_%d'%(i+1,j+1) + alter] # offical
                  param.altlin = param.altlin | 2
               except:
                  try:
                     param.cd[k] = header['CD%03d%03d'%(i+1,j+1) + alter]  # obsolete
                     param.altlin = param.altlin | 2
                  except:
                     param.cd[k] = 0.0                             # absent
      
         #------------------
         #   EQUINOX, EPOCH
         #------------------
         # The float conversions here shouldn't be necessary, but there
         # seem to exist FITS files which represent EQUINOX as a string.
         # (From the Sloan Digital Sky Survey?)
         try:
            self.equinox = float(header['EQUINOX' + alter]) 
         except:
            try:
               self.equinox = float(header['EPOCH'])
            except:
               try:
                  radesys = reftab[header['RADESYS' + alter].upper()]
                  if radesys==fk4 or radesys==fk4_no_e:
                     self.equinox = 1950.0
                  else:
                     self.equinox = 2000.0
               except:
                  self.equinox = 2000.0
      
         #------------------
         #   RADESYS
         #------------------
         try:
            self.radesys = reftab[header['RADESYS' + alter].upper()] 
         except:
            if self.equinox<1984.0:
               self.radesys = fk4
            elif self.equinox==2000.0:
               self.radesys = icrs
            else:
               self.radesys = fk5

         #--------------
         #    "EPOCH"
         #--------------

         self.epoch = '%s%.1f' % (['J','B'][self.radesys==fk4 or
                                            self.radesys==fk4_no_e],
                                  self.equinox)
            
         #--------------
         #    PV array
         #--------------
         pvlist = []
         for i in range(naxis):
            for m in range(100):                        # 0 <= m <= 99
               try:
                  pvcard = (i+1, m, header['PV%d_%d' % (i+1, m) + alter])
                  pvlist.append(pvcard)
               except:
                  pass
         npv = len(pvlist)
         if npv>param.npvmax:
            raise WCSerror, (-6, "too many PV cards (%d) - increase npvmax"%npv)
         for ipv in range(npv):
            param.pv[ipv].i     = pvlist[ipv][0] 
            param.pv[ipv].m     = pvlist[ipv][1]
            param.pv[ipv].value = pvlist[ipv][2]
         param.npv = npv

         #--------------
         #    PS array
         #--------------
         pslist = []
         for i in range(naxis):
            for m in range(100):                        # 0 <= m <= 99
               try:
                  pscard = (i+1, m, header['PS%d_%d' % (i+1, m) + alter])
                  pslist.append(pscard)
               except:
                  pass
         nps = len(pslist)
         if nps>param.npsmax:
            raise WCSerror, (-6, "too many PS cards (%d) - increase npsmax"%nps)
         for ips in range(nps):
            param.ps[ips].i = pslist[ips][0]
            param.ps[ips].m = pslist[ips][1]
            strncpy(param.ps[ips].value, pslist[ips][2], 72)
         param.nps = nps

         #----------------------
         #    LONPOLE, LATPOLE
         #----------------------
         try:
            param.lonpole = header['LONPOLE' + alter]
         except:
            pass
         try:
            param.latpole = header['LATPOLE' + alter]
         except:
            pass
            
         #----------------------
         #    DATE-OBS
         #----------------------
         try:
            self.dateobs = header['DATE-OBS']
            self.__dict__['epobs'] = 'F'+self.dateobs
         except:
            pass

         #----------------------
         #    MJD-OBS
         #----------------------
         try:
            self.mjdobs = header['MJD-OBS']
            self.__dict__['epobs']  = 'MJD'+str(self.mjdobs)
         except:
            pass

         #----------------------
         #    VELREF
         #----------------------
         try:
            param.velref = header['VELREF']
         except:
            pass

         #---------------------------------
         #    Initialise parameter struct
         #---------------------------------
         status = celfix(param)
         if (status>0):
            raise WCSerror, (status, fmt_errmsg(param))
         status = spcfix(param)
         if (status>0):
            raise WCSerror, (status, fmt_errmsg(param))
         status = wcsset(param)
         if (status):
            raise WCSerror, (status, fmt_errmsg(param))
         if self.debug:
            wcsprt(param)
            

         #---------------------------------
         #    Euler angles
         #---------------------------------
         self.euler = []
         for i in range(5):
            self.euler.append(param.cel.euler[i])

         #---------------------------------
         #    Projection category
         #---------------------------------
         self.category = prj_categories[param.cel.prj.category]

         #---------------------------------
         #    Sky system
         #---------------------------------
         lngtyp = param.lngtyp.decode("ascii")
         if lngtyp not in skytab and param.lng>=0:
            lngtyp = param.ctype[param.lng] # shouldn't wcslib have done this?
         try:
            systype = skytab[lngtyp]
         except:
            systype = None
         if systype in (equatorial, ecliptic):
            if self.radesys==icrs:
               self.skysys = (skytab[lngtyp], icrs)
            else:
               self.skysys = (skytab[lngtyp], self.radesys, self.epoch)
         else:
            self.skysys = systype

         if skyout is not None:
            self.skyout = skyout
         else:
            self.skyout = self.skysys

         self.usedate = usedate

         self.__setaxtypes()

         return

      else:
         raise Exception, "invalid source argument"

   def __del__(self):
      cdef wcsprm *param
      param = <wcsprm*>void_ptr(self.wcsprm)
      wcsfree(param)
      free(param)

   def __setattr__(self, name, value):
      self.__dict__[name] = value
      if name=='skyout' and value is None:
         self.__dict__[name] = self.skysys
      if name=='usedate' or name=='skyout' or name=='epobs':
         if self.skysys is not self.skyout:
            if self.usedate:
               epobs = self.epobs
            else:
               epobs = None
            skysys = self.skysys
            if epobs is not None and \
               len(skysys)==3 and \
               skyparser(self.skyout)[3] is None:
               skysys += (epobs,)
            self.forward = skymatrix(skysys, self.skyout)
            self.reverse = skymatrix(self.skyout, skysys)
         else:
            self.forward = None
            self.reverse = None

   def __getattr__(self, name):
      cdef wcsprm *param
      if name=='lonpole':
         param = <wcsprm*>void_ptr(self.wcsprm)
         return param.lonpole
      if name=='latpole':
         param = <wcsprm*>void_ptr(self.wcsprm)
         return param.latpole
      if name=='altspec':
         if self.specaxnum is None:
            self.altspec = None
            return self.altspec
         else:
            specindex = self.specaxnum-1
         stypes = ['FREQ', 'ENER', 'WAVN', 'VOPT', 'VRAD', 'VELO', 'WAVE',
                   'ZOPT', 'AWAV', 'BETA']
         altspec = []
         for s in stypes:
            ctype = s+'-???'
            try:
               spec = self.spectra(ctype)
               altspec.append((spec.ctype[specindex], spec.units[specindex]))
            except:
               pass
         self.altspec = altspec
         return self.altspec
            
      raise AttributeError, "'Projection' object has no attribute %s" % name

   def sub(self, axes=None, nsub=None):

      """
      Extract a new Projection object for a subimage from an existing one.

      - *axes* is a sequence of image axis numbers to extract.  Order is
        significant; *axes[0]* is the axis number of the input image that
        corresponds to the first axis in the subimage, etc.  If not specified,
        the first *nsub* axes are extracted. 
      - *nsub* is the number of axes to extract when *axes* is not specified.

      :returns: a new Projection object."""

      cdef wcsprm *param, *newpar
      cdef int c_nsub[1], *c_axes
      if nsub is None:
         nsub = 0
      if axes is None:
         axes = range(1,nsub+1)
      param = <wcsprm*>void_ptr(self.wcsprm)
      newpar = <wcsprm*>calloc(1, sizeof(wcsprm))
      c_axes = <int*>malloc(param.naxis*sizeof(int))
      if axes:
         if not issequence(axes):
            axes = [axes]
         i = 0
         for axis in axes:
            if i>=param.naxis:
               free(newpar)
               free(c_axes)
               raise WCSerror, (-1, 'Too many sub-projection axes')
            c_axes[i] = axis
            if axis>param.naxis or axis<1:
               free(newpar)
               free(c_axes)
               raise WCSerror, (-1, 'Invalid sub-projection axis number')
            i = i + 1
      if i:
         c_nsub[0] = i
      else:
         c_nsub[0] = nsub
      newpar.flag = -1
      status = wcssub(1, param, c_nsub, c_axes, newpar)
      if status:
         message = fmt_errmsg(newpar)
         free(newpar)
         free(c_axes)
         raise WCSerror, (status, message)
      status = wcsset(newpar)
      if status:
         message = fmt_errmsg(newpar)
         free(newpar)
         free(c_axes)
         raise WCSerror, (status, message)
      projection = Projection()
      for key in self.__dict__.keys():
         projection.__dict__[key] = self.__dict__[key]
      projection.wcsprm = <long>newpar
      if self.debug:
         wcsprt(newpar)

      free(c_axes)
      newax = ()
      try:
         for axis in axes:
            newax += (self.naxis[axis-1],)
      except:
         pass
      projection.naxis = newax
      projection.__setaxtypes()
      return projection

   def copy(self):
      cdef wcsprm *param
      param = <wcsprm*>void_ptr(self.wcsprm)
      return self.sub(range(1,param.naxis+1))

   def spectra(self, ctype, axindex=None):
      """
      Create a new Projection object in which the spectral axis is
      translated.  For example, a 'FREQ' axis may be translated into
      'ZOPT-F2W' and vice versa.  For non-standard frequency types, e.g. 
      FREQ-OHEL as used by GIPSY, corrections are applied first to obtain
      barycentric frequencies.  For more information, see chapter
      :doc:`spectralbackground`.

      - *ctype* -- Required spectral CTYPEi.  Wildcarding may be used,
        i.e.  if the final three characters are specified as '???', or if just
        the eighth character is specified as '?', the correct algorithm code
        will be substituted and returned. The attribute :attr:`altspec`
        provides a list of acceptable spectral types.
        For later reference, the value of *ctype* is stored in the attribute
        :attr:`altspecarg` of the new Projection object.
      - *axindex* -- Index of the spectral axis (0-relative).  If not
        specified, the first spectral axis identified by the CTYPE values of the
        object is assumed. 

      :returns: a new Projection object."""

      cdef wcsprm *param, *newpar
      cdef int c_axindex[1]
      if axindex is None:
         axindex = -1
      param = <wcsprm*>void_ptr(self.wcsprm)
      newpar = <wcsprm*>malloc(sizeof(wcsprm))
      newpar.flag = -1
      status = wcssub(1, param, NULL, NULL, newpar)
      if status:
         message = fmt_errmsg(newpar)
         free(newpar)
         raise WCSerror, (status, message)
      status = wcsset(newpar)
      if status:
         message = fmt_errmsg(newpar)
         free(newpar)
         raise WCSerror, (status, message)
      naxis = newpar.naxis
      header = self.source             # save attribute
      for i in range(naxis):
         ctypei = newpar.ctype[i]
         if ctypei==b'FREQ':
            try:
               header = WrappedHeader(self.source, self.alter)
               header.freqvalue()
               fb, dfb = header['BARYFREQ']
               newpar.crval[i] = fb
               newpar.cdelt[i] = dfb
            except KeyError:
               pass
      status = wcsset(newpar)
      if status:
         message = fmt_errmsg(newpar)
         free(newpar)
         raise WCSerror, (status, message)
      c_axindex[0] = axindex
      ctype_tmp = (ctype+' ')[:len(ctype)]  #  wcssptr modifies this argument!
      status = wcssptr(newpar, c_axindex, ctype_tmp)
      if status:
         message = fmt_errmsg(newpar)
         wcsfree(newpar)
         free(newpar)
         raise WCSerror, (status, message)
      newpar.cunit[newpar.spec][0] = '\0' # work-around for strange WCSLIB bug
      status = wcsset(newpar)
      if status:
         message = fmt_errmsg(newpar)
         wcsfree(newpar)
         free(newpar)
         raise WCSerror, (status, message)
      projection = Projection()
      for key in self.__dict__.keys():
         projection.__dict__[key] = self.__dict__[key]
      projection.source = header       # restore possibly changed attribute
      projection.wcsprm = <long>newpar
      projection.altspecarg = ctype
      projection.__setaxtypes()
      if self.debug:
         wcsprt(newpar)
      return projection

   def toworld(self, source=None):
      """
      Pixel-to-world transformation.
      *pixel* is an object containing one or more pixel coordinates and
      a similar object with the corresponding world coordinates will be
      returned.
      Note that FITS images are indexed from (1,1), not from (0,0) like Python
      arrays.
      Coordinates can be specified in a number of different ways.
      See section :ref:`wcs-coordinates`.
      When an exception due to invalid coordinates has occurred, this method
      can be called again without arguments to retrieve the result in which the
      invalid positions will have the value ``numpy.NaN`` ("not a number")."""

      cdef wcsprm *param
      cdef double *imgcrd, *phi, *theta, *world, *pixel, *c_xyz
      cdef int *stat
      if source is None:
         if self.world is not None:
            result = self.world
            self.world = None
            return result
         else:
            raise WCSerror, (-4, "no world coordinates available")
      self.world = None
      self.invalid = False
      param = <wcsprm*>void_ptr(self.wcsprm)
      coord = Coordinate(source, self.rowvec)
      if param.naxis != coord.ndims:
         raise WCSerror, (-3, "wrong pixel dimensionality: %d instead of %d" % (coord.ndims, param.naxis))
      imgcrd = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      phi    = <double*>malloc(coord.n*sizeof(double))
      theta  = <double*>malloc(coord.n*sizeof(double))
      world  = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      stat   = <int*>malloc(coord.n*sizeof(int))
      if self.gridmode:
         pixel = <double*>malloc(coord.n*coord.ndims*sizeof(double))
         pix2grd(<double*>void_ptr(coord.data), pixel, coord.n, param, +1)
      else:
         pixel = <double*>void_ptr(coord.data)
      status = wcsp2s(param, coord.n, coord.ndims, pixel,
                      imgcrd, phi, theta, world, stat)
      if self.gridmode:
         free(pixel)
      if self.forward is not None:
         world2world(self.forward, world, coord.n, coord.ndims,
                     param.lng, param.lat)
      if status==8:
         flag_invalid(world, coord.n, coord.ndims, stat, numpy.NaN)
         self.invalid = True
      result = coord.result(<long>world)
      if coord.dyn:
         free(world)
      free(imgcrd)
      free(phi)
      free(theta)
      free(stat)
      if status:
         if status==8:
            if not self.allow_invalid:
               self.world = result
               raise WCSinvalid, (status, fmt_errmsg(param))
         else:
            raise WCSerror, (status, fmt_errmsg(param))
      return result

   def topixel(self, source=None):
      """
      World-to-pixel transformation.  Similar to :meth:`toworld`, this method
      can also be called without arguments. 
"""
      cdef wcsprm *param
      cdef double *imgcrd, *phi, *theta, *pixel
      cdef int *stat
      if source is None:
         if self.pixel is not None:
            result = self.pixel
            self.pixel = None            
            return result
         else:
            raise WCSerror, (-4, "no pixel coordinates available")
      self.pixel = None
      self.invalid = False
      param = <wcsprm*>void_ptr(self.wcsprm)
      if isinstance(source, numpy.ndarray) and self.reverse is not None:
         source = source.copy()   # prevent overwriting source by world2world()
      coord = Coordinate(source, self.rowvec)
      if param.naxis != coord.ndims:
         raise WCSerror, (-3, "wrong pixel dimensionality: %d instead of %d" % (coord.ndims, param.naxis))
      imgcrd = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      phi    = <double*>malloc(coord.n*sizeof(double))
      theta  = <double*>malloc(coord.n*sizeof(double))
      pixel  = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      stat   = <int*>malloc(coord.n*sizeof(int))
      if self.reverse is not None:
         world2world(self.reverse, <double*>void_ptr(coord.data),
                     coord.n, coord.ndims, param.lng, param.lat)

      coordfix(<double*>void_ptr(coord.data),
                     coord.n, coord.ndims, param.lng, param.lat)
      status = wcss2p(param, coord.n, coord.ndims,
                      <double*>void_ptr(coord.data),
                      phi, theta, imgcrd, pixel, stat)
      if status==9:
         flag_invalid(pixel, coord.n, coord.ndims, stat, numpy.NaN)
         self.invalid = True
      if self.gridmode:
         pix2grd(pixel, pixel, coord.n, param, -1)
      result = coord.result(<long>pixel)
      if coord.dyn:
         free(pixel)
      free(imgcrd)   
      free(phi)
      free(theta)
      free(stat)
      if status:
         if status==9:
            if not self.allow_invalid:
               self.world = result
               raise WCSinvalid, (status, fmt_errmsg(param))
         else:
            raise WCSerror, (status, fmt_errmsg(param))
      return result

   def topixel1d(self, source):
      """
      Simplified method for one-dimensional projection objects.  Its
      argument can be a list, a tuple, an array or a scalar.  An object of the
      same class will be returned."""

      if isinstance(source, tuple):
         return tuple(self.topixel((list(source),))[0])
      else:
         return self.topixel((source,))[0]
         
   def toworld1d(self, source):
      """
      Simplified method for one-dimensional projection objects.  Its
      argument can be a list, a tuple, an array or a scalar.  An object of the
      same class will be returned. """

      if isinstance(source, tuple):
         return tuple(self.toworld((list(source),))[0])
      else:
         return self.toworld((source,))[0]

   def mixed(self, src_world, src_pixel, span=None, step=0.0, iter=7):
      u"""
      Hybrid transformation.

      When either the celestial longitude or latitude plus an element of
      the pixel coordinate is given, the remaining elements are solved by
      iteration on the unknown celestial coordinate.  Which elements are to be
      solved, is indicated by assigning NaN to those elements.  In case of
      multiple coordinates, the same elements must be indicated for every
      coordinate.  This operation is only possible for the projection's
      "native" sky system.  When a different sky system has been specified, an
      exception will be raised.  When either both celestial coordinates or
      both pixel coordinates are given, an operation equivalent to
      :meth:`topixel` or :meth:`toworld` is performed. 
      For non-celestial coordinate elements any
      NaN value will be replaced by a value derived from the corresponding
      element in the other coordinate. 

      - *span* -- a sequence containing the solution interval for the
        celestial coordinate, in degrees.  The ordering of the two limits is
        irrelevant.  Longitude ranges may be specified with any convenient
        normalization, for example [-120,+120] is the same as [240,480], except
        that the solution will be returned with the same normalization, i.e. 
        lie within the interval specified.  The default is the appropriate CRVAL
        value \u00B115\u00B0. 
      - *step* -- step size for solution search, in degrees.
        If zero, a sensible, although perhaps non-optimal default will be used.
      - *iter* -- if a solution is not found then the step size will be
        halved and the search recommenced. iter controls how many times
        the step size is halved. The allowed range is 5 - 10. 

      :returns: a tuple (*world*, *pixel*) containing the resulting
                coordinates."""

      cdef wcsprm *param
      cdef double *imgcrd, *phi, *theta, *pixout, *wldout, *pixin, *wldin
      cdef int *stat
      cdef double vspan[2], pixnan[99], wldnan[99]
      cdef int i_c, n_c
      first_err = 0
      if src_world is None:
         if self.result is not None:
            result = self.result
            self.result = None
            return result
         else:
            raise WCSerror, (-4, "no mixed coordinates available")
      self.pixel = None
      self.invalid = False
      param = <wcsprm*>void_ptr(self.wcsprm)
      world = Coordinate(src_world, self.rowvec)
      if param.naxis != world.ndims:
         raise WCSerror, (-3, "wrong pixel dimensionality: %d instead of %d" % (world.ndims, param.naxis))
      pixel = Coordinate(src_pixel, self.rowvec)
      if param.naxis != pixel.ndims:
         raise WCSerror, (-3, "wrong pixel dimensionality: %d instead of %d" % (pixel.ndims, param.naxis))
      if world.n!=pixel.n:
         raise WCSerror, (-7, "world and pixel lengths are different")
      imgcrd = <double*>malloc(pixel.n*param.naxis*sizeof(double))
      phi    = <double*>malloc(pixel.n*sizeof(double))
      theta  = <double*>malloc(pixel.n*sizeof(double))
      pixout = <double*>malloc(pixel.n*param.naxis*sizeof(double))
      wldout = <double*>malloc(pixel.n*param.naxis*sizeof(double))
      wldin  = <double*>void_ptr(world.data)
      if self.gridmode:
         pixin = <double*>malloc(pixel.n*param.naxis*sizeof(double))
         pix2grd(<double*>void_ptr(pixel.data), pixin, pixel.n, param, +1)
      else:
         pixin  = <double*>void_ptr(pixel.data)
      stat   = <int*>malloc(pixel.n*sizeof(int))

      nwnan = int(numpy.isnan(wldin[param.lng])) +\
              int(numpy.isnan(wldin[param.lat]))
      npnan = int(numpy.isnan(pixin[param.lng])) +\
              int(numpy.isnan(pixin[param.lat]))
      if (nwnan+npnan)!=2 and (nwnan+npnan)!=0:
         raise WCSerror, (-15, "incompatible mixed input coordinates")
      if nwnan==1:
         if self.forward or self.reverse:
            raise WCSerror(-8, "hybrid transformation disallows skyout")
         mixcel = numpy.isnan(wldin[param.lng])+1
         if span is None:
            if mixcel==1:
               sp_start = param.crval[param.lat]-15.0  # rather arbitrary range
               sp_end   = sp_start+30.0
               sp_start = (sp_start, -90.0)[sp_start<-90.0]
               sp_end = (sp_end, 90.0)[sp_end>90.0]
               span = (sp_start, sp_end)
            else:
               sp_start = param.crval[param.lng]-15.0  # rather arbitrary range
               sp_end   = sp_start+30.0
               span = (sp_start, sp_end)
         vspan[0] = span[0]
         vspan[1] = span[1]
         for i in range(pixel.ndims):
            if numpy.isnan(pixin[i]):
               if i==param.lng:
                  mixpix = param.lat
                  break
               if i==param.lat:
                  mixpix = param.lng
                  break

      for i in range(pixel.ndims):
         if i!=param.lat and i!=param.lng:
            if numpy.isnan(pixin[i])==numpy.isnan(wldin[i]):
               raise WCSerror, (-15, "incompatible mixed input coordinates")
         pixnan[i] = numpy.isnan(pixin[i])
         wldnan[i] = numpy.isnan(wldin[i])

      n_c = world.n*param.naxis
      for i_c from 0 <= i_c < n_c:               # copy from input to output
         wldout[i_c] = wldin[i_c]
         if pixnan[i_c%param.naxis]:
            pixout[i_c] = param.crpix[i_c%param.naxis] # temporary dummy value
         else:
            pixout[i_c] = pixin[i_c]
            
      # replace NaN in independent world coordinates by correct value
      status = wcsp2s(param, world.n, world.ndims,
                      pixout, imgcrd, phi, theta, wldout, stat)
 
      for i_c from 0 <= i_c < n_c:               # restore overwritten values
         axno = i_c%param.naxis
         if pixnan[axno] or (nwnan==1 and (axno==param.lat or axno==param.lng)):
            wldout[i_c] = wldin[i_c]

      if npnan==2 and self.reverse:              # no pixels, only world
         world2world(self.reverse, wldout,
                     world.n, world.ndims, param.lng, param.lat)
      if status and npnan!=2:
         # npnan!=2: suppress WCSLIB error when spatial pixels are both NaN
         if status==8:
            flag_invalid(wldout, world.n, world.ndims, stat, numpy.NaN)
            self.invalid = True
         if not first_err:
            first_err = status

      if nwnan!=1:
         status = wcss2p(param, world.n, world.ndims,
                         wldout, phi, theta, imgcrd, pixout, stat)
         if status:
            if status==9:
               flag_invalid(pixout, pixel.n, pixel.ndims, stat, numpy.NaN)
               flag_invalid(wldout, world.n, world.ndims, stat, numpy.NaN)
               self.invalid = True
            if not first_err:
               first_err = status
      else:
         for i in xrange(world.n):               # element-by-element transform
            offset = i*param.naxis
            status = wcsmix(param, mixpix, mixcel, vspan, step, iter,
                         &wldout[offset], phi, theta, imgcrd, &pixout[offset])
            if status:
               for ax in range(world.ndims):
                  wldout[offset+ax] = numpy.NaN
                  pixout[offset+ax] = numpy.NaN
               self.invalid = True
               if not first_err:
                  first_err = status
                  first_msg = fmt_errmsg(param)

      if self.forward:                           # skyout
         world2world(self.forward, wldout,
                     world.n, world.ndims, param.lng, param.lat)

      free(imgcrd)
      free(phi)  
      free(theta)
      free(stat)
      if first_err:
         if not (self.invalid and self.allow_invalid):
            raise WCSerror, (first_err, first_msg)

      for i_c from 0 <= i_c < n_c:             # final restore of originals
         if not pixnan[i_c%param.naxis]:
            pixout[i_c] = pixin[i_c]
         if not wldnan[i_c%param.naxis]:
            wldout[i_c] = wldin[i_c]

      if self.gridmode:
         pix2grd(pixout, pixout, pixel.n, param, -1)
         free(pixin)
      result = (world.result(<long>wldout), pixel.result(<long>pixout))
      if world.dyn:
         free(wldout)
      if pixel.dyn:
         free(pixout)
      if self.invalid:
         self.result = result
         if not self.allow_invalid:
            raise WCSinvalid, (status, fmt_errmsg(param))
      return result

   def inside(self, coords, mode):
      """
      Test whether one or more coordinates are inside the area defined
      by the attribute :attr:`naxis`. This is a convenience method not
      directly related to WCS.
      *coords* is an object containing one or more coordinates
      which depending on *mode* can be either world- or pixel coordinates.
      The argument *mode* must be 'world' or 'pixel'. The method returns a value
      True or False or, in the case of multiple coordinates, a list with
      these values.
"""
      cdef double *data

      mode = mode[0].upper()
      ndim = len(self.naxis)
      result = []

      if mode=='W':
         save_allow = self.allow_invalid
         self.allow_invalid = True
         try:
            pixels = self.topixel(coords)
         finally:
            self.allow_invalid = save_allow
      elif mode=='P':
         pixels = coords
      else:
         raise ValueError, "mode must be 'world' or 'pixel'"

      coord = Coordinate(pixels, self.rowvec)
      data = <double*>void_ptr(coord.data)

      for i in range(coord.n):
         elem = i*ndim         
         is_in = True
         for idim in self.naxis:
            data_el = data[elem]
            if numpy.isnan(data_el) or data_el<0.5 or data_el>idim+0.5:
               is_in = False
               break
            elem += 1
         result.append(is_in)

      if len(result)==1:
         result = result[0]
      return result

   def pixel2grid(self, source, dir=-1):
      """
      Pixel-to-grid conversion.
      *pixel* is an object containing one or more pixel coordinates and
      a similar object with the corresponding grid coordinates will be
      returned.
      Grid coordinates are CRPIX-relative pixel coordinates,
      e.g. used in GIPSY. If CRPIX is not integer, the
      nearest integer is used as reference.
      """
      cdef wcsprm *param = <wcsprm*>void_ptr(self.wcsprm)
      cdef double *grid
      coord = Coordinate(source, self.rowvec)
      grid = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      pix2grd(<double*>void_ptr(coord.data), grid, coord.n, param, dir)
      result = coord.result(<long>grid)
      if coord.dyn:
         free(grid)
      return result
      
   def grid2pixel(self, source):
      """ 
      Grid-to-pixel conversion.
      *grid* is an object containing one or more grid coordinates and
      a similar object with the corresponding pixel coordinates will be
      returned.
      Grid coordinates are CRPIX-relative pixel coordinates,
      e.g. used in GIPSY. If CRPIX is not integer, the
      nearest integer is used as reference.
      """
      return self.pixel2grid(source, +1)

   def str2pos(self, postxt, mixpix=None):
      """
      This method accepts a string that represents one or more positions in the
      projection object's coordinate system.  If the string contains a
      valid position, the method returns the arrays with the corresponding
      world- and pixel coordinates. If a
      position could not be converted, then an error message is returned. 

      :param postxt:   one or more positions to be parsed.
      :type postxt:    string
      :param mixpix:   for a world coordinate system with one spatial
                       axis, a pixel coordinate for the missing
                       spatial axis is required to be able to convert between
                       world- and pixel coordinates.
      :type mixpix:    integer or None

      :Returns:

      This method returns a tuple with four elements:

      * a NumPy array with the parsed positions in world coordinates 
      * a NumPy array with the parsed positions in pixel coordinates  
      * A tuple with the units that correspond to the axes
        in your world coordinate system.
      * An error message when a position could not be parsed

      Each position in the input string is returned in the output as an
      element of a numpy array with parsed positions. A position has the same
      number of coordinates as there are axes in the data defined by
      the projection object.

      For its implementation, this method uses the function
      :func:`positions.str2pos` from module :mod:`positions`.
      Please refer to that module's documentation for a detailed explanation.
      """
      
      import positions
      return positions.str2pos(postxt, self, mixpix)

   def __setaxtypes(self):
      cdef wcsprm *param
      param = <wcsprm*>void_ptr(self.wcsprm)
      wcsset(param)

      self.units = ()
      self.ctype = ()
      self.crpix = ()
      self.crval = ()
      self.cdelt = ()
      self.crota = ()
      naxis = param.naxis
      types = []
      encoding = "ascii"
      for i in range(naxis):
         self.units += (param.cunit[i].decode(encoding),)
         self.ctype += (param.ctype[i].decode(encoding),)
         self.crpix += (param.crpix[i],)
         self.crval += (param.crval[i],)
         self.cdelt += (param.cdelt[i],)
         self.crota += (param.crota[i],)
         types.append(None)

      if (param.altlin & 4) == 0:                       # CROTA not found?
         self.crota = None

      if param.altlin & 2:                              # CDi_j found?
         dim = param.naxis
         ar = numpy.zeros(shape=(dim*dim,), dtype='d')
         for i in range(dim*dim):
            ar[i] = param.cd[i]
         ar.shape = (dim, dim)
         self.cd = numpy.matrix(ar)
      else:
         self.cd = None

      dim = param.naxis                                 # PCi_j
      ar = numpy.zeros(shape=(dim*dim,), dtype='d')
      for i in range(dim*dim):
         ar[i] = param.pc[i]
      ar.shape = (dim, dim)
      self.pc = numpy.matrix(ar)

      pvlist = []                                       # PVi_m
      for ipv in range(param.npv):
         pvlist.append((param.pv[ipv].i, param.pv[ipv].m, param.pv[ipv].value))
      self.pv = pvlist

      pslist = []                                       # PSi_m
      for ips in range(param.nps):
         pslist.append((param.ps[ips].i, param.ps[ips].m, param.ps[ips].value))
      self.ps = pslist

      if param.lng<0:
         self.lonaxnum = None
      else:
         self.lonaxnum = param.lng+1
         types[param.lng] = lontype
      if param.lat<0:
         self.lataxnum = None
      else:
         self.lataxnum = param.lat+1
         types[param.lat] = lattype
      if param.spec<0:
         self.specaxnum = None
      else:
         self.specaxnum = param.spec+1
         types[param.spec] = spectype
      self.types = tuple(types)
      self.cunit = self.units
      self.restfrq = param.restfrq
      self.restwav = param.restwav

# ==========================================================================
#                             Class Transformation
# --------------------------------------------------------------------------
#  Celestial transformation class
#
class Transformation(object):
   """
:param sky_in, sky_out:
   the input- and output sky system.  Can be
   specified as e.g., "(equatorial, fk4, 'B1950.0')" or "galactic". 
:param rowvec:
   if set to True, input and output coordinates, when given
   as NumPy matrices, will be row vectors instead of the standard column
   vectors.

**Method:**

.. automethod:: transform(in[, reverse=False])

**Attribute:**

.. attribute:: rowvec
   
   If set to True, input and output coordinates, when given as
   NumPy matrices, will be row vectors instead of the standard column
   vectors.

Example::
      
   #!/bin/env python
   from kapteyn import wcs
   import numpy

   tran = wcs.Transformation((wcs.equatorial, wcs.fk4, 'B1950.0'), wcs.galactic)

   radec = numpy.matrix(([33.3, 177.2, 230.1],
                         [66.2, -11.5,  13.0]))
          
   lbgal = tran(radec)
   print lbgal
   print tran(lbgal, reverse=True)

"""
   
   def __init__(self, skyin, skyout, rowvec=False):
      self.rowvec = rowvec
      self.forward = skymatrix(skyin, skyout)
      self.reverse = skymatrix(skyout, skyin)
      
   def __call__(self, source, reverse=False):
      return self.transform(source, reverse)
         
   def transform(self, source, reverse=False):
      """
:param in:
   an object containing one or more coordinates to be
   transformed and out will receive a similar object with the
   transformed coordinates.
   Coordinates can be specified in a number of different
   ways. See section :ref:`wcs-coordinates`.
:param reverse:
   if True, the inverse transformation will be performed.

Instead of calling this method, the object itself can also be called
in the same way.
"""

      cdef double *world, *c_xyz
      coord = Coordinate(source, self.rowvec)
      if coord.ndims!=2:
         raise WCSerror, (-5, '%d dimensions instead of 2' % coord.ndims)
      world = <double*>malloc(coord.n*coord.ndims*sizeof(double))
      xyz = numpy.matrix(numpy.zeros(shape=(coord.n,3), dtype='d'))
      c_xyz = <double*>PyArray_DATA(xyz)
      to_xyz(<double*>void_ptr(coord.data), c_xyz, coord.n, 2, 0, 1)
      if reverse:
         (m_trans, e_prae, e_post) = self.reverse
      else:
         (m_trans, e_prae, e_post) = self.forward
      if e_prae:
         eterms(c_xyz, coord.n, -1, e_prae[0], e_prae[1], e_prae[2])
      xyz *= m_trans.T
      if e_post:
         eterms(c_xyz, coord.n, +1, e_post[0], e_post[1], e_post[2])
      from_xyz(world, c_xyz, coord.n, 2, 0, 1)
      result = coord.result(<long>world)
      if coord.dyn:
         free(world)
      return result


#  names imported by *:

__all__ = ['equatorial', 'ecliptic', 'galactic', 'supergalactic',
           'fk4', 'fk4_no_e', 'fk5', 'icrs', 'j2000', 'WCSerror',
           'WCSinvalid', 'Projection', 'Transformation',
           'lontype', 'lattype', 'spectype', 'coordmap']
__version__ = '1.3'

