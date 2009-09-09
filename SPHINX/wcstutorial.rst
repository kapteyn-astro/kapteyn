Tutorial wcs module
===================

.. highlight:: python
   :linenothreshold: 15

Introduction
------------

This tutorial aims at starters. Experienced users find relevant but compact documentation
in the module API documentation. In this tutorial we address different practical
situations where we need to convert between pixel- and world coordinates. 
Many examples are working scripts, others are very useful to try in an interactive
Python session.

:mod:`wcs` is the core of the kapteyn package. An important feature of that package is
that it provides a world coordinate system which is easy to incorporate in your own
(Python) environment and :mod:`wcs` provides the basic methods to do this.
Together with module  :mod:`celestial` it allows a user to transform between pixel coordinates
and world coordinates for a set of supported projections and sky systems.
Module :mod:`celestial` provides a rotation matrix for sky transformations and
is more or less embedded in :mod:`wcs`, so (for standard work) there is no
need to import it separately.

.. index:: Features of the wcs module

Module :mod:`wcs` module has a number of important features:

   * Flexible I/O of coordinates
   * Support for spatial and spectral data
   * Support for 'mixed' coordinates
   * Support for conversions between different celestial systems
   * Objects have useful attributes
   * Easy to combine with other software written in Python


.. index:: Coordinate representations
.. index:: I/O structure

Coordinate representations
--------------------------

One coordinate axis
...................

For experiments and debug sessions, module :mod:`wcs` allows for very simple 
and flexible input and output of coordinates. This module interfaces with 
Mark Calabretta's
`WCSLIB <http://www.atnf.csiro.au/people/mcalabre/WCS/>`_ and is, because of the
flexible I/O, a valuable tool to test this well known library.
 
Main goal of module :mod:`wcs`  is to 
enable transformations between pixel coordinates and world coordinates
The pixel coordinates are defined by the FITS standard. The transformation 
is defined by meta data which are usually found in FITS headers.
So it may be obvious that FITS files play a central role in the use of
Module :mod:`wcs`.

However, FITS data processed by :mod:`wcs` are in fact FITS keywords that are
stored in a Python dictionary. This invites to experiment with WCSLIB even more
because one can create a (minimal) FITS header from scratch.
In an attempt to create the most simple use of :mod:`wcs` we started to write a
minimal FITS header. It defines only one axis. The minimal requirement 
for FITS keywords are CTYPE, CRVAL, CRPIX and CDELT. A description of
these keywords can be found in 
`The FITS standard <http://fits.gsfc.nasa.gov/fits_standard.html>`_.

We entered an axis type in *CTYPE1* that WCSLIB does not recognize as a 
known type. With this trick we force the system to do a linear transformation.
It shows that you have to be careful with values for CTYPE because 
you will not be warned if a CTYPE is not recognized. 
 
For the conversions between pixel coordinates and world coordinates we 
defined methods in a class which we called the :class:`wcs.Projection` class.
An object of this class is created using the header of the FITS file for
which we want WCS transformations. It accepts also a user defined
Python dictionary with FITS keywords and values. We use this last option
in this tutorial to be more flexible when we want to apply changes in the header.

The methods for single axes are called :meth:`wcs.Projection.toworld1d` and 
:meth:`wcs.Projection.topixel1d`.
FITS defines CRVAL as the world coordinate that corresponds to 
the pixel value in CRPIX. Let's check this with the most basic 
example we could think of::

   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'PARAM',
              'CRVAL1' : 5,
              'CRPIX1' : 10,
              'CDELT1' : 1
            }
   proj = wcs.Projection(header)
   print proj.toworld1d(10)
   
   # Output:
   # 5.0
   

Indeed, at pixel coordinate 10 (=CRPIX), the world coordinate is 5 (=CRVAL).
If we want to know which pixel coordinate corresponds to world coordinate 5, then we use 
``proj.topixel1d(5)`` to get the answer (which is the value of CRPIX: 10).
Note that we forced the system to apply linear transformations only.

In many of the examples that we present in this tutorial we included a so called 
*closure* test. This is a test which uses the result of a transformation
to test the inverse transformation which should result into the original value. 
Sometimes the result is not exactly what you expect because we work with
a limited number precision. A simple closure test is::

   proj = wcs.Projection(header)
   w = proj.toworld1d(10)
   p = proj.topixel1d(w)
   print "CRPIX: ", p
   
   # Output:
   # CRPIX:  10.0


Coordinate transformations are often done in bulk, so 
of course the transformation methods accept more than one coordinate to convert. 
They can be represented as a Python list, a Python tuple or a NumPy array. 
The representation of the output is the same as that of the input coordinates.
The output of the next statements therefore is not a surprise::
   
   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'PARAM',
              'CRVAL1' : 5,
              'CRPIX1' : 10,
              'CDELT1' : 1
            }
   
   proj = wcs.Projection(header)
   
   w1 = proj.toworld1d( range(9,12) )
   w2 = proj.toworld1d( [9,10,11] )
   w3 = proj.toworld1d( (9,10,11) )
   w4 = proj.toworld1d( numpy.array([9,10,11]) )
   print w1, type(w1)
   print w2, type(w2)
   print w3, type(w3)
   print w4, type(w4)
   closure = proj.topixel1d(w4)    # Closure test
   print closure, type(closure)

   # Output:
   # [4.0, 5.0, 6.0] <type 'list'>
   # [4.0, 5.0, 6.0] <type 'list'>
   # (4.0, 5.0, 6.0) <type 'tuple'>
   # [ 4.  5.  6.] <type 'numpy.ndarray'>
   # [ 9.  10. 11.] <type 'numpy.ndarray'>


The first two sequences are lists. The third is a tuple and the last is a NumPy array. 
The pixel coordinates  9, 10 and 11 should give values in the neighbourhood of 
*CRVAL1* and the step size is 1 (*CDELT1=1*), in arbitrary units.


.. note:: 
       
      The advantage of NumPy arrays is that you can use them in mathematical
      expressions to process the array content. 
      For example: assume you have a sequence of velocities in a numpy array V
      but want to express the numbers in km/s, then change the content with expression:
      V /= 1000

For representation purposes we often want to print a pixel coordinate and the corresponding
world coordinate on one line. Then we often use Pythons built-in function *zip* 
to combine two sequences to avoid a call to transformation methods in the 
print loop::

   p = range(5,15)
   w = proj.toworld1d(p)
   for pix,wor in zip(p,w):
      print "%d: %f" % (pix,wor)
   
   # Output:
   # 9: 4.000000
   # 10: 5.000000
   # 11: 6.000000

.. Note::

   Class wcs has an attribute called **debug**. 
   If you set its value to *True* then you get debug information from WCSLIB
   showing what has been correctly parsed from the given header data. 
   Use it as follows::

      wcs.debug = True
      proj = wcs.Projection(header)


Next we apply the procedures described above to a real example 
where we created an artificial header with FITS data. The header describes
a single axis of spectral type. Units are standard FITS units and are given
in keyword *CUNIT1*. The example shows that we can access the keywords
from the artificial header (or a real FITS header) directly and use their 
values for example to find the length of the axis in pixels, or to find the units of 
the world coordinates of that axis::


   #!/usr/bin/env python
   from kapteyn import wcs
   header  = { 'NAXIS'  : 1,
               'NAXIS1' : 64,
               'CTYPE1' : 'FREQ',
               'CRVAL1' : 1.37835117405e9,
               'CRPIX1' : 32,
               'CUNIT1' : 'Hz',
               'CDELT1' : 9.765625e4
            }
   proj = wcs.Projection(header)
   n = header['NAXIS1']               # Get the length of the spectral axis
   p = range(1, n+1)                  # Set pixel range accordingly
   w = proj.toworld1d(p)              # Do the transformation
   print "Pixel  %s (%s)" % (header['CTYPE1'],header['CUNIT1'])
   for pix,frq in zip(p,w):
      print "%5d: %f" % (pix,frq)
   
   # Output:
   # Pixel  FREQ (Hz)
   #   1: 1375323830.300000
   #   2: 1375421486.550000
   #   3: 1375519142.800000
   #   4: 1375616799.050000
   #   5: 1375714455.300000


In the example we wanted to make a table with pixel coordinates and
the corresponding world coordinates. According to the header there are 64 pixels (*NAXIS1*)
along the axis so the first pixel coordinate is 1 and the last is 64. The axis
represents frequencies. A start frequency is given by *CRVAL1* and a step size
is given by *CDELT1*. Note that the coordinate transformation is linear.


Generic methods *toworld()* and *topixel()*
...........................................

The methods :meth:`wcs.Projection.toworld1d` and :meth:`wcs.Projection.topixel1d`
are special versions of the more general methods :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`.
These methods can be used to convert pixel data for more than one axis at the same time which
is necessary for coupled axes, for example in spatial maps where longitude and latitude
are not independent axes.

These general methods :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
accept the same sequences as the '1d' versions. 
The reason that we introduced the '1d' versions is that for non-experienced Python programmers
it usually is confusing that in the one dimensional case the general methods only
accept tuples and not scalars and that a tuple with one element (for example 10) needs to be
written as `(10,)`.

If you want to replace method toworld1d() by topixel1d() in the first example, then the relevant lines become:

>>> p = proj.toworld( (10,) )
>>> (5.0,)

for one scalar and for a list of values:

>>> p = proj.toworld( (range(9,12),) )
>>> ([4.0, 5.0, 6.0],)

If you want to extract the scalar or the list from the tuple, use element 0 of the tuple.

>>> p = proj.toworld( (range(9,12),) )
>>> print p[0]
>>> [4.0, 5.0, 6.0]


Two coordinate axes
...................

As described in the previous section we use
:meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
if the number of axes in our data is more than 1.
The input and output tuples for projection objects with two coordinate
axes consist of two elements. The first element corresponds to the first axis in
the projection object and the second element to the second axis.
The following Python code constructs an artificial header which describes the world coordinate
system of two spatial axes. Then we want to find the world coordinates of the
reference pixels (*CRPIX1*, *CRPIX2*) and expect the reference values (*CRVAL1*, *CRVAL2*)
as output tuple::

   #!/usr/bin/env python
   from kapteyn import wcs
   header  = { 'NAXIS'  : 2,
               'NAXIS1' : 5,
               'CTYPE1' : 'RA---NCP',
               'CRVAL1' : 45,
               'CRPIX1' : 5,
               'CUNIT1' : 'deg',
               'CDELT1' : -0.01,
               'NAXIS2' : 10,
               'CTYPE2' : 'DEC--NCP',
               'CRVAL2' : 30,
               'CRPIX2' : 5,
               'CUNIT2' : 'deg',
               'CDELT2' : +0.01,
            }
   proj = wcs.Projection(header)
   pixel = (5,5)
   world = proj.toworld(pixel)
   print world
   
   # Output:
   # (45.0, 30.0)


Comments about the composed header:
the header is composed from scratch.
but it could very well have been copied from an existing FITS header.
In either case you should verify items **CUNITn** and **CTYPEn** because they are are important.
In section 2.1.1 of [Ref1]_ we read that in WCSLIB:
   
.. note::
   *any CTYPEi not covered by convention and agreement shall be taken to be linear*.
   
The CTYPE consists of a coordinate type (max 4 characters) followed by '-'
followed by a three character code that represents the algorithm to calculate
the world coordinates ('ABCD-XYZ'). Shorter coordinate types are padded
with the '-' character, shorter algorithm codes are padded on the right
with blanks ('RA---NCP', 'RA---UV\_ '). So if we were sloppy and
wrote RA--NCP and DEC-NCP then WCSLIB assigns a linear conversion algorithm.
It does not complain, but you get unexpected results. If your CTYPE's are correct
but the units are not standard and are not recognized by WCSLIB, then you get
an Python exception after you tried to create the Projection object.
For example, if you specified CUNIT1='Degree' then the error message displayed by the
exception is:
*"Invalid coordinate transformation parameters".*

If you want to be sure that WCSLIB recognizes your coordinate type and unit,
you can print the Projection attributes :attr:`wcs.Projection.types` and :attr:`wcs.Projection.units`
as in the example below. Unrecognized types are returned as `None`.

>>> proj = wcs.Projection(header)
>>> print "WCS units: ",proj.units
    WCS units:  ('deg', 'deg')
>>> print "WCS type: ",proj.types
    WCS type:  ('longitude', 'latitude')


With the same variable *header* as in the previous script we
demonstrate that each element in the coordinate tuple can be a list of scalars.
Let's convert pixel positions (3,3), (4,4), ..., (7,7) etc. to their corresponding world coordinates::

   proj = wcs.Projection(header)
   x = range(3,8)
   y = range(3,8)
   pixel = (x,y)
   world = proj.toworld(pixel)
   print world
   
   # Output:
   # ([45.023089356221305, 45.011545841750113, 45.0, 44.988451831142257, 44.97690133535837], 
   #  [29.979985885372404, 29.989996472289789, 30.0, 30.009996474046854, 30.019985899953429])


The output is a tuple with *two* elements. Each element is a list. The first list contains the longitude
coordinates for input pixel coordinates (3,3), (4,4) etc. The second list contains the
latitude coordinates for the input pixel coordinates (3,3), (4,4) etc.

.. note::
   Note that longitude and
   latitude are not independent. You need always two pixel coordinates (x,y) to get a
   world coordinate pair (RA,DEC).

Here input and output coordinates for the methods :meth:`wcs.Projection.toworld` 
and :meth:`wcs.Projection.topixel`
are tuples. The dimension of the tuple corresponds to the number of axes in the Projection object,
and each element in the tuple can be a list of scalars.
In some situations it is more intuitive to start with a list of 2 dimensional positions.
The Python interface to WCSLIB allows for this type of input.
You can get the same coordinate output as the previous script if you replace the body by::

   proj = wcs.Projection(header)
   pixels = [(3,3), (4,4), (5,5), (6,6), (7,7)]
   world = proj.toworld(pixels)
   print world
   
   # Output:
   # [(45.023089356221305, 29.979985885372404), (45.011545841750113, 29.989996472289789), (45.0, 30.0), 
   # (44.988451831142257, 30.009996474046854), (44.97690133535837, 30.019985899953429)]


Note that the representation of the output differs from the previous script
because the representation of the input differs, i.e.: a list with tuples.
The dimension of the tuples being the number of axes in your projection object.

.. note::
   The coordinate representation in methods :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
   of the output is the same as that of the input.


.. index:: Mixing pixel- and world coordinates

Mixed transformations (pixel- and world coordinates) using method :meth:`wcs.Projection.mixed`
..............................................................................................

We describe the mixed() method in some detail in the section about data sets with
three or more axes. Here we show how to use the method in a simple case.
Suppose you want to mark data in a plot at constant declination in pixels
(i.e. parallel to the x-axis of the plot) but with equal steps in Right Ascension,
then you need method :meth:`wcs.Projection.mixed`::

   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy
   header  = { 'NAXIS'  : 2,
               'NAXIS1' : 5,
               'CTYPE1' : 'RA---TAN',
               'CRVAL1' : 45,
               'CRPIX1' : 5,
               'CUNIT1' : 'deg',
               'CDELT1' : -0.01,
               'NAXIS2' : 10,
               'CTYPE2' : 'DEC--TAN',
               'CRVAL2' : 30,
               'CRPIX2' : 10,
               'CUNIT2' : 'deg',
               'CDELT2' : +0.01,
            }
   proj = wcs.Projection(header)
   # 1 pixel and 1 world coordinate pair
   pixel_in = (numpy.nan, 10)
   world_in = (45.0, numpy.nan)
   world_out, pixel_out = proj.mixed(world_in, pixel_in)
   print world_out
   print pixel_out
   
   # Output:
   # (45.0, 30.0)
   # (5.0, 10.0)

   # A loop over a number of Right Ascensions at constant Declination
   for ra in range(44, 47):
      world_in = (ra,numpy.nan)
      world_out, pixel_out = proj.mixed(world_in, pixel_in)
      print "World: ", world_out, "Pixel: ", pixel_out
   
   # Output:
   # World:  (44.0, 29.99622120337045) Pixel:  (91.61133499750801, 10.000000000096229)
   # World:  (45.0, 30.0) Pixel:  (5.0, 10.0)
   # World:  (46.0, 29.99622120337045) Pixel:  (-81.61133499750801, 10.000000000096248)


First we have a pixel position of which the x coordinate is set to *unknown*. We use
a special value for this: `numpy.nan` which is the representation of NumPy's Not A Number.
The y coordinate is set to 10. For the :meth:`wcs.Projection.mixed`, we need to specify
the *unknown* values in the pixel position with a world coordinate. In the example
we entered 45.0 (deg). The mixed() method returns two tuples. One for the pixel position
and one for the position in world coordinates. The *unknown* values are calculated in an
iterative process.
The second part of the example is a loop over a number of world coordinates in Right Ascension,
and a constant pixel coordinate in the y-direction (i.e. 10). The output (as listed as comment
in the code) shows two things that need to be addressed.
First we notice that the output pixel is not exactly 10. This is related to finite
precision of numbers when a solution is calculated in an iterative way.
The second observation is more important: the
Declination varies while the y coordinate in pixels is constant. But this is exactly
what we expect for spatial data when a projection is involved.

A note about efficiency:

.. note::
   The transformation routines accept sequences of coordinates.
   Calculations with sequences are more efficient than repetitive calls in a loop.

So in our example it is more efficient to avoid the loop over the right ascensions.
This can be done by creating an input tuple with two lists.
The output is the same as in the example above, but the representation is different.
As we stated earlier, the representation of the output is the same as the
representation of the input (a tuple with two lists)::

   # As example above but without a loop
   ra = range(44, 47)
   dec = [numpy.nan]*len(ra)  # NumPy trick to repeat elements in a list.
   world_in = (ra, dec)
   x = [numpy.nan]*len(ra)
   y = [10]*len(ra)
   pixel_in = (x, y)
   world_out, pixel_out = proj.mixed(world_in, pixel_in)
   print world_out
   print pixel_out
   
   # Output:
   # ([44.0, 45.0, 46.0], [29.99622120337045, 30.0, 29.99622120337045])
   # ([91.61133499750801, 5.0, -81.61133499750801], [10.000000000096229, 10.0, 10.000000000096248])



.. index:: Projection objects representing data slices
.. index:: Sub-Projections

Three or more coordinate axes
.............................

In this section we discuss method :meth:`wcs.Projection.sub` 
which allows us to define coordinate transformations
for positions with less dimensions than the dimension of the data structure.
In practice we encounter many astronomical measurements based on three or more independent axes.
Well known examples are of course the data sets from radio interferometers.
Usually these are spatial maps observed at different frequencies and sometimes
as function of Stokes parameter (polarization). If we are only interested in
spatial maps and don't bother about the other axes,
we can create a Projection object with only the relevant axes.
This is done with the  :meth:`wcs.Projection.sub` method from the Projection class.

`map = proj.sub(axes=None, nsub=None)`

The method has two parameters. You can specify parameter *nsub* which sets the first
*nsub* axes from the original Projection object to the actual axes.
Or you can use the other parameter axes which is a tuple or a list with axis numbers.
Axis numbers in WCSLIB follow the fits standard so they start with 1.
The order in the sequence is important. The axis description sequence in a
FITS file is not bound to rules and luckily WCSLIB accepts permuted axis number sequences.
This can be illustrated with the next example.
First we show the code and then explain the output::

   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy
   header  = { 'NAXIS'  : 3,
               # First spatial axis
               'NAXIS1' : 5,
               'CTYPE1' : 'RA---TAN',
               'CRVAL1' : 45,
               'CRPIX1' : 5,
               'CUNIT1' : 'deg',
               'CDELT1' : -0.01,
               # A dummy axis
               'NAXIS2' : 5,
               'CTYPE2' : 'PARAM',
               'CRVAL2' : 444,
               'CRPIX2' : 99,
               'CDELT2' : 1.0,
               'CUNIT2' : 'wprf',
               # Second spatial axis
               'NAXIS3' : 0,
               'CTYPE3' : 'DEC--TAN',
               'CRVAL3' : 30,
               'CRPIX3' : 10,
               'CUNIT3' : 'deg',
               'CDELT3' : +0.01
            }
   proj = wcs.Projection(header)
   map = proj.sub( [1,3] )
   pixel = (header['CRPIX1'], header['CRPIX3'])
   world = map.toworld(pixel)
   print world
   
   # Output:
   # (45.0, 30.0)
   
   map = proj.sub( [3,1] )
   pixel = (header['CRPIX3'], header['CRPIX1'])
   world = map.toworld(pixel)
   print world
   
   # Output:
   # (30.0, 45.0)
   
   line  = proj.sub( 2 )
   crpix = header['CRPIX2']
   pixels = range(crpix-5,crpix+6)
   world = line.toworld1d(pixels)
   print world
   
   # Output:
   # [439.0, 440.0, 441.0, 442.0, 443.0, 444.0, 445.0, 446.0, 447.0, 448.0, 449.0]
   

We created a header representing a spatial map as function of some parameter
along the CTYPE2='PARAM' axis. This axis is not recognized by WCSLIB and a
linear transformation is applied. Also special is that the spatial axes do not
have conventional numbers. First we want to set up a transformation of
pixel (x,y) to (R.A., Dec) for the pixel values in (CRPIX1, CRPIX3) -which should transform to (CRVAL1, CRVAL3)-.
Then we reverse the spatial axis sequence to set up a transformation
from (y,x) to (Dec, R.A.). Finally we want a transformation only for the PARAM axis.
Its axis number is 2. With the output we show that for this axis indeed the
transformation between pixels and world coordinates is a linear. transformation.


The axis sequence in the :meth:`wcs.Projection.sub` method sets the axis order with parameter *axes*.
It sets in fact the order of the coordinates in the transformation methods :meth:`wcs.Projection.toworld`,
:meth:`wcs.Projection.topixel` and :meth:`wcs.Projection.mixed`.
Parameter *axes* is either a single integer or a list/tuple of integers e.g. sub(2) vs. sub([3,1]).

.. index:: Data in Numpy arrays
.. index:: Data in a Numpy matrix

NumPy arrays and matrices
-------------------------

NumPy matrices
..............

In many Python applications programmers use NumPy arrays and matrices because it is easy
to manipulate them. First let's explore what can be done with a NumPy matrix as
coordinate representation. A NumPy matrix is a rank 2 array with  special properties. 
The first list in
the numpy.matrix() constructor in the next example is the first row in the matrix
and the second list is the second row. The first row contains the x coordinate of
the pixels and the second row contains the y coordinates.
In the next script we want to convert pixel positions (4,5), (5,5) and (6,5) to
world coordinates. So the first list in the matrix constructor are the x coordinates [4,5,6]
and the second are the y coordinates [5,5,5]. We convert these with::

   proj = wcs.Projection(header)
   pixel = numpy.matrix( [[4,5,6],[5,5,5]] )
   world = proj.toworld(pixel)
   print world
   # Output:
   # [[ 45.01154701  45.          44.98845299]
   # [ 29.99999798  30.          29.99999798]]

   pixel = proj.topixel(world)
   print pixel
   
   # Output:
   # [[ 4.00000001  5.          5.99999999]
   # [ 5.          5.          5.        ]]


The output is what we expected. It is a NumPy matrix with two rows.
The first row contains the longitudes and the second the latitudes.
The numbers seem ok (three RA's at almost constant declination).
We added a closure test by using the output world coordinates as input
for the :meth:`wcs.Projection.topixel` method. As you can see, the closure test
returns the original input.

There is also a matrix representation that is equivalent to the list of coordinate
tuples in the previous section.
We want an input matrix to contain the coordinates: `[[4,5],[5,5],[6,5]]`.
For this representation you have to set an attribute of the projection object.
The name of the attribute is :attr:`wcs.Projection.rowvec`. Its default value is `False`.
When you set it to `True` then each row in the matrix represents a position in x and y.
Here is an example::
   
   proj = wcs.Projection(header)
   proj.rowvec = True
   pixel = numpy.matrix( [[4,5],[5,5],[6,5]] )
   world = proj.toworld(pixel)
   print world
   
   # Output:
   # [[ 45.01154701  29.99999798]
   # [ 45.          30.        ]
   # [ 44.98845299  29.99999798]]
   
   pixel = proj.topixel(world)
   print pixel
   
   # Output:
   # [[ 4.00000001  5.        ]
   # [ 5.          5.        ]
   # [ 5.99999999  5.        ]]


.. note::
   The rowvec attribute can also be set in the constructor of the projection object as follows:
   `proj = wcs.Projection(header, rowvec=True)`


NumPy arrays
............

It is possible to build a NumPy array with x coordinates and another for the y coordinates.
You can use these arrays in a tuple. Then the elements in the tuple are not lists, as in
the previous section, but NumPy arrays.
With the same example in mind as the one with the NumPy matrix we demonstrate this
option in the following script::
   
   proj = wcs.Projection(header)
   x = numpy.array( [4,5,6] )
   y = numpy.array( [5,5,5] )
   pixel = (x, y)
   world = proj.toworld(pixel)
   print world
   
   # Output:
   # (array([ 45.01154701,  45. ,  44.98845299]), array([ 29.99999798,  30. ,  29.99999798]))

   pixel = proj.topixel(world)
   print pixel
   
   # Output:
   # (array([ 4.00000001,  5.        ,  5.99999999]), array([ 5.,  5.,  5.]))


As you can see, the representation of the output is the same as that of the input.
The result is a tuple and the elements of the tuple are 1 dimensional (rank 1, shape N) NumPy arrays.
The first array contains the RA's and the second the Dec's.
The closure test also gives the expected result.


Using NumPy arrays to convert an entire map
...........................................

For applications that transform all the positions in a data set (or in a subset of the data)
in one run (e.g. for re-projections of images), it is possible to store all the positions
in a NumPy array with shape (NAXIS2, NAXIS1, 2) (note the order).
The array can be handled by the :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel` in one step.
You could say that we have a two-dimensional array of which the elements are coordinate pairs.
The example code below could be part of the body of a real application that re-projects an image::

   from kapteyn import wcs
   import numpy
   
   header = {  'NAXIS'  : 2,
               'NAXIS1' : 5,
               'CTYPE1' : 'RA---TAN',
               'CRVAL1' : 45,
               'CRPIX1' : 5,
               'CUNIT1' : 'deg',
               'CDELT1' : -0.01,
               'NAXIS2' : 10,
               'CTYPE2' : 'DEC--TAN',
               'CRVAL2' : 30,
               'CRPIX2' : 10,
               'CUNIT2' : 'deg',
               'CDELT2' : +0.01,
            }

   proj = wcs.Projection(header)
   n1 = 10
   n2 = 8
   pixel = numpy.zeros(shape=(n2,n1,2))
   for y in xrange(n2):
      for x in xrange(n1):
         pixel[y, x] = (x+1, y+1)
   
   world = proj.toworld(pixel)
   print world
   
   # Output:
   # [[[ 45.04614616  29.90999204]
   #   [ 45.03460962  29.90999556]
   #   [ 45.02307308  29.90999807]
   #   [ 45.01153654  29.90999957]
   # etc.
   
   pixel = proj.topixel(world)
   print pixel
   
   # Output:
   # [[[  1.   1.]
   #   [  2.   1.]
   #   [  3.   1.]
   #   [  4.   1.]
   # etc.


In this example we have NAXIS2=10 y values and NAXIS1=5 x values.
The indices start at 0, but the FITS pixel indices start at 1.
That's why the coordinate tuple reads as (x+1, y+1).

.. note::
   In this module the values in the NumPy arrays and matrices are of type 'f8' (64 bit).

.. index:: Attributes of a Projection object

Attributes
----------

Attributes lonaxnum, lataxnum and specaxnum
...........................................

In the previous examples we had foreknowledge of the axis numbers that represented a spatial
axis or a spectral axis. If you read a header from a FITS file then it is not always obvious
what the axes represent and in which order they are stored in the FITS header.
In those circumstances the projection attributes :attr:`wcs.Projection.lonaxnum`,
:attr:`wcs.Projection.lataxnum` and :attr:`wcs.Projection.specaxnum` are very useful.
These attributes are axis numbers, i.e. they start with 1 and the highest number is
equal to header item 'NAXIS'.
In the source below we provide a header which shows an unexpected axis order representing
a number of spatial maps as function of frequency. For demonstration purposes we create
two separate Projection objects. The first, called *line*, represents the spectral axis.
This is a sub projection of the parent projection object and the axis number is that of the
spectral axis. We add a spectral translation to get velocities in the output.

The second, called *map*, is the spatial map with axis longitude first and latitude second.
We try to create these objects in a try/except clause. For any header, this results in
the requested sub projections for a spatial map and spectral axis or an error message
and an exception.
The construction with the attributes and the try/except clause saves us tedious work because without,
we need to find and inspect the axis numbers ourselves.

.. note::
   If WCSLIB cannot find a value of one of the requested attributes, its value is set to `None`

::
         
   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 3,
              'NAXIS3' : 5,
              'CTYPE3' : 'RA---NCP',
              'CRVAL3' : 45,
              'CRPIX3' : 5,
              'CUNIT3' : 'deg',
              'CDELT3' : -0.01,
              'CTYPE2' : 'FREQ',
              'CRVAL2' : 1378471216.4292786,
              'CRPIX2' : 32,
              'CUNIT2' : 'Hz',
              'CDELT2' : 97647.745732,
              'RESTFRQ': 1.420405752e+9,
              'NAXIS1' : 10,
              'CTYPE1' : 'DEC--NCP',
              'CRVAL1' : 30,
              'CRPIX1' : 15,
              'CUNIT1' : 'deg',
              'CDELT1' : +0.01
            }
   try:
      proj = wcs.Projection(header)
      line = proj.sub(proj.specaxnum).spectra('VRAD')
      map  = proj.sub( (proj.lonaxnum, proj.lataxnum) )
   except:
      print "Could not find a spatial map AND a spectral line!"
      raise
   
   print proj.lonaxnum, proj.lataxnum, proj.specaxnum
   
   # Output:
   # 3 1 2

   # A transformation along the spectral axis:
   pixels = range(30, 35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   
   # Output:
   # 30 8891.97019336
   # 31 8871.36054878
   # 32 8850.75090419
   # 33 8830.14125961
   # 34 8809.53161503

   # A transformation of a coordinate in a spatial map:
   ra  = header['CRVAL'+str(proj.lonaxnum)]
   dec = header['CRVAL'+str(proj.lataxnum)]
   print map.topixel( (ra,dec) )
   
   # Output:
   # (5.0, 15.0)
   
   # Are these indeed the CRPIXn?
   ax1 = "CRPIX"+str(proj.lonaxnum)
   ax2 = "CRPIX"+str(proj.lataxnum)
   print map.topixel( (ra,dec) ) == (header[ax1], header[ax2])
   
   # Output:
   # True
   

Note the check at the end of the code.  It should return `True`. We started with world coordinates
equal to the values of CRVALn from the header and we assert that these correspond
to pixel values equal to the corresponding CRPIXn.

.. index:: Position-Velocity plots
.. index:: XV maps

Two dimensional data slices with only one spatial axis
.......................................................

Suppose we have a 3D data set with CTYPE's: (RA---NCP, DEC--NCP, VOPT-F2W) and we want to
write coordinate labels in a plot that represents the data as function of one spatial axis
and the spectral axis (usually called a position-velocity plot or XV map)? It is obvious that
we need extra information about the spatial axis that is left out.
Usually this is a pixel position that corresponds to the position on the missing
axis along which a data slice is taken. These data slices are fixed on
pixel coordinates and not on world coordinates.

Assume the XV data we want to plot has axis types DEC--NCP and VOPT-F2W, then we need to specify
at which pixel coordinate in Right Ascension the data is extracted.

What we need is a sub-projection (i.e. a Projection object which is modified by method *sub()*)
which represents the WCSLIB types:
('latitude', 'spectral', 'longitude').
Given the CTYPE's from the header, the axis permutation sequence that is needed for
the sub projection is (2,3,1).
Now we require a method that for instance calculates for a given world coordinate
in Declination (e.g. 60.1538880206 deg) and a velocity (e.g. -243000.0 m/s)
and a fixed pixel for R.A. (e.g. 51) the corresponding pixel coordinates.

The required method is called :meth:`wcs.Projection.mixed`. In a previous section we discussed
its use. Method *mixed()* has for a Projection object *p* the following
syntax and parameters.

`world, pixel = p.mixed(world, pixel, span=None, step=0.0, iter=7)`

It is a hybrid transformation suited for celestial coordinates.
It uses an iterative method to find an unknown pixel- or world coordinate.
The iteration is controlled by parameters span, step and iter.
They have reasonable defaults which usually give good results.
The method needs knowledge about elements that need to be solved. Unknown values that
need to be solved are initially set to NaN (i.e. numpy.nan).

With the numbers we listed, the input world coordinate tuple will be
`world_in = (60.1538880206, -243000.0, numpy.nan)`.
The input pixel tuple will be: `pixel_in = (numpy.nan, numpy.nan, 51)`
then we find the missing coordinates after applying the lines::
   
   subproj = proj.sub([2,3,1])
   world_in = (60.1538880206, -243000.0, numpy.nan)
   pixel_in = (numpy.nan, numpy.nan, 51)
   world_out, pixel_out = subproj.mixed(world_in, pixel_in)
   print "world_out = ", world_out
   # world_out = (60.1538880206, -243000.0, -51.282084795900005)
   print "pixel_out = ", pixel_out
   # pixel_out = (51.0, -20.0, 51.0)
   
The *mixed()* method in wcs is more powerful than its equivalent in the C-version
of WCSLIB. It accepts the same coordinate representations as for *topixel()* and *toworld()*
whereas the library version accepts only one coordinate pair per call.


.. index:: Suppressing exceptions in coordinate transformations
.. index:: Exception suppression

Invalid coordinates
-------------------

Suppress exceptions for invalid coordinates
...........................................

We introduced matrices and arrays as coordinate representations to facilitate
the input and output of many coordinates in one call. This is in many practical
situations the most efficient way to process those coordinates.
However if there is a pixel coordinate in a sequence that could not be converted
to a world coordinate then an exception will be raised and your script will stop.
One can suppress the exception and flag the unknown coordinate. You need to set the
:attr:`wcs.Projection.allow_invalid` attribute of the projection object.
Invalid coordinates then are flagged in the output with a NaN (i.e. numpy.nan).
On the other hand, if the input contains a NaN, the corresponding converted
coordinate will also be a NaN. You can test whether a value is a NaN with
function *numpy.isnan()*. NaN's cannot be compared so a simple test as in:

>>> x = numpy.nan
>>> if x == numpy.nan:
      
will fail because the result is always `False` 

In practice it will be difficult to get into problems if you convert from world coordinates
to pixel coordinates, but when you start with pixel coordinates then it is possible
that a corresponding world coordinate is not available. For a projection like Aitoff's projection
it is obvious that the rectangle in which an all sky map in this the projection is enclosed,
contains such pixels.

Here is an example how one can deal with invalid transformations::
      
   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy
   header = { 'NAXIS'  : 2,
              'NAXIS1' : 5,
              'CTYPE1' : 'RA---AIT',
              'CRVAL1' : 45,
              'CRPIX1' : 5,
              'CUNIT1' : 'deg',
              'CDELT1' : -0.01,
              'NAXIS2' : 10,
              'CTYPE2' : 'DEC--AIT',
              'CRVAL2' : 30,
              'CRPIX2' : 5,
              'CUNIT2' : 'deg',
              'CDELT2' : +0.01,
            }
   proj = wcs.Projection(header)
   proj.allow_invalid = True
   pixel_in = numpy.matrix( [[4000,5000,6000],[5000,5000,7580]] )
   world = proj.toworld(pixel_in)
   print "World coordinates:\n",world
   pixel_out = proj.topixel(world)
   print "Back to pixels:\n", pixel_out
   
   if numpy.isnan(pixel_out).any():
      print "Some pixels could not be converted"
   
   indices = numpy.where(numpy.isnan(pixel_out))
   print "Index of NaNs: ", indices
   print pixel_in[indices]

.. index:: Reading headers from FITS files
.. index:: Header data from a FITS file

Reading data from a FITS file
-----------------------------

Reading a FITS header
.....................

Until now, we created our own header which was a Python dictionary which could be processed by
the :mod:`wcs` module.
Usually our starting point is a FITS file.
A FITS file can contain more than one header. Header data is read from a FITS file with methods from
module :mod:`pyfits`.
Select the unit you want and store it in a variable (like *header*) so that it can be parsed by wcs.
Below we demonstrate how to read the first header from a FITS file. 
.. index:: Reading headers from FITS files (example)

A flag is set to enter WCSLIB's debug mode::

   #!/usr/bin/env python
   from kapteyn import wcs
   import pyfits
   
   wcs.debug = True
   f = raw_input('Enter name of FITS file: ')
   hdulist = pyfits.open(f)
   header = hdulist[0].header
   proj = wcs.Projection(header)
   
   # Part of the output of arbitrary FITS file:
   # Output:
   #      flag: 137
   #      naxis: 3
   #      crpix: 0x99b53d8
   #               51           51          -20
   #         pc: 0x99adf10
   #   pc[0][]:   1            0            0
   #   pc[1][]:   0            1            0
   #   pc[2][]:   0            0            1
   #      cdelt: 0x99b71c8
   #            -0.007166     0.007166     4200
   #      crval: 0x992bd30
   #            -51.282       60.154      -2.43e+05
   #      cunit: 0x99ad768
   #            "deg"
   #            "deg"
   #            "m/s"
   #      ctype: 0x999a7f8
   #            "RA---SIN"
   #            "DEC--SIN"
   #            "VELO"
   

For testing and debugging one often wants to inspect the items in a FITS header.
PyFITS has a nice method to make a list with all the FITS cards.
In the next example we added a little filter, using list comprehension,
to filter all items that start with 'HISTORY'. Also we added output for the
two projection attributes :attr:`wcs.Projection.types` and :attr:`wcs.Projection.units`.
The script is a useful tool to inspect the FITS file and to check its parsing by WCSLIB::
   
   #!/usr/bin/env python
   from kapteyn import wcs
   import pyfits
   
   f = raw_input('Enter name of FITS file: ')
   hdulist = pyfits.open(f)
   header = hdulist[0].header
   clist = header.ascardlist()
   c2 = [str(k) for k in header.ascardlist() if not str(k).startswith('HISTORY')]
   for i in c2:
      print i
   
   proj = wcs.Projection(header)
   print "WCS found types: ", proj.types
   print "WCS found units: ", proj.units


Reading WCS data for a spatial map
..................................

For some world coordinate related applications we want to force the input to represent a spatial map.
A spatial map has axes of type longitude and latitude. For example if you need
to re-project a map from one projection system to another, then you need a matching axis pair,
representing a spatial system. If you don't know beforehand what the numbers are of the axes
in your FITS file that represent these types, you need a way of checking this.
There are some rules. First, we must be able to create a Projection object according
to the WCSLIB rules (i.e. the axes must have a valid name and extension).
For spatial axes, WCSLIB also requires a matching axis pair.
So if you have a FITS file with a R.A. axis and not a Dec axis then module :mod:`wcs` will
generate an exception with the message *Inconsistent or unrecognized coordinate axis types*.

Finally, if you have a valid header and made a Projection object, then you still have to find
the axis numbers that represent a 'longitude' axis and a 'latitude' axis
(remember: the number of axes in your data could be more than 2) and the latitude axis
could be defined earlier than the longitude axis so the order is also important.

In a previous section we discussed the attributes :attr:`wcs.Projection.lonaxnum`
and :attr:`wcs.Projection.lataxnum`. They can be used to find the requested spatial axis
numbers (remember their value is `None` if the requested axis is not available).
In the following script we try to create the Projection and sub Projection objects with
Python's try/except mechanism.

.. index:: FITS: Creating a Projection object for a spatial map in a FITS file (example)

If we have a valid projection and the right axes,
then we check the axes types (and order) with attribute `wcs.Projection.types`::
   
   #!/usr/bin/env python
   from kapteyn import wcs
   import pyfits
   
   f = raw_input('Enter name of FITS file: ')
   hdulist = pyfits.open(f)
   header = hdulist[0].header
   try:
      proj = wcs.Projection(header)
      map = proj.sub((proj.lonaxnum, proj.lataxnum))
   except:
      print "Aborting program. Could not find (valid) spatial map."
      raise
   
   # Just a check:
   print map.types


Celestial transformations with wcs
----------------------------------

Celestial systems
.................

Methods :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
convert between pixel coordinates and world coordinates. If these
world coordinates are spatial, they are calculated for the sky- and reference system
as defined in the header (FITS header, GIPSY header, header dictionary).
To compare positions one must therefore ensure that these positions
are all defined in the same sky- and reference system.
If such a position is given in another system (e.g. galactic instead of equatorial),
then you have to transform the position to the other sky- and/or reference system.
Sometimes you might find a so called *alternate* header in the header information
of a FITS file. In an alternate header the WCS related keywords end on a character
(e.g. CRVAL1A).

Usually these alternate headers describe a world coordinate system for another
sky system. But because there could also be different epochs involved, it is
worthwhile to have a system that can transform world coordinates between
sky- and reference systems and that can do epoch transformations as well.

For the Kapteyn Package we wrote module :mod:`celestial`. This module can be used
as stand alone module if one is interested in celestial transformations of world
coordinates only.
But the module is well integrated in module :mod:`wcs` so one can use it
in the context of :mod:`wcs`, i.e. it defines a class :class:`wcs.Transformation`.
for conversions of world coordinates between sky-/reference systems
and also, if pixel coordinates are involved, methods
:meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`
can interpret an alternative sky-/reference system as the system for which
a coordinate has to be calculated.
The alternative sky-/reference system is stored in attribute
:attr:`wcs.projection.skyout`.

.. note::
   If you need transformations of world coordinates between any of the supported
   input sky-/reference system then you should use objects and methods
   from class :class:`wcs.Transformation`.

   If you need to convert pixel coordinates in a system defined by (FITS)
   header information, then set the **skyout** attribute of a Projection
   object and use methods
   :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`

The celestial definitions are described in detail in the background information of
module :mod:`celestial`. We list the most important features of a
celestial definition:
   
Supported Sky systems (detailed information in :ref:`celestial-skysystems`):

   1. Equatorial: Equatorial coordinates (α, δ), see next list with reference systems
   2. Ecliptic: Ecliptic coordinates (λ, β) referred to the ecliptic and mean equinox
   3. Galactic: Galactic coordinates (lII, bII)
   4. Supergalactic: De Vaucouleurs Supergalactic coordinates (sgl, sgb)

Supported Reference systems (detailed information in :ref:`celestial-refsystems`):

   1. FK4: Mean place pre-IAU 1976 system.
   2. FK4_NO_E: The old FK4 (barycentric) equatorial system but without the
      "E-terms of aberration"
   3. FK5: Mean place post IAU 1976 system.
   4. ICRS: The International Celestial Reference System.
   5. J2000: This is an equatorial coordinate system based on the mean dynamical equator
      and equinox at epoch J2000.

Epochs (detailed information in :ref:`celestial-epochs`):

The equinox and epoch of observations are instants of time and are of type string.
These strings are parsed by a function of module :mod:`celestial` called :func:`celestial.epochs`.
The parser rules are described in the API documentation for that function.
Each string starts with a prefix. Supported prefixes are:

   #. B:   Besselian epoch
   #. J:   Julian epoch
   #. JD:  Julian date
   #. MJD: Modified Julian Day
   #. RJD: Reduced Julian Day
   #. F:   Old and new FITS format (old: `DD/MM/YY`  new: `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`)


**Example:**
Next example is a simple test program for epoch specifications.
The function :func:`celestial.epochs` returns a tuple with three elements:

   * the Besselian epoch
   * the Julian epoch
   * the Julian date.

::
      
   #!/usr/bin/env python
   from kapteyn import wcs
   
   ep = ['J2000', 'j2000', 'j 2000.5', 'B 2000', 'JD2450123.7',
         'mJD 24034', 'MJD50123.2', 'rJD50123.2', 'Rjd 23433',
         'F29/11/57', 'F2000-01-01', 'F2002-04-04T09:42:42.1']
   
   for epoch in ep:
      B, J, JD = wcs.epochs(epoch)
      print "%24s = B%f, J%f, JD %f" % (epoch, B, J, JD)


The output is::

    #                  J2000 = B2000.001278, J2000.000000, JD 2451545.000000
    #                  j2000 = B2000.001278, J2000.000000, JD 2451545.000000
    #               j 2000.5 = B2000.501288, J2000.500000, JD 2451727.625000
    #                 B 2000 = B2000.000000, J1999.998723, JD 2451544.533398
    #            JD2450123.7 = B1996.109887, J1996.108693, JD 2450123.700000
    #              mJD 24034 = B1924.680025, J1924.680356, JD 2424034.500000
    #             MJD50123.2 = B1996.109887, J1996.108693, JD 2450123.700000
    #             rJD50123.2 = B1996.108518, J1996.107324, JD 2450123.200000
    #              Rjd 23433 = B1923.033172, J1923.033539, JD 2423433.000000
    #              F29/11/57 = B1957.910029, J1957.909651, JD 2436171.500000
    #            F2000-01-01 = B1999.999909, J1999.998631, JD 2451544.500000
    # F2002-04-04T09:42:42.1 = B2002.257054, J2002.255728, JD 2452368.904654

The strings that start with prefix 'F' are strings read from FITS keywords
that represent the date of observation.

The sky definition
..................

Given an arbitrary celestial position and a sky system specification you can transform
to any of the other sky system specifications.
Module wcs recognizes the following built-in sky specifications:
::
      
   wcs.equatorial - wcs.ecliptic - wcs.galactic - wcs.supergalactic

Reference systems are:
::
      
   wcs.fk4 - wcs.fk4_no_e - wcs.fk5 - wcs.icrs - wcs.j2000


The syntax for an equatorial sky specification is either a tuple
(order of the elements is arbitrary):
::
      
   (sky system, equinox, reference system, epoch of observation)
   e.g.: obj.skyout = (wcs.equatorial, "J1983.5", wcs.fk4, "B1960_OBS")

or a string with minimal match::

   (equatorial, equinox, referencesystem, epoch of observation"
   e.g.: obj.skyout = "equa J1983.5 FK4 B1960_OBS"
   
Celestial transformations
.........................

In this section we check some basic celestial coordinate transformations.
Background information can be found in [Ref2]_ or in the background information for
module celestial.

Two parameters instantiate an object from class Transformation. The first is a definition
of the input celestial system and the second is a definition for the celestial output system.
Method :meth:`wcs.Transformation.transform` transforms coordinates associated with
the celestial input system to coordinates connected to the celestial output system.

The galactic pole has FK4 coordinates (192.25,27.4) in degrees.
If we want to verify this, we need to convert
this FK4 coordinate to the corresponding galactic coordinate, which should be (0,90)
within the limits of precision of the used numbers. The following script shows that this could be true::

   from kapteyn import wcs
   
   world_eq = (192.25, 27.4)   # FK4 coordinates of galactic pole
   tran = wcs.Transformation("EQ,fk4,B1950.0", "GAL")
   world_gal = tran.transform(world_eq)
   print world_gal
   
   # Output:
   # (120.8656324107187, 89.999949251695512)
   
   # Closure test:
   world_eq = tran.transform(world_gal, reverse=True)
   print world_eq
   
   # Output:
   # (192.25, 27.400000000000002)

We added a closure test (parameter `reverse=True`) to give you some feeling about the accuracy.
Closure tests usually show errors < 1e-12. We expected the pole at 
90 deg., but the difference is about 5e-05 deg. That is too much so 
there must be another reason for the difference. The reason is described 
in the background information of module :mod:`celestial`. 
The galactic pole is not a star
and the so called elliptic terms of aberration (only for FK4) are not apply to its
position.
So in fact the pole is given in FK4-NO-E coordinates. If we repeat the exercise with
the appropriate input celestial definition, we get::
   
   from kapteyn import wcs

   world_eq = (192.25, 27.4)   # FK4 coordinates of galactic pole
   tran = wcs.Transformation("EQUATORIAL, fk4_no_e, B1950.0", "galactic")
   world_gal = tran.transform(world_eq)
   print world_gal
   
   # Output:
   # (0.0, 90.0)

   world_eq = tran.transform(world_gal, reverse=True)
   print world_eq
   
   # Output:
   # (192.25, 27.400000000000002)


which gives the result as expected.
Note that we used a special feature of the Transformation class.
The two previous examples show that the transformation class is very useful to check
basic celestial transformations.

As another test of a standard celestial transformation, let's check 
the transformation between galactic and supergalactic coordinates.
The supergalactic pole (0,90) deg. has galactic(II) world coordinates (47.37,6.32) deg.
The conversion program becomes then::

   from kapteyn import wcs
   
   world_gal = (47.37, 6.32)   # Galactic l,b (II) of supergalactic pole
   tran = wcs.Transformation(wcs.galactic, wcs.supergalactic)
   world_sgal = tran.transform(world_gal)
   print world_sgal
   
   # Output:
   # (0.0, 90.0)
   
   world_eq = trans.transform(world_sgal, reverse=True)
   print world_gal
   
   # Output:
   # (47.369999999999997, 6.3200000000000003)


which agrees with the theory.

The sky system specifications allow for defaults.
So if one wants coordinates in the equatorial system with reference system
FK5 and equinox J2000 then the specification `wcs.fk5` will suffice.
Below we demonstrate how to transform a coordinate from the FK4 system to FK5.
In fact we want to demonstrate that FK4 is slowly rotating with respect
to the inertial FK5 system.
We do that by varying the assumed time of observation and convert the
position (R.A.,Dec) = (0,0).
This behaviour is explained in the background documentation of module :mod:`celestial`::

   #!/usr/bin/env python
   from kapteyn import wcs
   
   world_eq1 = (0,0)
   s_out = wcs.fk5
   epochs = range(1950,2010,10)
   for ep in epochs:
      s_in = "EQUATORIAL B1950 fk4 " + 'B'+str(ep)
      tran = wcs.Transformation(s_in, s_out)
      world_eq2 = tran.transform(world_eq1)
      print 'B'+str(ep), world_eq2
   
   # Output:
   # B1950 (0.64069100057541584, 0.27840943507737015)
   # B1960 (0.64069761256120361, 0.2783973383470032)
   # B1970 (0.64070422454697784, 0.27838524161663253)
   # B1980 (0.64071083653273853, 0.27837314488625808)
   # B1990 (0.64071744851848544, 0.27836104815588009)
   # B2000 (0.64072406050421915, 0.27834895142549831)
   

Usually FK4 catalog values are in equinox and epoch B1950.0,
so this program shows an exceptional case.

.. note::
     We are not restricted
     to the transformation of one coordinate. The input of positions follow the rules of
     coordinate representations as described for
     methods :meth:`wcs.Projection.toworld` and :meth:`wcs.Projection.topixel`.


Combining projections and celestial transformations
....................................................

In previous sections we showed examples how to use methods of an object of class
Projection to convert between pixel coordinates and world coordinates.
We added the option to change the celestial definition. If your data is a 
spatial map and its sky system is FK5, then we can convert pixel positions to
world coordinates in for example galactic coordinates by specifying 
a value for attribute :attr:`wcs.Projection.skyout`. In our case this would be for
a projection object called *proj*:
       
>>> proj.skyout = wcs.galactic
     

In the next example we test (like in one of the previous examples)
a conversion between an equatorial system
and the galactic system. The FK4-NO-E coordinates of the galactic pole
are the values (*CRVAL1*, *CRVAL2*) from the header.
First we calculate a couple of world coordinates in the native celestial definition.
Then we verify that that native system is indeed FK4-NO-E and the equinox is B1950.
That can be verified with:

>>> proj.skyout = (wcs.equatorial, wcs.fk4_no_e, 'B1950')

Finally we test the conversion to galactic coordinates with:

>>> proj.skyout = wcs.galactic

With the output sky set to galactic, we find the galactic pole in galactic
coordinates i.e. (90,0) deg. Finally we want to know what the values of the input
pixel coordinates are if the output sky system is supergalactic.
The galactic pole is (90, 6.32) deg. in supergalactic coordinates.
Within the limits of the precision of the used numbers we find the expected output with this script::

   from kapteyn import wcs
   header = { 'NAXIS'  : 2,
              'NAXIS1' : 5,
              'CTYPE1' : 'RA---TAN',
              'CRVAL1' : 192.25,
              'CRPIX1' : 5,
              'CUNIT1' : 'degree',
              'CDELT1' : -0.01,
              'NAXIS2' : 10,
              'CTYPE2' : 'DEC--TAN',
              'CRVAL2' : 27.4,
              'CRPIX2' : 5,
              'CUNIT2' : 'degree',
              'CDELT2' : +0.01,
              'RADESYS': 'FK4-NO-E',
              'EQUINOX': 1950.0
            }
   
   proj = wcs.Projection(header)
   
   pixel = [(4,5),(5,5),(6,5)]   # List with coordinate tuples
   world = proj.toworld(pixel)
   print world
   # [(192.26126360281495, 27.399999547653639), (192.25, 27.399999999999999), ...   

   proj.skyout = "Equatorial FK4-NO-E B1950"
   world = proj.toworld(pixel)
   print world
   # [(192.26126360281495, 27.399999547653639), (192.24999999999997, 27.400000000000002),...

   proj.skyout = "galactic"
   world = proj.toworld(pixel)
   print world
   # [(33.00000000001878,  89.990000000101531), (0.0, 90.0), ...
   
   proj.skyout = wcs.supergalactic
   world = proj.toworld(pixel)
   print world
   # [(90.002497049104363, 6.3296871263660073), (90.000000000000014, 6.319999999999995), ...


Note that the second tuple on each line of the output represents the world
coordinates at CRPIX.
Also important is the observation that the longitude for galactic coordinates
shows erratic behaviour. The reason is that close to a pole, the longitudes
are less well defined (and undefined on the pole) and the errors in longitudes
become important because we are calculating with numbers with a limited precision.

Attributes of a Projection object related to celestial systems
..............................................................

There are a number of attributes of an object of class :class:`wcs.Projection`,
related to celestial systems,
that can be used to inspect the parsed FITS header. The native system in the previous 
example could be derived from attribute :attr:`wcs.Projection.skysys`::

   from kapteyn import wcs
   header = { 'NAXIS'  : 2,
              'NAXIS1' : 5,
              'CTYPE1' : 'RA---TAN',
              'CRVAL1' : 192.25,
              'CRPIX1' : 5,
              'CUNIT1' : 'degree',
              'CDELT1' : -0.01,
              'NAXIS2' : 10,
              'CTYPE2' : 'DEC--TAN',
              'CRVAL2' : 27.4,
              'CRPIX2' : 5,
              'CUNIT2' : 'degree',
              'CDELT2' : +0.01,
              'RADESYS': 'FK4-NO-E',
              'EQUINOX': 1950.0,
              'MJD-OBS': 36010.2
            }
   
   proj = wcs.Projection(header)
   print "Attributes of 'proj':"
   print "skysys:    ", proj.skysys
   print "equinox:   ", proj.equinox
   print "epoch:     ", proj.epoch
   print "dateobs:   ", proj.dateobs
   print "mjdobs:    ", proj.mjdobs
   print "epobs:     ", proj.epobs
   
   # Attributes of 'proj':
   # skysys:     (0, 5, 'B1950.0')
   # equinox:    1950.0
   # epoch:      B1950.0
   # dateobs:    None
   # mjdobs:     36010.2
   # epobs:      MJD36010.2

Below a table with a short explanation of the attributes.
More information about epochs and equinoxes can be found
in the documentation of :mod:`celestial`.
 
========== ===============================================================
Attribute    Explanation
========== ===============================================================
skysys     A single value or tuple which defines the native system.
           Tuples can contain the sky system, the reference system,
           the equinox and the date of observation.
equinox    equinox is a floating point number. It is read from the 
           FITS header (keyword EQUINOX).
           The equinox is a moment in time
           used for the definition of an equatorial system.
epoch      This attribute is the epoch of the equinox. That is the 
           value of the equinox with prefix 'J' or 'B'. The context 
           (a.o. keyword RADYSYS) sets the prefix.
dateobs    Date of observation. Floating point number given by FITS
           keyword DATE-OBS
mjdobs     Date of observation. Floating point number given by FITS 
           keyword MJD-OBS
epobs      Date of observation as an epoch, i.e. copied from
           mjdobs or dateobs and prefixed by 'F' or 'MJD'
========== ===============================================================


Available functions from :mod:`celestial`
.........................................

Some of the functions defined in the module :mod:`celestial` are also available in the
namespace of :mod:`wcs`. One of these is :func:`celestial.epochs` for which we wrote an example in
the previous section. Others are :func:`celestial.lon2hms`, :func:`celestial.lon2dms`
and :func:`celestial.lat2hms` to format 
degrees into hours, minutes, seconds or degrees, minutes and seconds. 
Finally the function :func:`celestial.skymatrix` is also available to :mod:`wcs`; it calculates the rotation 
matrix to convert a coordinate from one sky system to another and it calculates
the E-terms (see background documentation  for celestial) if appropriate. Usually you will only use this
function to compare rotation matrices with matrices from the literature or to 
do some debugging. Some examples on the Python command line:

**Formatting spatial coordinates:**

>>> wcs.lon2hms(45.0)
'03h 00m  0.0s'
>>> wcs.lon2hms(23.453839, 4)
'01h 33m 48.9214s'
>>> wcs.lon2dms(245.0, 4)
Out[10]: ' 245d  0m  0.0000s'
>>> wcs.lat2dms(45.0)
'+45d 00m  0.0s'
>>> help(wcs.lon2hms)

**Calculate a rotation matrix:**

>>> wcs.skymatrix(wcs.galactic, wcs.supergalactic)
(matrix([[ -7.35742575e-01,   6.77261296e-01,  -6.08581960e-17],
        [ -7.45537784e-02,  -8.09914713e-02,   9.93922590e-01],
        [  6.73145302e-01,   7.31271166e-01,   1.10081262e-01]]), None, None)


Spectral transformations
------------------------

Introduction
............

The most important documentation about conversions of spectral coordinates in WCSlib is found
paper "Representations of spectral coordinates in FITS" (paper III, [Ref3]_ )
In the next sections we show how :mod:`wcs`/WCSLIB can deal with spectral conversions
with the focus on conversions between
frequencies and velocities. We discuss conversion examples shown in the paper
in detail and try to illustrate how :mod:`wcs` deals with FITS data from 
(legacy) AIPS and GIPSY sources. In many of those files the reference frequencies
and reference velocities are not given in the same reference system
(e.g. topocentric v.s. barycentric). It is estimated that there are many of
these FITS files and that their headers generate wrong results when they enter
the constructor for the :class:`wcs.Projection` class unmodified. 
For FITS files generated with legacy software some extra parsing of the FITS header is applied.
This procedure is described in more detail in the background information related to
spectral coordinates.



Transformations between frequencies and velocities
..................................................

We built applications that use WCSLIB to convert grid positions, in an image or a spectrum,
to world coordinates. For spectral axes with frequency as the primary type
(e.g. in the FITS header we read CTYPE3='FREQ'), it is possible to convert
between pixel coordinates and frequencies, but also, if the header provides the correct information, 
between pixel coordinates and velocities.
WCSlib expects that in a FITS header the given frequencies are bound to the same
standard of rest (i.e. reference system) as the given reference velocity.
In practice however there are many FITS files that list the frequencies in 
the topocentric system and a reference velocity in an inertial system
(barycentric, lsrk). In those FITS files the inertial systems are usually
abbreviated with 'HEL' or 'LSR' (Heliocentric, Local Standard of Rest)
and the velocities are usually not the true velocities but are either
the so called *radio* or *optical* velocities (of which we give the definitions in the background
information about spectral coordinates).


Basic spectral line header example
..................................

In "Representations of spectral coordinates in FITS" ([Ref3]_ ) section 10.1 
deals with an example of a VLA spectral line cube which is regularly sampled
in frequency (CTYPE3='FREQ'). The section describes how one can define
alternative FITS headers to deal with different velocity definitions. 
We want to examine this exercise in more detail than provided in the
article to illustrate how a FITS header can be modified.
In the background information you find a more elaborate discussion. Here we 
summarize some results.

The topocentric spectral properties in the FITS header from the paper are found to be::

   CTYPE3= 'FREQ'
   CRVAL3=  1.37835117405e9
   CDELT3=  9.765625e4
   CRPIX3=  32
   CUNIT3= 'Hz'
   RESTFRQ= 1.420405752e+9
   SPECSYS='TOPOCENT'

Usually such descriptions are part of a header that describes a three dimensional data structure
where the first two axes represent a spatial map as function of the 
third axis which is a spectral axis.
This example tells us that the spatial data corresponding with channel 32 was observed 
at a topocentric frequency (SPECSYS='TOPOCENT') of 1.37835117405 GHz.
The step size in frequency is 97.65625 kHz.
A rest frequency (1.420405752e+9 Hz) is needed to convert frequencies to velocities.
Description of standard FITS keywords can be found in [FITS]_ 

The topocentric frequency (for the receiver) was derived from a barycentric optical
velocity of 9120 km/s that was requested by an observer.

We prepared a minimal header to simulate this FITS header. 
and calculate world coordinates for the spectral axis 
The numbers are frequencies. The units are *Hz* and the central frequency is *CRVAL3*.
The step in frequency is *CDELT3*. Our minimal header (here presented as a Python dictionary)
shows only one axis so our header items got axis number 1 (e.g. *CRVAL1*, *CDELT1*, etc.)::


   from kapteyn import wcs
   header = { 'NAXIS'  :  1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' :  1.37835117405e9,
              'CRPIX1' :  32,
              'CUNIT1' : 'Hz',
              'CDELT1' :  9.765625e4
            }
   proj = wcs.Projection(header)
   pixels = range(30,35)
   Fwcs = proj.toworld1d(pixels)
   for p,f in zip(pixels, Fwcs):
      print p, f

   # Output:
   30 1378155861.55
   31 1378253517.8
   32 1378351174.05
   33 1378448830.3
   34 1378546486.55

The output shows frequency as function of pixel coordinate. Pixel coordinate 32 (=*CRPIX1*) shows the value
of *CRVAL1*. Now we have a method to find at which frequency a spatial map in the data cube was 
observed.


WCSlib velocities from frequency data
.....................................

Usually similar FITS headers provide information about a velocity. 
Velocities is what we need for the analysis of the kinematics and dynamics
of the observed objects. But there are several definitions for velocities
(*radio*, *optical*, *apparent radial*). 

For the radio interferometer, like the WSRT, an observer requesting for an observation, needs to specify:
   
   * A rest frequency
   * A velocity or Doppler shift
   * A frame definition (bary or lsrk)
   * A conversion type (z, radio, optical)
   * A time of observation. This time is needed (together with the location of 
     the observatory) to calculate the topocentric frequencies needed 
     for the receivers
 

*The observer requests that an observation must correspond to a velocity or Doppler shift
(see list below) and a reference system. Only then topocentric frequencies for the
receivers can be calculated.*
 
To convert to another spectral type the constructor from class :class:`wcs.Projection` needs to know
which spectral type we want to convert to. The translation is set then with :meth:`wcs.Projection.spectra`.
which stands for *spectral translation*.

The parameter that we need to set the translation is *ctype*. It's syntax follows the
FITS convention, see note below.

.. note:: The first four
          characters of a spectral CTYPE specify the new coordinate type, the fifth
          character is ‘-’ and the next three characters specify a predefined
          algorithm for computing the world coordinates from intermediate
          physical coordinates ([Ref3]_ ).

The spectral types that are supported are (from [Ref3]_):
        
=========  ============================ ======= ======= ================
Type       Name                         Symbol   Units  Associated with
=========  ============================ ======= ======= ================
FREQ       Frequency                    ν       Hz      ν
ENER       Energy                       E       J       ν
WAVN       Wavenumber                   κ       1/m     ν
VRAD       Radio velocity               V       m/s     ν
WAVE       Vacuum wavelength            λ       m       λ
VOPT       Optical velocity             Z       m/s     λ
ZOPT       Redshift                     z       \-      λ
AWAV       Air wavelength               λa      m       λa
VELO       Apparent radial velocity     v       m/s     v
BETA       Beta factor (v/c)            β       \-      v
=========  ============================ ======= ======= ================

The non-linear algorithm codes are (from [Ref3]_):
        
==== ========================= ===========================     
Code sampled in                 Expressed as
==== ========================= ===========================
F2W  Frequency                  Wavelength
F2V  Frequency                  Apparent radial velocity
F2A  Frequency                  Air wavelength
W2F  Wavelength                 Frequency
W2V  Wavelength                 Apparent radial velocity
W2A  Wavelength                 Air wavelength
V2F  Apparent radial velocity   Frequency
V2W  Apparent radial velocity   Wavelength
V2A  Apparent radial velocity   Air wavelength
A2F  Air wavelength             Frequency
A2W  Air wavelength             Wavelength
A2V  Air wavelength             Apparent radial velocity
==== ========================= ===========================


If we want to convert pixel coordinates to optical velocities for our example header, 
then module :mod:`wcs` needs to create a new projection object with *ctype* = VOPT-F2W
because VOPT represents an optical velocity and F2W sets the non linear algorithm
which converts from the domain where the step size is constant (frequency) to
a velocity associated with wavelength (see table above).
The following script shows how to use the method 
:meth:`wcs.Projection.spectra` to create this new object and how to 
convert the pixel coordinates::
   
   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : 1.37835117405e9,
              'CRPIX1' : 32,
              'CUNIT1' : 'Hz',
              'CDELT1' : 9.765625e4,
              'RESTFRQ': 1.420405752e+9
            }
   proj = wcs.Projection(header)
   spec = proj.spectra('VOPT-F2W')
   pixels = range(30,35)
   Vwcs = spec.toworld1d(pixels)
   print "Pixel, velocity (%s)" % spec.units
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0
   
   # Output:
   # Pixel, velocity (m/s)
   # 30 9190.68652655
   # 31 9168.7935041
   # 32 9146.90358389
   # 33 9125.01676527
   # 34 9103.13304757

Some comments about this example:
      
    * It shows how to add the
      spectral translation to the projection object. For a conversion from frequency to 
      optical velocity one can define a new object `spec = proc.spectra('VOPT-F2W')` or change
      the current object with: `proj = wcs.Projection(header).spectra('VOPT-F2W')`.
    * The output is a list with pixel coordinates and *topocentric* velocities. This explains
      why we don't see the requested velocity (9120 km/s) at CRPIX because that velocity was barycentric.
    * When we enter an invalid algorithm code for the velocity, the script will raise an exception.


**Why do we need a rest frequency?**

To get a velocity, the rest frequency needs to be added (RESTFRQ=) to our minimal header.
What you get then is a list of velocities according to:

.. math::
   :label: eq1
   
        Z = c ( \frac{\lambda - \lambda_0}{\lambda_0}) = c\ (\frac{\nu_0 - \nu}{\nu})

We adopted variable *Z* for velocities following the optical definition.
The frequency as (linear) function of pixel coordinate is:

.. math::
   :label: eq2
   
      \nu = \nu_{ref} + (N - N_{\nu_{ref}}) \delta \nu
      

where:

   * :math:`\nu_{ref}` is the *reference frequency* (CRVAL1)
   * :math:`N` is the pixel coordinate (FITS definition) we are interested in,
   * :math:`N_{\nu_{ref}}` is the frequency reference pixel (CRPIX1)
   * :math:`\delta \nu` is the frequency increment (CDELT1)


Let's check this with a small script::

   from kapteyn import wcs
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : 1.37835117405e9,
              'CRPIX1' : 32,
              'CUNIT1' : 'Hz',
              'CDELT1' : 9.765625e4,
              'RESTFRQ': 1.420405752e+9
            }
   proj = wcs.Projection(header)
   spec = proj.spectra(ctype='VOPT-F2W')
   pixels = range(30,35)
   Vopt = spec.toworld1d(pixels)
   
   print "Pixel coordinate and velocity (%s) with wcs module:" % spec.units
   for p,Z in zip(pixels, Vopt):
      print p, Z/1000.0
   
   print "\nPixel coordinate and velocity (%s) with documented formulas:" % spec.units
   for p in pixels:
      nu = header['CRVAL1'] + (p-header['CRPIX1'])*header['CDELT1']
      Z = wcs.c*(header['RESTFRQ']-nu)/nu     # wcs.c is speed of light in m/s
      print p, Z/1000.0
   
   # Pixel coordinate and velocity (m/s) with wcs module:
   # 30 9190.68652655
   # 31 9168.7935041
   # 32 9146.90358389
   # 33 9125.01676527
   # 34 9103.13304757
   
   # Pixel coordinate and velocity (m/s) with documented formulas:
   # 30 9190.68652655
   # 31 9168.7935041
   # 32 9146.90358389
   # 33 9125.01676527
   # 34 9103.13304757

More checks are documented in the background information for spectral coordinates. This one should
give you some idea how WCSLIB transforms spectral coordinates. But we still didn't address the question about
the reference systems. 
In our code example, this velocity *Z* is topocentric (defined in the reference system of the observatory)
and is not suitable for comparisons because the Earth is moving around its axis and around the Sun.
Other reference systems are the barycenter of the Solar system and the Local Standard of Rest.
During observations one knows the location of the source, the time of observation and the location
of the observatory on Earth. Software then can calculate the (true) velocity of the Earth with
respect to a selected inertial reference system and we can transform from topocentric
velocities to velocities in another system. Usually these correction velocities (called *topocentric correction*)
are not recorded in the FITS file of the data set.

In the background information about spectral coordinates we give a recipe how one can
change the value of the reference frequency in CRVAL1 to a barycentric value.
The result is CRVAL1=1.37847121643e+9
If you substitute this value for CRVAL1 in the previous script, the output is::

    Pixel coordinate and velocity (m/s) with wcs module:
    30 9163.77531673
    31 9141.88610757
    32 9119.99999984
    33 9098.11699288
    34 9076.23708605

At pixel coordinate 32 (CRPIX1) the velocity is 9120 km/s as we required. So :mod:`wcs` always
returns velocities in the same system as the system of reference frequency.

.. warning::

      Reference frequencies given in FITS keyword CRVALn refer to a reference system.
      This system should be given with FITS keyword SPECSYS= (e.g. SPECSYS='TOPOCENT').
      Module :mod:`wcs`
      converts between frequencies and velocities in the *same* reference system.
      You should inspect your FITS header to find what this system is.

Spectral CTYPE's with special extensions
........................................

There are many (old) FITS headers which describe a system where the reference frequency is
topocentric and the required reference velocity is given for another reference system.
These velocities are given with keywords like VELR= or DRVALn= and the reference system
for the velocities is given as an extension in CTYPEn (e.g.: CTYPE3='FREQ-OHEL').
Image processing systems like AIPS and GIPSY
have their own tools to deal with this. If :mod:`wcs` recognizes a legacy header, it tries
to convert the reference frequency to the system of the required velocity::
   
   from kapteyn import wcs

   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ-OHEL',
              'CRVAL1' : 1.415418199417E+09,
              'CRPIX1' : 32,
              'CUNIT1' : 'HZ',
              'CDELT1' : -7.812500000000E+04,
              'VELR'   : 1.050000000000E+06,
              'RESTFRQ': 0.14204057520000E+10
            }
   
   proj = wcs.Projection(header)
   ctype = 'FREQ-???'
   if ctype != None:
      spec = proj.spectra(ctype)
      print "\nSelected spectral translation with algorithm code:", spec.ctype[0]
   else:
      spec = proj
   
   crpix = header['CRPIX1']
   print "CRVAL from header=%f, CRVAL modified=%f" % (header['CRVAL1'], spec.crval[0])
   print "CDELT from header=%f, CDELT modified=%f" % (header['CDELT1'], spec.cdelt[0])
   for i in range(-2,+3):
      px = crpix + i
      world = spec.toworld1d(px)
      print "%d %f" % (px, world)

   # Output:
   # Selected spectral translation with algorithm code: FREQ
   # CRVAL from header=1415418199.417000, CRVAL modified=1415448253.482287
   # CDELT from header=-78125.000000, CDELT modified=-78123.341180
   # 30 1415604500.164647
   # 31 1415526376.823467
   # 32 1415448253.482287
   # 33 1415370130.141107
   # 34 1415292006.799927

   
As spectral translation we selected 'FREQ'. 
If you inspect the output list with frequencies then you will see that the list 
doesn't show the topocentric frequencies (with CRVAL1 at CRPIX1) but 
frequencies in the reference system of the given (helocentric) velocity.
The attributes `spec.crval[0]` and `spec.cdelt[0]` show new values unequal to the
header values. 

If you want a list with topocentric frequencies then just omit to apply 
the :meth:`wcs.Projection.spectra` method (i.e. use `ctype = None` in example).
The output is what we expect::
   
   # Output:
   # CRVAL from header=1415418199.417000, CRVAL modified=1415418199.417000
   # CDELT from header=-78125.000000, CDELT modified=-78125.000000
   # 30 1415574449.417000
   # 31 1415496324.417000
   # 32 1415418199.417000
   # 33 1415340074.417000
   # 34 1415261949.417000


A note about algorithm codes
.............................

It is not always easy to figure out what the algorithm code should be if you
want to convert to another spectral type. Therefore WCSLIB allows wildcard characters
for the last or the last three characters in CTYPEn. In our example valid entries are:
    
    * `spec = proj.spectra(ctype='VOPT-F2W')`
    * `spec = proj.spectra(ctype='VOPT-F2?')`
    * `spec = proj.spectra(ctype='VOPT-???')`


The missing algorithm code is returned in :attr:`wcs.Projection.ctype` as in::

   >>> spec = proj.spectra(ctype='VOPT-???')
   >>> print "Spectral translation with algorithm code:", spec.ctype[0]
       Spectral translation with algorithm code: VOPT-F2W


Module :mod:`wcs` uses this feature to build a list with all spectral translations that
are allowed for a given Projection object. For each type in the table with spectral types,
the wildcards are used to find the algorithm code (assuming that for the given Projection objects
and the spectral type only one algorithm is possible). A tuple is created with the
allowed spectral translation as first element and its associated unit as second element) and the
tuple is added to the list :attr:`wcs.Projection.altspec`.

The attribute is useful if you want to write code that prompts a user to enter a spectral
translation from a list of allowed translations. It can be used as follows::

   from kapteyn import wcs
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VOPT',
              'CRVAL1' : 9120,
              'CRPIX1' : 32,
              'CUNIT1' : 'km/s',
              'CDELT1' : -21.882651442,
              'RESTFRQ': 1.420405752e+9
            }
   
   proj = wcs.Projection(header)
   print "Allowed spectral translations:"
   for as in proj.altspec:
      print as
   spec = proj.spectra(ctype='FREQ-???')
   print "\nSelected spectral translation with algorithm code:", spec.ctype[0]
   
   # Output:
   # Allowed spectral translations:
   # ('FREQ-W2F', 'Hz')
   # ('ENER-W2F', 'J')
   # ('VOPT', 'm/s')
   # ('VRAD-W2F', 'm/s')
   # ('VELO-W2V', 'm/s')
   # ('WAVE', 'm')
   # ('ZOPT', '')
   # ('BETA-W2V', '')
   
   # Selected spectral translation with algorithm code: FREQ-W2F

   
From velocities to frequencies
...............................

In the background information about spectral coordinates we calculated that for a barycentric
system the step size in barycentric velocity is -21.882651442 km/s.
Then we are able to setup a header with velocities and use a spectral translation that converts to frequencies,
as in the next example::

   from kapteyn import wcs
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VOPT-F2W',
              'CRVAL1' : 9120,
              'CRPIX1' : 32,
              'CUNIT1' : 'km/s',
              'CDELT1' : -21.882651442,
              'RESTFRQ': 1.420405752e+9
            }

   proj = wcs.Projection(header)
   spec = proj.spectra(ctype='FREQ-???')
   print "Spectral translation with algorithm code:", spec.ctype[0]
   pixels = range(30,35)
   Freq = spec.toworld1d(pixels)

   print "Pixel coordinate and frequency (%s)" % spec.units
   for p,f in zip(pixels, Freq):
      print p, f

   # Output:
   # Pixel coordinate and frequency (Hz):
   # 30 1378275920.94
   # 31 1378373568.68
   # 32 1378471216.43
   # 33 1378568864.18
   # 34 1378666511.92

The reference frequency is at pixel coordinate 32 and its value (1378471216.43) is exactly the
barycentric reference frequency that we used before. What happens if we left out the algorithm code
in the header? The output differs (except for the reference frequency at pixel 32). That is because
it is assumed that the increments in wavelength are constant and not those in frequency.
This is confirmed by the returned algorithm code which is *FREQ-W2F* if CTYPE1='VOPT'


Processing real FITS data
.........................

With the knowledge we have at this moment, it is easy to make a small utility
that looks for a spectral axis in a FITS file and if it can find one, it convert 5 pixel coordinates in
the neighbourhood of CRPIX to world coordinates for all allowed spectral translations::

   from kapteyn import wcs
   import pyfits
   
   f = raw_input("Enter name of FITS file: ")
   hdulist = pyfits.open(f)
   header = hdulist[0].header
   proj = wcs.Projection(header)
   ax = proj.specaxnum
   if ax == None:
      print "No spectral axis available"
   else:
      print "Spectral type from header:", proj.ctype[ax-1]
      crpix = header['CRPIX'+str(ax)]
      for alt in proj.altspec:
         line = proj.sub((ax,)).spectra(alt[0])
         print "Pixel, world for translation %s" % alt[0]
         for i in range(-2,+3):
            px = crpix + i
            world = line.toworld1d(px)   #  to world coordinates
            print "%d %.10g (%s)" % (px, world, alt[1])

The projection object reads its header data from the first hdu of the FITS file
(`hdulist[0].hdr`) and is set to only convert the spectral axis of the data set:
`proj.(.sub((ax,)))`.
Remember that the argument is a Python tuple but we have only one axis so the tuple has an extra comma.
Header items can be read from the header directly (e.g. `header['CRPIX3']`). That's how we find
the value of CRPIX for the spectral axis. The allowed spectral translations are
read from attribute :attr:`wcs.Projection.altspec`. 

We ran the example for a fits file called *mclean.fits* which is a HI data cube and the
third axis is the spectral axis::

   Enter name of FITS file: mclean.fits
   Spectral type from header: FREQ
   Pixel, world for translation FREQ
   28 1415604500 (Hz)
   29 1415526377 (Hz)
   30 1415448253 (Hz)
   31 1415370130 (Hz)
   32 1415292007 (Hz)
   Pixel, world for translation ENER
   28 9.379902296e-25 (J)
   29 9.379384645e-25 (J)
   30 9.378866994e-25 (J)
   31 9.378349343e-25 (J)
   32 9.377831692e-25 (J)
   Pixel, world for translation VOPT-F2W
   28 1016794.655 (m/s)
   29 1033396.411 (m/s)
   30 1050000 (m/s)
   31 1066605.422 (m/s)
   32 1083212.677 (m/s)
   etc. etc. 

References
----------
.. [Ref1] `Representations of world coordinates in FITS`
          `<http://www.atnf.csiro.au/people/mcalabre/WCS/wcs.pdf>`_  Greisen E.W. and Calabretta M.R.
          
.. [Ref2] `Representations of celestial coordinates in FITS`
          `<http://www.atnf.csiro.au/people/mcalabre/WCS/ccs.pdf>`_  Calabretta M.R. and Greisen E.W.

.. [Ref3] `Representations of spectral coordinates in FITS`
          `<http://www.atnf.csiro.au/people/mcalabre/WCS/scs.pdf>`_  E. W. Greisen, M. R. Calabretta, F. G. Valdes, and S. L. Allen

.. [FITS] `Definition of the Flexible Image Transport System (FITS), FITS Standard Version 3.0`
          `<http://fits.gsfc.nasa.gov/fits_standard.html>`_  FITS Working Group , Commission 5: Documentation and Astronomical Data, International Astronomical Union
           