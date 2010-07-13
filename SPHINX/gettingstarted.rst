How to start
========================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------

This chapter is intended to be a guide on how to use the modules
from the Kapteyn Package for your own astronomical software.
The Kapteyn Package provides building blocks for software that has a focus
on the use of world coordinates and/or plotting image data.

To get an overview of what is possible, have a look at :doc:`maputilstutorial` which
contains many examples of world coordinate annotations and
plots of astronomical data. It can be a good starting point to use the source code
in the example scripts to process your own data by making only small changes
to the code.

If you are only interested in coordinate transformations, then the
:doc:`wcstutorial` is a good starting point.
 

Which module and documents to use?
------------------------------------

.. tabularcolumns:: |p{100mm}|p{50mm}|

+-------------------------------------------+---------------------------+
|You want:                                  |You need:                  |
+===========================================+===========================+
|For a set of world coordinates, I want     |:mod:`wcs`,                |
|to transform these to another projection   |:doc:`wcstutorial`         |
|system. I have a FITS header.              |                           |
|                                           |                           |
+-------------------------------------------+---------------------------+
|I want to transform world coordinates      |:mod:`wcs`,                |
|between sky- and reference systems         |:doc:`wcstutorial`         |
+-------------------------------------------+---------------------------+
|I want a parser to convert a string with   |:mod:`positions`           |
|position information to pixel- and/or      |                           |
|world coordinates.                         |                           |
+-------------------------------------------+---------------------------+
|I want to transform image data in a FITS   |:mod:`maputils`,           |
|file from one projection system to another |:doc:`maputilstutorial`    |
+-------------------------------------------+---------------------------+
|I want to build a utility that converts a  |:mod:`maputils`,           |
|header with a PC or CD matrix to a         |:doc:`maputilstutorial`    |
|'classic' header with CRPIX, CRVAL, CDELT  |                           |
|and CROTA                                  |                           |
+-------------------------------------------+---------------------------+
|I want to create a utility that can        |:mod:`maputils`,           |
|display a mosaic of image data             |:doc:`maputilstutorial`    |
+-------------------------------------------+---------------------------+
|I want to plot an all sky map with         |:mod:`maputils`,           |
|graticules                                 |:doc:`maputilstutorial`    |
+-------------------------------------------+---------------------------+
|I want to calculate flux in a set of       |:mod:`maputils`,           |
|images                                     |:mod:`shapes`,             |
|                                           |:doc:`maputilstutorial`    |
+-------------------------------------------+---------------------------+
|I want to create a simple FITS file viewer |:mod:`maputils`,           |
|with user interaction for the colors etc.  |:doc:`maputilstutorial`    |
+-------------------------------------------+---------------------------+
|I want to read a large data file very fast |:mod:`tabarray`,           |
|                                           |:doc:`tabarraytutorial`    |
+-------------------------------------------+---------------------------+
|Given a year, month and day number, I want |:mod:`celestial`,          |
|the corresponding Julian date              |:doc:`wcstutorial`         |
+-------------------------------------------+---------------------------+
|I want to know the obliquity of the        |:mod:`celestial`,          |
|ecliptic at a Julian date?                 |:doc:`wcstutorial`,        |
|                                           |:doc:`celestialbackground` |
+-------------------------------------------+---------------------------+
|I want to convert my spectral axis from    |:mod:`wcs`,                |
|frequency to relativistic velocity         |:doc:`maputilstutorial`,   |
|                                           |:doc:`spectralbackground`  |
+-------------------------------------------+---------------------------+


Functionality of the modules in the Kapteyn Package
-----------------------------------------------------

Wcs
.....

   * Given a FITS header or a Python dictionary with header information about a World
     Coordinate System (WCS), transform between pixel- and world coordinates.
   * Different coordinate representations are possible (tuple of scalars, NumPy array etc.)
   * Transformations between sky and reference systems.
   * Epoch transformations
   * Support for 'alternate' headers (a header can have more than one description of a WCS)
   * Support for mixed coordinate transformations (i.e. pixel- and world coordinates at
     input are mixed).
   * Spectral coordinate translations, e.g. convert a frequency axis to an optical
     velocity axis.


Celestial
.........

   * Coordinate transformations between sky and reference systems. Also available via
     module :mod:`wcs`
   * Epoch transformations. Also available via
     module :mod:`wcs`
   * Many utility functions e.g. to convert epochs, to parse strings
     that define sky- and reference systems, calculate Julian dates,
     precession angles etc.


Wcsgrat
........

   * Most of the functionality in this module is provided via user friendly methods in
     module :mod:`maputils`.
   * Calculate grid lines showing constant latitude as function of varying longitude
     or vice versa.
   * Methods to set the properties of various plot elements like tick marks, tick labels
     and axis labels.
   * Methods to calculate positions of labels inside a plot (e.g. for all sky plots).


Maputils
.........

        * Easy to combine with Matplotlib
        * Convenience methods for methods of modules :mod:`wcs`, :mod:`celestial`, :mod:`wcsgrat`
        * Overlays of different graticules (each representing a different sky system),
        * Plots of data slices from a data set with more than two axes
          (e.g. a FITS file with channel maps from a radio interferometer observation)
        * Plots with a spectral axis with a ‘spectral translation’ (e.g. Frequency to Radio velocity)
        * Rulers with distances in world coordinates, corrected for projections.
        * Plots for data that cover the entire sky (allsky plot)
        * Mosaics of multiple images (e.g. HI channel maps)
        * A simple movie loop program to view ‘channel’ maps.
        * Interactive colormap selection and modification.


Positions
..........

   * Convert strings to positions in pixel- and world coordinates


Rulers
........

   * Plot a straight line with markers at constant distance in world coordinates.
     Its functionality is available in module :mod:`maputils`


Shapes
........

   * Advanced plotting with user interaction. A user defines a shape (polygon,
     ellipse, circle, rectangle, spline) in an image and the shape propagates
     (in world coordinates) to other images. A shape object keeps track of
     its area (in pixels) and the sum of the pixels within the shape. From these
     a flux can be calculated.


Tabarray
.........

   * Fast I/O for data in ASCII files on disk.


Mplutil
...........

   * Various advanced utilities for event handling in Matplotlib. Most of its
     functionality is used in module :mod:`maputils`.
