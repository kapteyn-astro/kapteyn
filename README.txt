The Kapteyn Package is a collection of Python modules and applications
developed by members of the computer group (*) of the Kapteyn
Astronomical Institute, University of Groningen, The Netherlands.  The
purpose of the package is to provide tools for the development of
astronomical applications with Python. 

The package is suitable for both inexperienced and experienced users and
developers and documentation is provided for both groups.  The
documentation also provides in-depth chapters about celestial
transformations and spectral translations. 

Some of the package's features:

   * The handling of spatial and spectral coordinates, WCS projections
     and transformations between different sky systems.  Spectral
     translations (e.g., between frequencies and velocities) are supported
     and also mixed coordinates.  (Modules wcs and celestial. Module wcs
     uses Mark Calabretta's WCSLIB which is distributed with the package.)

   * Versatile tools for writing small and dedicated applications for
     the inspection of FITS headers, the extraction and display of (FITS)
     data, interactive inspection of this data (color editing) and for the
     creation of plots with world coordinate information.  (Module maputils)
     As one example, a gallery of all-sky plots is provided. 

   * A class for the efficient reading, writing and manipulating simple
     table-like structures in text files.  (Module tabarray)

   * Utilities for use with matplotlib such as obtaining coordinate
     information from plots, interactively modifiable colormaps and timer
     events (module mplutil); tools for parsing and interpreting coordinate
     information entered by the user (module positions).

   * A function to search for gaussian components in a profile
     (module profiles) and a class for non-linear least squares fitting
     (module kmpfit). 

---
*) Currently Hans Terlouw and Martin Vogelaar
