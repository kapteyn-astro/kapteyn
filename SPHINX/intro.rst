Introduction
============

The Kapteyn Package is a collection of Python modules and applications
developed by the computer group of the
`Kapteyn Astronomical Institute <http://www.astro.rug.nl>`_,
University of Groningen, The Netherlands. 
The purpose of the package is to provide tools for the development of
astronomical applications with Python.

The package is suitable for both inexperienced and experienced users and
developers and documentation is provided for both groups.  The
documentation also provides in-depth chapters about celestial
transformations and spectral translations. 

Some of the package's features:

* The handling of spatial and spectral coordinates, WCS projections
  and transformations between different sky systems.  Spectral
  translations (e.g., between frequencies and velocities) are supported
  and also mixed coordinates.  (Modules wcs and celestial)

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


Overview
--------

The following modules are included:

- :mod:`wcs`, a binary module which handles spatial and spectral
  coordinates and provides
  WCS projections and transformations between different sky systems.
  Spectral translations (e.g., between frequencies and velocities) are
  supported and also mixed coordinates.

- :mod:`celestial`, containing NumPy-based functions for creating
  matrices for transformation between different celestial systems.
  Also a number of other utility functions are included.

- :mod:`wcsgrat`, for calculating parameters for WCS graticules.
  It does not require a plot package.

- :mod:`maputils`. Provides methods for reading FITS files.
  It can extract 2-dim image data from data sets with three or more axes.
  A class is added which prepares FITS data to plot itself as an image
  with Matplotlib.

- :mod:`positions`, enabling a user/programmer to specify positions in
  either pixel- or world coordinates.

- :mod:`rulers`, defining a class for drawing rulers.

- :mod:`shapes`, defining a class for interactively drawing shapes that
  define an area in an image. For each area a number of properties of the data
  is calculated. This module can duplicate a shape in different
  images using transformations to world coordinates.
  This enables one for instance to compare flux in two images with
  different WCS systems.

- :mod:`mplutil`, utilities for use with matplotlib.
  Classes AxesCallback, providing a more powerful
  mechanism for handling events from LocationEvent and derived classes
  than matplotlib provides itself; TimeCallback for handling timer events
  and VariableColormap which implements a matplotlib Colormap subclass
  with special methods that allow the colormap to be modified.

- :mod:`tabarray`, providing a class for the efficient reading, writing and
  manipulating simple table-like structures in text files. 

.. ascarray left out
  :mod:`ascarray`, a binary module containing the base function for
  module :mod:`tabarray`.

.. index:: prerequisites, Python, NumPy, PyFITS, matplotlib

Prerequisites
-------------

To install the Kapteyn Package, at least Python_ 2.4
and NumPy_ (both with header files) are required.
For using it, the availability of PyFITS_ and matplotlib_ is recommended.
Windows users may also need to install Readline_ or an equivalent package.

Mark Calabretta's WCSLIB_ does not need to be installed separately anymore.
Its code is now included in the Kapteyn Package under the
GNU Lesser General Public License.

.. _Python: http://www.python.org/
.. _NumPy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _matplotlib: http://matplotlib.sourceforge.net/
.. _Readline: http://newcenturycomputers.net/projects/readline.html
.. _WCSLIB: http://www.atnf.csiro.au/people/mcalabre/WCS/

.. index:: download

Download
--------

The Kapteyn Package and the example scripts can be downloaded via links on
the package's homepage: http://www.astro.rug.nl/software/kapteyn/

.. index:: install

Installing
----------

First unpack the downloaded .tar.gz or .zip file and go to the
resulting directory. Then one of the following options can be chosen:

#. Install into your Python system (you usually need root permission
   for this)::

      python setup.py install

#. If you prefer not to modify your Python installation, you can 
   create a directory under which to install the module
   e.g., *mydir*. Then install as follows::

      python setup.py install --install-lib mydir

   To use the package you then need to include *mydir* in your PYTHONPATH.

   .. index:: GIPSY

#. If you want to use this package only for GIPSY, you can
   install it as follows::

      python setup.py install --install-lib $gip_exe

   The GIPSY installation procedure normally does this automatically,
   so usually this will not be necessary.

Windows installer
.................

An experimental installer for Microsoft Windows (together with other
packages that the Kapteyn Package depends on) is also available.
Currently only for Python 2.6 on 32-bit systems.
http://www.astro.rug.nl/software/kapteyn_windows/

Scisoft problem
...............

If you have Scisoft installed on your computer, it may interfere with
the installation of the Kapteyn Package.  To install it properly,
disable the setup of Scisoft in your startup file (e.g. ~/.cshrc,
.profile) by commenting it out. 


Contact
-------


The authors can be reached at:

   Kapteyn Astronomical Institute

   Postbus 800

   NL-9700 AV Groningen

   The Netherlands

   Telephone: +31 50 363 4073

   E-mail:    gipsy@astro.rug.nl

------------------

.. target-notes::


.. experiments:

   (remove leading blanks to activate)

   Epilogue
   --------
   
   Suppose a droplet of liquid is placed in an external medium that exerts
   a pressure :math:`P` on the droplet.
   Then the work done by the droplet on expansion is empirically given by
   
   .. math::
   
      dW=P\thinspace dV-\gamma\thinspace da
   
   where :math:`da` is the increase in the surface area of the droplet and
   :math:`\gamma` the coefficient of surface tension.
   The first law now takes the form
   
   .. math::
      :label: firstlaw
   
      dU=dQ-P\thinspace dV+\gamma\thinspace da
   
   Integrating this, we obtain for the internal energy of a droplet of
   radius :math:`r` the expression
   
   .. math::
   
      U={{4}\over {3}}\pi r^3u_\infty +4\pi \gamma r^2
   
   where :math:`u_\infty` is the internal energy per unit volume of an
   infinite droplet. (Now we understand the first law :eq:`firstlaw`
   a lot better!)
   
   .. inline plot example
   
   .. plot::
      
      from matplotlib.pyplot import plot
      plot(range(10))
      
   .. plot:: rechtelijn.py
      :height: 300
   
   (The End)
