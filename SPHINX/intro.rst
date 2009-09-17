Introduction
============

The Kapteyn Package is a collection of Python modules and applications
developed by the computer group of the Kapteyn Astronomical Institute,
University of Groningen, The Netherlands. 
The purpose of the package is to provide tools for the development of
astronomical applications with Python.

It is focused on both unexperienced and experienced users and developers
and documentation for both categories is provided. 

There is much development in progress and possibly some interfaces can
change in a next release. 

Overview
--------

The the following modules are included:

- :mod:`wcs`, a binary module which handles spatial and spectral coordinates and provides
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

- :mod:`tabarray`, providing a class for the efficient reading, writing and
  manipulating simple table-like structures in text files. 

- :mod:`mplutil`, utilities for use with matplotlib.
  Currently only class AxesCallback, which provides a more powerful
  mechanism for handling events from LocationEvent and derived classes
  than matplotlib provides itself.

.. ascarray left out
  :mod:`ascarray`, a binary module containing the base function for
  module :mod:`tabarray`.

.. index:: prerequisites, Python, WCSLIB, NumPy, PyFITS, matplotlib

Prerequisites
-------------

To install the Kapteyn Package, at least Python_ 2.4, Mark Calabretta's WCSLIB_,
and NumPy_ are required. For using it, the availability of
PyFITS_ and matplotlib_ are recommended.
   
.. _Python: http://www.python.org/
.. _WCSLIB: http://www.atnf.csiro.au/people/mcalabre/WCS/
.. _NumPy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _matplotlib: http://matplotlib.sourceforge.net/

.. index:: download

Download
--------

The Kapteyn Package can be downloaded from this location:
http://www.astro.rug.nl/software/kapteyn/kapteyn.tar.gz.

.. index:: install

Installing
----------

First unpack the downloaded kapteyn.tar.gz and go to the
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
