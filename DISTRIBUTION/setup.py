from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from kapteyn import __version__ as version
import sys, os, numpy

include_dirs = []
numdir = os.path.dirname(numpy.__file__)
ipath = os.path.join(numdir, numpy.get_include())
include_dirs.append(ipath)

description = """The Kapteyn Package is a collection of Python modules
and applications developed by the computer group of the Kapteyn
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
      information entered by the user (module positions)."""

classifiers = [
   'Development Status :: 5 - Production/Stable',
   'Programming Language :: Python',
   'Programming Language :: Cython',
   'Programming Language :: C',
   'Intended Audience :: Science/Research',
   'Topic :: Scientific/Engineering :: Astronomy',
   'Topic :: Scientific/Engineering :: Visualization',
   'License :: OSI Approved :: BSD License',
   'Operating System :: POSIX :: Linux',
   'Operating System :: MacOS :: MacOS X'
   ]   

setup(
   name="kapteyn",
   version=version,
   description='Kapteyn Package',
   author='J.P. Terlouw, M.G.R. Vogelaar',
   author_email='gipsy@astro.rug.nl',
   url='http://www.astro.rug.nl/software/kapteyn/',
   download_url = "http://www.astro.rug.nl/software/kapteyn/kapteyn.tar.gz",
   long_description=description,
   platforms = ['Linux', 'Mac OSX'],
   license = 'BSD',
   classifiers = classifiers,
   ext_package='kapteyn',
   ext_modules=[
      Extension(
         "wcs",
          ["src/wcs.c", "src/xyz.c", "src/eterms.c"],
          include_dirs=include_dirs,
          libraries = ["wcs"]
      ),
      Extension(
         "ascarray",
         ["src/ascarray.c"],
         include_dirs=include_dirs
      ),
      Extension(
         "_nd_image",
         ["src/nd_image.c","src/ni_filters.c", "src/ni_fourier.c",
          "src/ni_interpolation.c", "src/ni_measure.c", "src/ni_morphology.c",
          "src/ni_support.c"],
         include_dirs=include_dirs
      )
   ],
   package_dir={'kapteyn': 'kapteyn'},
   packages=['kapteyn'],
   package_data={'kapteyn': ['lut/*.lut']},
)
