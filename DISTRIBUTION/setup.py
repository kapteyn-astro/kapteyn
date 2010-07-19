from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from kapteyn import __version__ as version

import sys, os

try:
   import numpy
except:
   print '''
-- Error.
The Kapteyn Package requires NumPy, which seems to be unavailable here.
Please check your Python installation.
'''
   sys.exit(1)

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
   ['Development Status :: 5 - Production/Stable',
    'Development Status :: 4 - Beta'][int('b' in version)],
   'Programming Language :: Python',
   'Programming Language :: Cython',
   'Programming Language :: C',
   'Intended Audience :: Science/Research',
   'Topic :: Scientific/Engineering :: Astronomy',
   'Topic :: Scientific/Engineering :: Visualization',
   'License :: OSI Approved :: BSD License',
   'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
   'Operating System :: POSIX :: Linux',
   'Operating System :: MacOS :: MacOS X'
   ]   

kapteyn_src = [
   "ascarray.c",
   "eterms.c",
   "wcs.c",
   "xyz.c"
]

wcslib_src = [
   "cel.c",
   "lin.c",
   "log.c",
   "prj.c",
   "spc.c",
   "sph.c",
   "spx.c",
   "tab.c",
   "wcs.c",
   "wcsfix.c",
   "wcshdr.c",
   "wcstrig.c",
   "wcsulex.c",
   "wcsunits.c",
   "wcsutil.c",
   "wcsutrn.c"
]

scipy_src = [
   "nd_image.c",
   "ni_filters.c",
   "ni_fourier.c",
   "ni_interpolation.c",
   "ni_measure.c",
   "ni_morphology.c",
   "ni_support.c",
]

wcs_src       = (   ['src/'        + source for source in kapteyn_src]
                  + ['src/wcslib/' + source for source in wcslib_src]  )

_nd_image_src = ['src/scipy/'  + source for source in scipy_src]


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
         "wcs", wcs_src,
          include_dirs=include_dirs
      ),
      Extension(
         "ascarray",
         ["src/ascarray.c"],
         include_dirs=include_dirs
      ),
      Extension(
         "_nd_image", _nd_image_src,
         include_dirs=include_dirs
      )
   ],
   package_dir={'kapteyn': 'kapteyn'},
   packages=['kapteyn'],
   package_data={'kapteyn': ['lut/*.lut']},
)
