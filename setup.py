from setuptools import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
import sys
sys.path.insert(0, ".")
from kapteyn import __version__ as version
from glob import glob
import sys, os


# from https://github.com/msabramo/cython-test/
# get cython before running setup(..)
#from setuptools.dist import Distribution
#Distribution(dict(setup_requires='Cython'))

try:
   import numpy
except:
   print('''
-- Error.
The Kapteyn Package requires NumPy, which seems to be unavailable here.
Please check your Python installation.
''')
   sys.exit(1)

try:
   wcslib_dir = glob('src/wcslib*/C/')[0]
except:
   print('''
-- Error.
Unable to find WCSLIB source distribution.
''')
   sys.exit(1)

include_dirs = []
numdir = os.path.dirname(numpy.__file__)
ipath = os.path.join(numdir, numpy.get_include())
include_dirs.append(ipath)
include_dirs.append('src')
include_dirs.append(wcslib_dir)

short_descr = "Kapteyn Package: Python modules for astronomical applications"

description = """The Kapteyn Package is a collection of Python modules
and applications developed by the computer group of the Kapteyn
Astronomical Institute, University of Groningen, The Netherlands.  The
purpose of the package is to provide tools for the development of
astronomical applications with Python. 

The package is suitable for both inexperienced and experienced users and
developers and documentation is provided for both groups.  The
documentation also provides in-depth chapters about celestial
transformations, spectral translations and non-linear least squares fitting.

The package's most important features:

    * The handling of spatial and spectral coordinates, WCS projections
      and transformations between different sky systems.  Spectral
      translations (e.g., between frequencies and velocities) are supported
      and also mixed coordinates.  (Modules wcs and celestial, Module wcs
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

    * A function to search for gaussian components in a profile (module
      profiles) and a class for non-linear least squares curve fitting
      (module kmpfit)"""

classifiers = [
   ['Development Status :: 5 - Production/Stable',
    'Development Status :: 4 - Beta'][int('b' in version)],
   'Programming Language :: Python',
   'Programming Language :: Cython',
   'Programming Language :: C',
   'Intended Audience :: Science/Research',
   'Topic :: Scientific/Engineering :: Astronomy',
   'Topic :: Scientific/Engineering :: Visualization',
   'Topic :: Scientific/Engineering :: Mathematics',
   'License :: OSI Approved :: BSD License',
   'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
   'Operating System :: POSIX :: Linux',
   'Operating System :: MacOS :: MacOS X',
   'Operating System :: Microsoft :: Windows'
   ]

download_url = "http://www.astro.rug.nl/software/kapteyn/kapteyn-%s.tar.gz" % version

wcsmod_src = [
   "eterms.c",
   "wcs.pyx",
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
   "wcsprintf.c",
   "wcstrig.c",
   "wcsunits.c",
   "wcsutil.c",
   "wcserr.c",
   "flexed/wcsulex.c",
   "flexed/wcsutrn.c"
]

ndimg_src = [
   "nd_image.c",
   "ni_filters.c",
   "ni_fourier.c",
   "ni_interpolation.c",
   "ni_measure.c",
   "ni_morphology.c",
   "ni_support.c",
]

wcs_src       = (   ['src/'        + source for source in wcsmod_src]
                  + [wcslib_dir    + source for source in wcslib_src]  )

_nd_image_src = ['src/ndimg/'  + source for source in ndimg_src]

define_macros = []

# MS Windows adjustments
#
if sys.platform == 'win32':
    define_macros.append(('YY_NO_UNISTD_H', None))
    define_macros.append(('_CRT_SECURE_NO_WARNINGS', None))

# avoid using buggy Apple compiler
#
if sys.platform=='darwin':
   from distutils import ccompiler
   import subprocess
   import re
   c = ccompiler.new_compiler()
   process = subprocess.Popen(c.compiler+['--version'], stdout=subprocess.PIPE)
   output = process.communicate()[0].strip()
   version = output.split()[0].decode("ascii")
   if re.match('i686-apple-darwin[0-9]*-llvm-gcc-4.2', version):
      os.environ['CC'] = 'clang'
#from Cython.Build import cythonize

class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

def extensions():
    from Cython.Build import cythonize
    extensions = [
      Extension(
         "wcs", wcs_src,
          include_dirs=include_dirs,
          define_macros=define_macros
      ),
      Extension(
         "ascarray",
         ["src/ascarray.pyx"],
         include_dirs=include_dirs
      ),
      Extension(
         "profiles",
         ["src/profiles.pyx", "src/gauestd.c"],
         include_dirs=include_dirs
      ),
      Extension(
         "_nd_image", _nd_image_src,
         include_dirs=include_dirs
      ),
      Extension(
         "kmpfit",
         ["src/kmpfit.pyx", "src/mpfit.c"],
         include_dirs=include_dirs
      ),

      ]
    return cythonize(extensions)

setup(
   name="kapteyn",
   version=version,
   description=short_descr,
   author='J.P. Terlouw, M.G.R. Vogelaar, M.A. Breddels',
   author_email='gipsy@astro.rug.nl',
   url='http://www.astro.rug.nl/software/kapteyn/',
   download_url = download_url,
   long_description=description,
   platforms = ['Linux', 'Mac OSX', 'Windows'],
   license = 'BSD',
   setup_requires=["Cython", "numpy"],
   classifiers = classifiers,
   ext_package='kapteyn',
   ext_modules=lazy_cythonize(extensions),
   package_dir={'kapteyn': 'kapteyn'},
   packages=['kapteyn'],
   package_data={'kapteyn': ['lut/*.lut']},
)


""""""