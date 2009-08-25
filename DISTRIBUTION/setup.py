from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from kapteyn import __version__ as version
import os, numpy

include_dirs = []
numdir = os.path.dirname(numpy.__file__)
ipath = os.path.join(numdir, numpy.get_include())
include_dirs.append(ipath)

description = """The Kapteyn Package is a collection of Python modules and applications
made by the computer group of the Kapteyn Astronomical Institute,
University of Groningen, The Netherlands.  The purpose of the package is
to provide tools for the development of astronomical applications with
Python.  This package is still being developed and possibly some
interfaces can change in a next release.  Base module wcs has
intensively been tested and seems to be stable.  The wcs module
supports:

    * spatial and spectral coordinates
    * spectral translations (e.g. translate frequencies into velocities)
    * mixed coordinates (one world- and one pixel coordinate)
    * transformations between different sky systems"""

setup(
   name="kapteyn",
   version=version,
   description='Kapteyn Package',
   author='J.P. Terlouw, M.G.R. Vogelaar',
   author_email='gipsy@astro.rug.nl',
   url='http://www.astro.rug.nl/software/kapteyn/',
   download_url = "http://www.astro.rug.nl/software/kapteyn/kapteyn.tar.gz",
   long_description=description,
   platforms = ['Linux'],
   license = 'BSD',
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
      )
   ],
   packages=['kapteyn']
)
