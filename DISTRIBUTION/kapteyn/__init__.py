"""Kapteyn package.

"""

from os import path

package_dir = path.abspath(path.dirname(__file__))

__all__=['celestial', 'wcs', 'wcsgrat', 'tabarray', 'maputils', 'ellinteract',
         'mplutil', 'positions']

__version__='1.9.2b11'
