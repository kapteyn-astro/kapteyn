"""Kapteyn package.

"""

from os import path

package_dir = path.abspath(path.dirname(__file__))

__all__=['celestial', 'wcs', 'wcsgrat', 'tabarray', 'maputils',
         'mplutil', 'positions', 'shapes', 'rulers', 'filters',
         'interpolation','kmpfit']

__version__='2.3'
