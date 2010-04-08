Module interpolation
====================

This module is a slightly modified version of  SciPy's
`scipy.ndimage.interpolation <http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.interpolation>`_.
It is included in the Kapteyn Package mainly for convenience reasons.
In this way users of the package do not need to have all of SciPy installed,
of which only the function :func:`map_coordinates()` is currently used.

Modification
------------

If the source array contains one or more NaN values, and the *order*
argument is larger than 1, the unmodified module's function
:func:`map_coordinates()` will return an array with all NaN values.
The modification prevents this by replacing NaN values by nearby finite
values.
