SciPy modules
=============

Mainly for convenience, SciPy's modules
`scipy.ndimage.filters <http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.filters>`_
and
`scipy.ndimage.interpolation <http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.interpolation>`_
have been included in the Kapteyn Package as :mod:`kapteyn.filters` and
:mod:`kapteyn.interpolation`.
In this way users of the package do not need to have all of SciPy installed,
of which only a few functions are currently used.
To these modules the :doc:`SciPy license <license>` applies which
is compatible with the Kapteyn Package's license.

Function :func:`map_coordinates()` from module :mod:`interpolation` has
slightly been modified.
If the source array contains one or more NaN values, and the *order*
argument is larger than 1, the unmodified function will return an array with all
NaN values.
The modification prevents this by replacing NaN values by nearby finite
values.

