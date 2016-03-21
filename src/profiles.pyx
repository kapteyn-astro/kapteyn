"""
===============
Module profiles
===============

.. author:: Hans Terlouw <gipsy@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5


Function
--------

.. autofunction:: gauest(x, y, rms, cutamp, cutsig, q [, ncomp=200, smode=0, flat=False])

Reference
---------

.. [Schwarz1968]
   **Schwarz**, U.J., 1968.
   *Analysis of an Observed Function into Components,
   using its Second Derivative*,
   Communication from the Netherlands Foundation for Radio Astronomy
   and the Kapteyn Astronomical Laboratory at Groningen, 19 405.
   (:download:`local copy<EXTERNALDOCS/Schwarz-1968.pdf>`)

"""

from c_numpy cimport import_array, PyArray_DATA

cdef extern from "stdlib.h":
   ctypedef int size_t
   void* malloc(size_t size)
   void free(void* ptr)

cdef extern int gauestd_c( double *y,
                           double *work,
                           int    *n,
                           double *p,
                           int    *ncomp,
                           double *rms,
                           double *cutoff,
                           double *minsig,
                           int    *q,
                           int    *smode )

import numpy
import functools
import_array()

MAXPAR = 200   # same value as MAXPAR parameter in gauestd.c

def gauest(x, y, rms, cutamp, cutsig, q, ncomp=200, smode=0, window=False,
           flat=False):
   """
   Function to search for gaussian components in a profile.

:param x:
   a one-dimensional NumPy array (or slice) containing the profile's
   x-coordinates.
:param y:
   a one-dimensional NumPy array (or slice) containing the profile's values.
:param rms:
   the  r.m.s. noise level of the profile.
:param cutamp:
   critical amplitude of gaussian. Gaussians
   below this amplitude will be discarded.
:param cutsig:
   critical dispersion of gaussian.
:param q:
   smoothing parameter used in calculating the
   second derivative of the profile. It must be
   greater than zero.
:param ncomp:
   maximum number of gaussian components to be found.
   It should be :math:`\geq 1`.
:param smode:
   order in which gaussian components are delivered. 0: decreasing amplitude,
   1: decreasing dispersion, 2: decreasing flux.
:param flat:
   True if a 'flat' result shoud be returned. See below.
:returns: a list with up to *ncomp* tuples of which each tuple contains
   the amplitude, the centre and the dispersion of the gaussian, in that
   order. If the argument *flat* is True, a 'flat' list with
   *ncomp* * 3 numbers is returned which may directly be used as initial
   estimates for :class:`kmpfit.Fitter`.
   
In this function the second derivative of
the profile in the signal region is calculated by fitting
a second degree polynomal. The smoothing parameter *q*  
determines the number of points used for this (:math:`2q+1`).
The gaussians are then estimated as described by
[Schwarz1968]_.

.. seealso::
   :ref:`Tutorial chapter <gauest>` "Automatic
   initial estimates for profiles with multi component Gaussians."


"""

   cdef double *y_c, *work_c, *p_c
   cdef double rms_c=rms, cutamp_c=cutamp, cutsig_c=cutsig
   cdef int    n_c, ncomp_c=ncomp, q_c=q, smode_c=smode, i_c

   if ncomp>MAXPAR or ncomp<0:
      raise ValueError, 'incorrect maximum number of parameters: %d' % ncomp

   if smode>2 or smode<0:
      raise ValueError, 'incorrect smode: %d' % smode

   if x is _gauest.x and numpy.all(x==_gauest.xcopy): # same x-values?
      xse = _gauest.xse
      equidis = _gauest.equidis
      sortindx = _gauest.sortindx
   else:
      _gauest.xcopy = x.copy()
      if numpy.all(numpy.diff(x)>0):                  # sorted?
         xs = x
         sortindx = None
      else:
         sortindx = numpy.argsort(x)
         xs = x[sortindx]
      _gauest.sortindx = sortindx
      
      step = xs[1] - xs[0]
      equidis = numpy.all(abs(numpy.diff(xs)-step)<0.01*step)
      _gauest.equidis = equidis
      if equidis:                                     # equidistant?
         xse = xs
      else:
         xse = numpy.linspace(xs[0], xs[-1], len(xs))
      _gauest.xse = xse

   if sortindx is not None:                           # should data be sorted?
      ys = y[sortindx]
   else:
      ys = y
      
   if equidis:                                        # equidistant?
      yse = ys
   else:
      yse = numpy.interp(xse, xs, ys)

   if not yse.dtype=='d':
      yse = yse.astype('f8')
   if not (yse.flags.contiguous and yse.flags.aligned):
      yse = yse.copy()

   y_c      = <double*>PyArray_DATA(yse)   
   n_c      = len(yse)
   work_c   = <double*>malloc(n_c*sizeof(double))    
   p_c      = <double*>malloc(3*ncomp_c*sizeof(double))

   d = xse[1]-xse[0]
   if d==0.0:
      raise ValueError, "Zero samples step size"
   cutsig /= d
   nfound = gauestd_c(y_c, work_c, &n_c, p_c, &ncomp_c,
                      &rms_c, &cutamp_c, &cutsig_c, &q_c, &smode_c)

   result = []
   ncomp = min(ncomp, nfound)
   for i_c in range(ncomp):
      result.append((p_c[i_c*3],
                     xse[0] + p_c[i_c*3+1]*d,
                     p_c[i_c*3+2]*d))

   free(work_c)
   free(p_c)

   if flat:
      result = list(functools.reduce(lambda x,y: x+y, result))
   return result

class _Gauest(object):
   def __init__(self):
      self.x = None
      self.sortindx = None

_gauest = _Gauest()
