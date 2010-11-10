"""
==============
Module fitting
==============

.. moduleauthor:: Hans Terlouw <J.P.Terlouw@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5

**NOTICE** the name of this module is subject to change.

Functions
---------

.. autofunction:: gauest(y, rms, cutamp, cutsig, q [, ncomp=200, smode=0])

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

import_array()

MAXPAR = 200   # same value as MAXPAR parameter in gauestd.c

def gauest(y, rms, cutamp, cutsig, q, ncomp=200, smode=0, window=False):
   """
   Function to search for gaussian components in a profile.

:param y:
   a one-dimensional NumPy array (or slice) containing the profile.
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
   It should be :math:`\leq 0`.
:param smode:
   order in which gaussian components are delivered. 0: decreasing amplitude,
   1: decreasing dispersion, 2: decreasing flux.
:returns: a list with up to *ncomp* tuples of which each tuple contains
   the amplitude, the centre and the dispersion of the gaussian, in that
   order.
   
In this function the second derivative of
the profile in the signal region is calculated by fitting
a second degree polynomal. The smoothing parameter *q*  
determines the number of points used for this (:math:`2q+1`).
The gaussians are then estimated as described by
[Schwarz1968]_.

"""

   cdef double *y_c, *work_c, *p_c
   cdef double rms_c=rms, cutamp_c=cutamp, cutsig_c=cutsig
   cdef int    n_c, ncomp_c=ncomp, q_c=q, smode_c=smode, i_c

   if ncomp>MAXPAR or ncomp<0:
      raise ValueError, 'incorrect maximum number of parameters: %d' % ncomp

   if smode>2 or smode<0:
      raise ValueError, 'incorrect smode: %d' % smode

   if not y.dtype=='d':
      y = y.astype('f8')
   if not (y.flags.contiguous and y.flags.aligned):
      y = y.copy()

   y_c      = <double*>PyArray_DATA(y)   
   n_c      = len(y)
   work_c   = <double*>malloc(n_c*sizeof(double))    
   p_c      = <double*>malloc(3*ncomp_c*sizeof(double))

   nfound = gauestd_c(y_c, work_c, &n_c, p_c, &ncomp_c,
                      &rms_c, &cutamp_c, &cutsig_c, &q_c, &smode_c)

   result = []
   ncomp = min(ncomp, nfound)
   for i_c in range(ncomp):
      result.append((p_c[i_c*3], p_c[i_c*3+1], p_c[i_c*3+2]))

   free(work_c)
   free(p_c)

   return result
