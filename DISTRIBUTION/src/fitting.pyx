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
                           int    *q )

import numpy

import_array()

MAXPAR = 200   # same value as MAXPAR parameter in gauestd.c

def gauest(y, rms, cutamp, cutsig, q, ncomp=200, window=False):

   cdef double *y_c, *work_c, *p_c
   cdef double rms_c=rms, cutamp_c=cutamp, cutsig_c=cutsig
   cdef int    n_c, ncomp_c=ncomp, q_c=q, i_c

   if ncomp>MAXPAR or ncomp<0:
      raise ValueError, 'incorrect maximum number of parameters: %d' % ncomp

   if not y.dtype=='d':
      y = y.astype('f8')
   if not (y.flags.contiguous and y.flags.aligned):
      y = y.copy()

   y_c      = <double*>PyArray_DATA(y)   
   n_c      = len(y)
   work_c   = <double*>malloc(n_c*sizeof(double))    
   p_c      = <double*>malloc(3*ncomp_c*sizeof(double))

   nfound = gauestd_c(y_c, work_c, &n_c, p_c, &ncomp_c,
                      &rms_c, &cutamp_c, &cutsig_c, &q_c)

   result = []
   ncomp = min(ncomp, nfound)
   for i_c in range(ncomp):
      result.append((p_c[i_c*3], p_c[i_c*3+1], p_c[i_c*3+2]))

   free(work_c)
   free(p_c)

   return result
