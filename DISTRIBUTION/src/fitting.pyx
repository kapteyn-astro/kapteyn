from c_numpy cimport import_array, PyArray_DATA

cdef extern from "stdlib.h":
   ctypedef int size_t
   void* malloc(size_t size)
   void free(void* ptr)

cdef extern int gauestd_c( double *y,
                           double *work,
                           int    *n,
                           double *p,
                           int    *np,
                           double *rms,
                           double *cutoff,
                           double *minsig,
                           int    *q )

import numpy

import_array()

MAXPAR = 200   # same value as MAXPAR parameter in gauestd.c

def gauest(y, rms, cutamp, cutsig, q, np=200, window=False):

   cdef double *y_c, *work_c, *p_c
   cdef double rms_c, cutamp_c, cutsig_c
   cdef int    n_c, np_c, q_c, i_c

   if np>MAXPAR or np<0:
      raise ValueError, 'incorrect maximimum number of parameters: %d' % np

   if not y.dtype=='d':
      y = y.astype('f8')
   if not (y.flags.contiguous and y.flags.aligned):
      y = y.copy()

   y_c      = <double*>PyArray_DATA(y)   
   n_c      = len(y)
   work_c   = <double*>malloc(n_c*sizeof(double))    
   np_c     = np
   p_c      = <double*>malloc(3*np_c*sizeof(double))
   rms_c    = rms
   cutamp_c = cutamp
   cutsig_c = cutsig
   q_c      = q

   nfound = gauestd_c(y_c, work_c, &n_c, p_c, &np_c,
                      &rms_c, &cutamp_c, &cutsig_c, &q_c)
 
   free(work_c)

   result = []
   for i_c in range(nfound):
      result.append((p_c[i_c*3], p_c[i_c*3+1], p_c[i_c*3+2]))

   return result
