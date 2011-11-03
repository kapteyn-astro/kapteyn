"""
=============
Module kmpfit
=============

.. moduleauthor:: Hans Terlouw <J.P.Terlouw@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5

.. warning::

   This chapter is currently being written and as such incomplete.


Introduction
------------

This module provides the class Fitter, which interfaces with the implementation
in C of
`MPFIT <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_,
Craig Markwardt's non-linear least squares curve fitting routines for IDL.

"""

from numpy cimport import_array, npy_intp
from numpy cimport PyArray_SimpleNewFromData, NPY_DOUBLE, PyArray_DATA
import numpy
from libc.stdlib cimport calloc, free
from kmpfit cimport *

import_array()

cdef int xmpfunc(int *mp, int n, double *x, double **fvecp, double **dvec,
                      void *private_data) except -1:
   cdef double *e, *f, *y, *fvec, *d, *cjac
   cdef int i, j, m
   cdef npy_intp* shape=[n]

   self = <Fitter>private_data
   p = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, x)
   if self.modfunct is not None:                         # model function
      f = <double*>PyArray_DATA(self.modfunct(p, self.xvals))
      e = self.c_inverr
      y = self.c_yvals
      if mp[0]:
         m = mp[0]
         fvec = fvecp[0]
      else:
         deviates = numpy.zeros((self.m,), dtype='d')
         fvec = <double*>PyArray_DATA(deviates)
         fvecp[0] = fvec
         mp[0] = m = self.m
         self.deviates = deviates     # keep a reference to protect from GC
         self.allocres()

      for i in range(m):
         fvec[i] = (y[i] - f[i]) * e[i]
         
      if self.deriv is not None:
# +++ derivative code to be added
         pass # compute derivatives and put in 'dvec'

      return 0
   else:                                                 # residuals function
      if self.dictarg:
         deviates = self.resfunct(p, **self.resargs)
      else:
         deviates = self.resfunct(p, self.resargs)

      f = <double*>PyArray_DATA(deviates)
      if mp[0]:
         m = mp[0]
         fvec = fvecp[0]
         for i in range(m):
            fvec[i] = f[i]
      else:
         fvecp[0] = f
         mp[0] = deviates.size
         self.m = mp[0]
         self.deviates = deviates       # keep a reference to protect from GC

         self.allocres()
      
      if dvec!=NULL and self.deriv is not None:
         for i in range(n):
            self.dflags[i] = bool(<int>dvec[i])
         if self.dictarg:
            jac = self.deriv(p, self.dflags, **self.resargs)
         else:
            jac = self.deriv(p, self.dflags, self.resargs)
         cjac = <double*>PyArray_DATA(jac)
         for j in range(n):
            d = dvec[j]
            if d!=NULL:
               for i in range(m):
                  d[i] = cjac[i*n+j]

      return 0

cdef class Fitter:

   cdef mp_par *c_pars
   cdef int m, npar, dictarg
   cdef double *c_inverr, *c_yvals, *xall
   cdef mp_config *config
   cdef mp_result *result
   cdef object params_t
   cdef object modfunct, xvals, yvals, errvals, inverr, pars
   cdef object resfunct, resargs
   cdef object deriv, dflags
   cdef object deviates

   def __cinit__(self):
      self.config = <mp_config*>calloc(1, sizeof(mp_config))
      self.result = <mp_result*>calloc(1, sizeof(mp_result))
      
   def __dealloc__(self):
      free(self.config)
      free(self.result.resid)
      free(self.result.xerror)
      free(self.result.covar)
      free(self.result)
      free(self.c_pars)
      free(self.xall)
      
   def __init__(self, modfunct=None, resfunct=None, deriv=None, params=None,
                xvalues=None, yvalues=None, errors=None, parinfo=None,
                ftol=None, xtol=None, gtol=None, epsfcn=None,
                stepfactor=None, covtol=None, maxiter=None, maxfev=None,
                resargs={}):
      if modfunct is not None and resfunct is not None:
         raise ValueError('cannot specify both model- and residuals functions')
      if resargs is not None and resargs is None:
         raise ValueError('resargs meaningless without residuals function')
      self.npar = 0
      self.m = 0
      self.modfunct = modfunct                  # model function
      self.resfunct = resfunct                  # residuals function
      self.deriv = deriv
      self.params = params                      # fitting parameters
      self.xvalues = xvalues
      self.yvalues = yvalues
      self.errors = errors
      self.parinfo = parinfo                    # parameter constraints
      self.ftol = ftol
      self.xtol = xtol
      self.gtol = gtol
      self.epsfcn = epsfcn
      self.stepfactor = stepfactor
      self.covtol = covtol
      self.maxiter = maxiter
      self.maxfev = maxfev
      self.resargs = resargs                    # args to residuals function
      self.dictarg = isinstance(resargs, dict)  # keyword args or one object?
   
   property params:
      def __get__(self):
         cdef npy_intp* shape = [self.npar]
         value = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, self.xall)
         if self.params_t is not None:
            return self.params_t(value)
         else:
            return value
      def __set__(self, value):
         if value is None:
            return
         cdef int i, l
         cdef double *xall
         if not isinstance(value, numpy.ndarray):
            self.params_t = type(value)
            l = len(value)
         else:
            l = value.size
         if self.npar==0:
            self.npar = l
         elif l!=self.npar:
            raise ValueError('inconsistent parameter array size')
         xall = <double*>calloc(self.npar, sizeof(double))
         for i in range(self.npar):
            xall[i] = value[i]
         self.xall = xall
         if self.dflags is None:
            self.dflags = [False]*self.npar              # flags for deriv()
         if self.deriv is not None and self.pars is None:
            self.parinfo = [{'side': 3}]*self.npar

   property xvalues:
      def __get__(self):
         return self.xvals
      def __set__(self, value):
         if value is None:
            return
         if self.modfunct is None:
            raise ValueError('xvalues meaningless without model function')
         if self.m!=0:
            if value.size!=self.m:
               raise ValueError('inconsistent xvalues array size')
         else:
            self.m = value.size
         self.xvals = value
    
   property yvalues:
      def __get__(self):
         return self.yvals
      def __set__(self, value):
         if value is None:
            return
         if self.modfunct is None:
            raise ValueError('yvalues meaningless without model function')
         if self.m!=0:
            if value.size!=self.m:
               raise ValueError('inconsistent yvalues array size')
         else:
            self.m = value.size
         if not value.dtype=='d':
            value = value.astype('f8')
         if not value.flags.contiguous and value.flags.aligned:
            value = value.copy()
         self.yvals = value
         self.c_yvals = <double*>PyArray_DATA(value)

   property errors:
      def __get__(self):
         return self.errvals
      def __set__(self, value):
         if value is None:
            return
         if self.modfunct is None:
            raise ValueError('errors meaningless without model function')
         if self.m!=0:
            if value.size!=self.m:
               raise ValueError('inconsistent errors array size')
         else:
            self.m = value.size
         self.errvals = value
         self.inverr = 1./value
         self.c_inverr = <double*>PyArray_DATA(self.inverr)
         
   property parinfo:
      def __get__(self):
         return self.pars
      def __set__(self, value):
         if value is None:
            return
         cdef mp_par *c_par
         l = len(value)
         if self.npar==0:
            self.npar = l
         elif l!=self.npar:
            raise ValueError('inconsistent parameter array size')
         self.pars = value
         if self.c_pars==NULL:
            self.c_pars = <mp_par*>calloc(self.npar, sizeof(mp_par))
         ipar = 0         
         for par in self.pars:
            if par is not None:
               c_par = &self.c_pars[ipar]

               try:
                  c_par.fixed = par['fixed']
               except:
                  c_par.fixed = 0

               try:
                  limits = par['limits']
                  for limit in (0,1):
                     if limits[limit] is not None:
                        c_par.limited[limit] = 1
                        c_par.limits[limit] = limits[limit]
               except:
                  for limit in (0,1):
                     c_par.limited[limit] = 0
                     c_par.limits[limit] = 0.0
               
               try:
                  c_par.step = par['step']
               except:
                  c_par.step = 0
                  
               try:
                  c_par.side = par['side']
               except:
                  c_par.side = 0

               try:
                  c_par.deriv_debug = par['deriv_debug']
               except:
                  c_par.deriv_debug = 0

            ipar += 1
      def __del__(self):
         free(self.c_pars)
         self.c_pars = NULL

   property ftol:
      def __get__(self):
         return self.config.ftol
      def __set__(self, value):
         if value is not None:
            self.config.ftol = value
      def __del__(self):
         self.config.ftol = 0.0

   property xtol:
      def __get__(self):
         return self.config.xtol
      def __set__(self, value):
         if value is not None:
            self.config.xtol = value
      def __del__(self):
         self.config.xtol = 0.0

   property gtol:
      def __get__(self):
         return self.config.gtol
      def __set__(self, value):
         if value is not None:
            self.config.gtol = value
      def __del__(self):
         self.config.gtol = 0.0

   property epsfcn:
      def __get__(self):
         return self.config.epsfcn
      def __set__(self, value):
         if value is not None:
            self.config.epsfcn = value
      def __del__(self):
         self.config.epsfcn = 0.0

   property stepfactor:
      def __get__(self):
         return self.config.stepfactor
      def __set__(self, value):
         if value is not None:
            self.config.stepfactor = value
      def __del__(self):
         self.config.stepfactor = 0.0

   property covtol:
      def __get__(self):
         return self.config.covtol
      def __set__(self, value):
         if value is not None:
            self.config.covtol = value
      def __del__(self):
         self.config.covtol = 0.0

   property maxiter:
      def __get__(self):
         return self.config.maxiter
      def __set__(self, value):
         if value is not None:
            self.config.maxiter = value
      def __del__(self):
         self.config.maxiter = 0

   property maxfev:
      def __get__(self):
         return self.config.maxfev
      def __set__(self, value):
         if value is not None:
            self.config.maxfev = value
      def __del__(self):
         self.config.maxfev = 0

   property bestnorm:
      def __get__(self):
         return self.result.bestnorm

   property orignorm:
      def __get__(self):
         return self.result.orignorm
         
   property niter:
      def __get__(self):
         return self.result.niter

   property nfev:
      def __get__(self):
         return self.result.nfev

   property status:
      def __get__(self):
         return self.result.status

   property nfree:
      def __get__(self):
         return self.result.nfree

   property npegged:
      def __get__(self):
         return self.result.npegged
         
   property version:
      def __get__(self):
         return self.result.version

   property covar:
      def __get__(self):
         cdef npy_intp* shape = [self.npar, self.npar]
         value = PyArray_SimpleNewFromData(2, shape, NPY_DOUBLE,
                                           self.result.covar)
         return numpy.matrix(value)

   property resid:
      def __get__(self):
         cdef npy_intp* shape = [self.m]
         value = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE,
                                           self.result.resid)
         return value
 
   property xerror:
      def __get__(self):
         cdef npy_intp* shape = [self.npar]
         value = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE,
                                           self.result.xerror)
         return value

   cdef allocres(self):
      # allocate arrays in mp_result_struct
      self.result.resid = <double*>calloc(self.m, sizeof(double))
      self.result.xerror = <double*>calloc(self.npar, sizeof(double))
      self.result.covar = <double*>calloc(self.npar*self.npar, sizeof(double))

   def fit(self, params=None):
      cdef mp_par *parinfo
      if params is not None:
         self.params = params
      status = mpfit(<mp_func>xmpfunc, self.npar, self.xall,
                     self.c_pars, self.config, <void*>self, self.result)
      if status<=0:
         raise RuntimeError('fatal mpfit error, status=%d' % status)
      return status

   def __call__(self, yvalues=None, xvalues=None, params=None):
      self.yvalues = yvalues
      self.xvalues = xvalues
      self.params = params
      self.fit()
      return self.params
