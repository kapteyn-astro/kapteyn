u"""
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

This module provides the class Fitter, which uses the implementation in
C of `MPFIT <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_,
Craig Markwardt's non-linear least squares curve fitting routines for
IDL.  MPFIT uses the Levenberg-Marquardt technique to solve the
least-squares problem, which is a particular strategy for iteratively
searching for the best fit.  In its typical use, MPFIT will be used to
fit a user-supplied function (the "model") to user-supplied data points
(the "data") by adjusting a set of parameters.  MPFIT is based upon the
robust routine MINPACK-1 (LMDIF.F) by Mor\u00e9 and collaborators. 

For example, a researcher may think that a set of observed data
points is best modelled with a Gaussian curve.  A Gaussian curve is
parameterized by its mean, standard deviation and normalization.
MPFIT will, within certain constraints, find the set of parameters
which best fits the data.  The fit is "best" in the least-squares
sense; that is, the sum of the weighted squared differences between
the model and data is minimized.

This version allows upper and lower bounding constraints to be placed on
each parameter, or the parameter can be held fixed. 


Class Fitter
------------
.. autoclass:: Fitter(resfunct=None, deriv=None, modfunct=None, ...)

Example
-------

Example::

   #!/usr/bin/env python
   
   import numpy
   from kapteyn import kmpfit
   
   def residuals(p, x, y, w):
      a,b,c = p
      return (y - (a*x*x+b*x+c))/w
   
   x = numpy.arange(-50,50,0.2)
   y = 2*x*x + 3*x - 3 + 2*numpy.random.standard_normal(x.shape)
   w = numpy.ones(x.shape)
   
   a = {'x': x, 'y': y, 'w': w}
   
   f = kmpfit.Fitter(resfunct=residuals, params0=[1, 2, 0], resargs=a)
   
   f.fit()                                     # call fit method
   print f.params
   print f.message
   # result:
   # [2.0001022845514451, 3.0014019147386, -3.0096629062273133]
   # mpfit (potential) success: Convergence in chi-square value (1)
   
   a['y'] = 3*x*x  - 2*x - 5 + 0.5*numpy.random.standard_normal(x.shape)
   print f(params0=[2, 0, -1])                 # call Fitter object
   # result:
   # [3.0000324686457871, -1.999896340813663, -5.0060187435412962]






"""

from numpy cimport import_array, npy_intp
from numpy cimport PyArray_SimpleNewFromData, NPY_DOUBLE, PyArray_DATA
import numpy
from libc.stdlib cimport calloc, free
from kmpfit cimport *

import_array()

MP_OK = {
   1: 'Convergence in chi-square value',
   2: 'Convergence in parameter value',
   3: 'Convergence in chi-square and parameter value',
   4: 'Convergence in orthogonality',
   5: 'Maximum number of iterations reached',
   6: 'ftol is too small; no further improvement',
   7: 'xtol is too small; no further improvement',
   8: 'gtol is too small; no further improvement'
}

MP_ERR = {
     0: 'General input parameter error',
   -16: 'User function produced non-finite values',
   -17: 'No user function was supplied',
   -18: 'No user data points were supplied',
   -19: 'No free parameters',
   -20: 'Memory allocation error',
   -21: 'Initial values inconsistent w constraints',
   -22: 'Initial constraints inconsistent',
   -23: 'General input parameter error',
   -24: 'Not enough degrees of freedom'
}

cdef int xmpfunc(int *mp, int n, double *x, double **fvecp, double **dvec,
                      void *private_data) except -1:
   cdef double *e, *f, *y, *fvec, *d, *cjac
   cdef int i, j, m
   cdef npy_intp* shape=[n]

   self = <Fitter>private_data
   for i in range(n):
      if x[i]!=x[i]:
         self.message = 'Non-finite parameter from mpfit.c'
         raise ValueError(self.message)
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
   """
:param resfunct:
      residuals function, see description below.
:param deriv:
      optional derivatives function, see description below. If a derivatives
      function is given, user-computed explicit derivatives are automatically
      set for all parameters in the attribute :attr:`parinfo`, but this can
      be changed by the user.
:param modfunct:
      model function, see description below.
:param ...:
      other parameters, each corresponding with one of the configuration
      attributes described below.

Objects of this class are callable and return the fitted parameters.

**Residuals function**

The residuals function must return a NumPy (dtype='d') array with weighted
deviations between the model and the data. Its first argument is a NumPy
array containing the parameter values. Depending on the type of
the *resargs* attribute, the function can take one or more other arguments:

- if *resargs* is a dictionary, this dictionary is used to provide the
  function with keyword arguments, e.g. if *resargs* is
  ``{'x': xdata, 'y': ydata, 'e': errdata}``, a function *f* will be called
  as ``f(params, x=xdata, y=ydata, e=errdata)``.
- if *resargs* is any other Python object, e.g. a list *l*,
  a function *f* will be called as ``f(params, l)``.

**Derivatives function**

The optional derivates function can be used to compute function
derivatives, which are used in the minimization process.  This can be
useful to save time, or when the derivative is tricky to evaluate
numerically. 

The first two arguments are a NumPy array containing the parameter values
and a list with boolean values corresponding with the parameters. If
such a boolean is True, the derivative should be computed, otherwise it
may be ignored. This usually depends on the attribute :attr:`parinfo`, in which
parameters can be fixed or numerical derivates can be specified.
In the same way as with the residuals function, the function can take one
or more other arguments, depending on the type of the attribute :attr:`resargs`.

The function must return a NumPy array with partial derivatives with respect
to each parameter. It must have shape *(m,n)*, where *m*
is the number of data points and *n* the number of parameters.

**Model function**

A model function can be used as an alternative to a residuals function
when a fixed expression for deviations between model and data is adequate.
It takes two arguments: a NumPy array containing the parameter values and
an a NumPy array with values of the independent variable ("x").
It must return a NumPy (dtype='d') array with function values:
``f(params, x)``.


**Configuration attributes**

The following attributes can be set by the user to specify a
Fitter object's behaviour.

.. attribute:: params0

   A NumPy array or a list with the initial estimates for the parameters.

.. attribute:: resargs

   Python object with information for the residuals function and the
   derivatives function. See there.

.. attribute:: parinfo

   A list of directories with parameter contraints, one directory
   per parameter. Each directory can specify the following:

   - *fixed*: a boolean value, whether the parameter is to be held fixed or
     not. Default: not fixed.
   - *limits*: a two-element list with upper end lower parameter limits or
     None, which indicates that the parameter is not bounded on this side.
     Default: no limits.
   - *step*: the step size to be used in calculating the numerical derivatives.
     Default: step size is computed automatically.
   - *side*: the sidedness of the finite difference when computing numerical
     derivatives.  This field can take four values:

      0 - one-sided derivative computed automatically

      1 - one-sided derivative :math:`(f(x+h) - f(x)  )/h`

      -1 - one-sided derivative :math:`(f(x)   - f(x-h))/h`

      2 - two-sided derivative :math:`(f(x+h) - f(x-h))/2h`

      3 - user-computed explicit derivatives

     Where :math:`h` is the value of the attribute the parameter *step*
     described above.
     The "automatic" one-sided derivative method will chose a
     direction for the finite difference which does not
     violate any constraints.  The other methods do not
     perform this check.  The two-sided method is in
     principle more precise, but requires twice as many
     function evaluations.  Default: 0.

   - *deriv_debug*: flag to enable/disable console debug logging of
     user-computed derivatives, as described above.  1=enable
     debugging; 0=disable debugging.  If debugging is enabled,
     then *side* should be set to 0, 1, -1 or 2, depending on which numerical
     derivative you wish to compare to.
     Default: 0.

.. attribute:: ftol

   Relative :math:`\chi^2` convergence criterium. Default: 1e-10

.. attribute:: xtol

   Relative parameter convergence criterium. Default: 1e-10

.. attribute:: gtol

   Orthogonality convergence criterium. Default: 1e-10

.. attribute:: epsfcn

   Finite derivative step size. Default: 2.2204460e-16 (MACHEP0)

.. attribute:: stepfactor

   Initial step bound. Default: 100.0

.. attribute:: covtol

   Range tolerance for covariance calculation. Default: 1e-14

.. attribute:: maxiter

   Maximum number of iterations. Default: 200

.. attribute:: maxfev

   Maximum number of function evaluations. Default: 0 (no limit)

.. attribute:: xvalues

   Only to be used with a model function. A NumPy array with values
   of the independent variable.

.. attribute:: yvalues

   Only to be used with a model function. A NumPy array with data values.

.. attribute:: errors

   Only to be used with a model function. A NumPy array with data uncertainties.


**Result attributes**

After calling the method :meth:`fit`, the following attributes
are available to the user:

.. attribute:: params

   A NumPy array, list or tuple with the fitted parameters. The type of
   the object is the same as the type of :attr:`params0`.

.. attribute:: xerror

   Final parameter uncertainties (:math:`1 \sigma`)

.. attribute:: covar

   Final parameter covariance (NumPy-) matrix.

.. attribute:: chi2_min

   Final :math:`\chi^2`.

.. attribute:: orignorm

   Starting value of :math:`\chi^2`.

.. attribute:: rchi2_min

   Minimum reduced :math:`\chi^2`.

.. attribute:: stderr

   Standard errors.

.. attribute:: npar

   Number of parameters.

.. attribute:: nfree

   Number of free parameters.

.. attribute:: npegged

   Number of pegged parameters.

.. attribute:: dof

   Number of degrees of freedom.

.. attribute:: resid

   Final residuals

.. attribute:: niter

   Number of iterations.

.. attribute:: nfev

   Number of function evaluations.

.. attribute:: version

   mpfit.c's version string

.. attribute:: status

   Fitting status code.

.. attribute:: message

   Message string.


**Method:**

.. automethod:: fit(params0=None)
"""

   cdef mp_par *c_pars
   cdef int m, dictarg
   cdef readonly int npar
   cdef double *c_inverr, *c_yvals, *xall
   cdef mp_config *config
   cdef mp_result *result
   cdef object params_t, parms0
   cdef object modfunct, xvals, yvals, errvals, inverr, pars
   cdef object resfunct, resargs
   cdef object deriv, dflags
   cdef object deviates
   cdef readonly object message

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
      
   def __init__(self, resfunct=None, deriv=None, modfunct=None, params0=None,
                parinfo=None, xvalues=None, yvalues=None, errors=None,
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
      self.params0 = params0                    # fitting parameters
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

   property params0:
      def __get__(self):
         return self.parms0
      def __set__(self, value):
         self.params = value
         self.parms0 = value
   
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
            self.message = 'inconsistent parameter array size'
            raise ValueError(self.message)
         xall = <double*>calloc(self.npar, sizeof(double))
         for i in range(self.npar):
            xall[i] = value[i]
         free(self.xall)
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
            self.message = 'xvalues meaningless without model function'
            raise ValueError(self.message)
         if self.m!=0:
            if value.size!=self.m:
               self.message = 'inconsistent xvalues array size'
               raise ValueError(self.message)
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
            self.message = 'yvalues meaningless without model function'
            raise ValueError(self.message)
         if self.m!=0:
            if value.size!=self.m:
               self.message = 'inconsistent yvalues array size'
               raise ValueError(self.message)
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
            self.message = 'errors meaningless without model function'
            raise ValueError(self.message)
         if self.m!=0:
            if value.size!=self.m:
               self.message = 'inconsistent errors array size'
               raise ValueError(self.message)
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
            self.message = 'inconsistent parinfo list length'
            raise ValueError(self.message)
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

   property nofinitecheck:
      def __get__(self):
         return self.config.nofinitecheck
      def __set__(self, value):
         if value is not None:
            self.config.nofinitecheck = value

   property maxfev:
      def __get__(self):
         return self.config.maxfev
      def __set__(self, value):
         if value is not None:
            self.config.maxfev = value
      def __del__(self):
         self.config.maxfev = 0

   property chi2_min:
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

   property dof:
      def __get__(self):
         return self.m - self.nfree

   property rchi2_min:
      def __get__(self):
         return self.chi2_min/self.dof
         
   property stderr:
      def __get__(self):
         return numpy.sqrt(numpy.diagonal(self.covar)*self.rchi2_min) 

   cdef allocres(self):
      # allocate arrays in mp_result_struct
      self.result.resid = <double*>calloc(self.m, sizeof(double))
      self.result.xerror = <double*>calloc(self.npar, sizeof(double))
      self.result.covar = <double*>calloc(self.npar*self.npar, sizeof(double))

   def fit(self, params0=None):
      """
Perform a fit with the current values of parameters and other attributes.

Optional argument *params0*: initial fitting parameters.
(Default: previous initial values are used.)
"""
      cdef mp_par *parinfo
      if params0 is not None:
         self.params0 = params0
      else:
         self.params = self.params0
      status = mpfit(<mp_func>xmpfunc, self.npar, self.xall,
                     self.c_pars, self.config, <void*>self, self.result)
      if status<=0:
         if status in MP_ERR:
            self.message = 'mpfit error: %s (%d)' % (MP_ERR[status], status)
         else:
            self.message = 'mpfit error, status=%d' % status
         raise RuntimeError(self.message)
      
      if status in MP_OK:
         self.message = 'mpfit (potential) success: %s (%d)' % \
                                                    (MP_OK[status], status)
      else:
         self.message = None
      return status

   def __call__(self, yvalues=None, xvalues=None, params0=None):
      self.yvalues = yvalues
      self.xvalues = xvalues
      self.fit(params0)
      return self.params
