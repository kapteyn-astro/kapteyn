.. _kmpfit_tutorial:
   
Tutorial kmpfit module
========================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------

In this tutorial we try to show how flexible the least squares fit routine in :mod:`kmpfit` is.
A least squares fit is an algorithm that minimizes the sum of squares of residuals which are
the difference between your data and that of the model using given parameters.
It tries to find parameters for which this sum is a minimum.
There is some flexibility in what you define as residual. The most common residual is defined
by the difference of the data and the model in Y only.



The residual function
---------------------

Assume we have data for which we know that the relation between X (the explanatory variable)
and Y (the response variable) is linear, then a model could be written as::

   def model(p, x):
      a,b = p
      y = a + b*x
      return y

Parameter ``x`` is a NumPy array and ``p`` is a list with parameters. This function
calculates y values for a given set of parameters and an array with x values.

Then it is simple to define a so called residual function which calculates the 
residuals between data points and model::

   def residuals(p, data):
      x, y = data
      return y - model(p,x)

This function has always two parameters. The first one is ``p`` which is (again) the
list with parameter values in the order as defined in your model, and ``data``
which is an object that stores the data arrays that you need in your residual function.
The object could be anything but usually it is a tuple which stores the required 
NumPy arrays with data. We will explain a bit more about this object when we discuss
the constructor of a *Fitter* object.
We need not to worry about the sign of the residuals because the
fit routine calculates the the square of the residuals itself. Of course we can combine both
functions ``model`` and ``residuals`` in one function, but usually it is handy to have
the model function available for is you want for instance a plot of the model 
with the best-fit parameters.

For experiments with least square fits, it is often convenient to start with artificial data
which resembles the model with certain parameters, and add some gaussian distributed 
noise to the y values.
This is what we have done in the next couple of lines:

The number of data points and the mean and width of the normal distribution 
which we use to add some noise::

   N = 50
   mean = 0.0; sigma = 0.6

Finally we create a range of x values and use our model with arbitrary model parameters
to create y values::

   xstart = 2.0; xend = 10.0
   x = numpy.linspace(3.0, 10.0, N)
   paramsreal = [1.0, 1.0]
   noise = numpy.random.normal(mean, sigma, N)
   y = model(paramsreal, x) + noise


Now we have to tell the constructor of the `Fitter` object what the 
residuals function is and which arrays the residuals function needs 
To create a Fitter object we use the line::

   fitobj = kmpfit.Fitter(residuals=residuals, data=(x,y))


Least squares fitters need initial estimates of the model parameters.
As you probably know, our problem is an example of 'linear regression' and this
catagory of models have best fit parameters that can be calculated analytically.
Then the fit results are not very sensitive to the initial values you supply.
So set the values of our initial parameters in the model (a,b) to (0,0). Use these values 
in the call to :meth:`kmpfit.fit`. The result of the fit is stored in attributes 
of the Fitter object (`fitobj`). We show the use of attributes
`status`, `errmes`, and `params`. This last attribute stores the 'best fit' parameters::

   paramsinitial = (0.0, 0.0)
   fitobj.fit(params0=paramsinitial)
   if (fitobj.status <= 0):
      print 'Error message = ', fitobj.errmsg
   else:
      print "Optimal parameters: ", fitobj.params

Below we show a complete example. If you run it, you should get a plot like the one
below the source code. It will not be exactly the same because we used a random number generator
to add some noise to the data.

**Example: kmpfit_example_simple.py - Simple use of kmpfit**

.. plot:: EXAMPLES/kmpfit_example_simple.py
   :include-source:
   :align: center



Gaussian profiles
-----------------

There are many examples where an astronomer needs to know the characteristics of a gaussian profile.
Fitting best parameters for a model that represents a Gauss function, is a way to obtain a measure for
the peak value, the position of the peak and the width of the peak. It does not reveal any skewness or
kurtosis of the profile, but often these are not important. We write the Gauss function as:

.. math::
   :label: gaussianfunction

   f(x) = A{e^{-\frac{1}{2} {\left(\frac{x - \mu}{\sigma}\right)}^2}} + z_0

Here :math:`A` represents the peak of the Gauss, :math:`\mu` the mean, i.e. the position of the peak
and :math:`\sigma` the width of the peak. We added :math:`z_0` to add a background to the profile
characteristics. In the early days of fitting software, there were no implementations that did not need
partial derivatives to find the best fit parameters. The fit routine in `kmpfit` is based on 
Craig Markwardt 's non-linear least squares curve fitting routines for IDL called MPFIT.
It uses the Levenberg-Marquardt technique to solve the least-squares problem, 
which is a particular strategy for iteratively searching for the best fit. 


Explicit partial derivatives
----------------------------

In the documentation of the IDL version of mpfit.pro, the author states that it
is often sufficient and even faster to allow the fit routine to calculate the
derivatives numerically. However, when we work with *kmpfit*, we need an external function
to evaluate the residuals. Such functions delay the fit routine so in Python it is efficient 
to keep the number of function calls as low as possible. With explicit partial derivatives
we gain an increase in speed of about 20%, at least for fitting gaussian profiles.
The real danger in using explicit partial derivatives seems to be that one easily makes
small mistakes in deriving the necessary equations. This is not always obvious in test-runs.
For the Gauss function in :eq:`gaussianfunction` we derived the following partial derivatives:


.. math::
   :label: partialderivatives

   \frac{\partial f(x)}{\partial A} &= e^{-\frac{1}{2} {\left(\frac{x - \mu}{\sigma}\right)}^2}\\
   \frac{\partial f(x)}{\partial \mu} &= A{e^{-\frac{1}{2} {\left(\frac{x-\mu}{\sigma}\right)}^2}}. \frac{(x-\mu)}{\sigma^2}\\
   \frac{\partial f(x)}{\partial \sigma} &= A{e^{-\frac{1}{2} {\left(\frac{x-\mu}{\sigma}\right)}^2}}. \frac{{(x-\mu)}^2}{\sigma^3}\\
   \frac{\partial f(x)}{\partial z_0} &= 1


If we want to use explicit partial derivatives in *kmpfit* we need the external residuals
to return the derivative of the model f(x) at x, with respect to any of the parameters.
If we denote a parameter from the set of parameters :math:`P = (A,\mu,\sigma,z_0)` 
with index i, then one calculates 
the derivative with a function ``FGRAD(P,x,i)``.
In fact, kmpfit needs the derivative of the **residuals** and if we defined the residuals
as ``residuals = (data-model)/err``, the residuals function should return:

.. math::
   :label: dervresidual

   \frac{\partial f(x)}{\partial P(i)} =\frac{ -FGRAD(P,x,i)}{err}

where ``err`` is the array with weights.

Below, we show a code example of how one can implement explicit partial derivatives.
We created a function, called ``my_derivs`` which calculates the derivatives for each 
parameter. We tried to make the code efficient but you should be able to recognize 
the equations from :eq:`partialderivatives`. The return value is equivalent with :eq:`dervresidual`.
The function has a fixed signature because it is called by the fitter which expects
that the arguments are in the right order. This order is:

   * p               
     -List with model parameters, generated by the fit routine
   * dflags            
     -List with booleans. One boolean for each model parameter.
     If the value is ``True`` then an explicit partial derivative is
     required. The list is generated by the fit routine.
   * a1, a2, ... an
     -Names of arrays given in ``resargs`` argument in constructor of Fitter object.

There is no need to process the ``dflags`` list in your code. There is no problem if 
you return all the derivatives even when they are not necessary.

.. note::

   A function which returns derivatives should create its own work array to store the 
   calculated values. The shape of the array should be (len(parameterlist), len(x data array)).

The function ``my_derivs`` is then::

   def my_derivs(p, dflags, x, y, err):
      A, mu, sigma, zerolev = p
      pderiv = numpy.zeros([len(p), len(x)])  # You need to create the required array
      sig2 = sigma*sigma
      sig3 = sig2 * sigma
      xmu  = x-mu
      xmu2 = xmu**2
      expo = numpy.exp(-xmu2/(2.0*sig2))
      fx = A * expo
      for i, flag in enumerate(dflags):
         if flag:
            if i == 0:
               pderiv[0] = expo
            elif i == 1:
               pderiv[1] = fx * xmu/(sig2)
            elif i == 2:
               pderiv[2] = fx * xmu2/(sig3)
            elif i == 3:
               pderiv[3] = 1.0
      return numpy.divide(pderiv, -err)

Note that all the values per parameter are stored in a row. With NumPy's ``divide`` we
divide each row elements-wise by the error. A minus sign is added to fulfill the 
requirement in equation :eq:`dervresidual`.
The constructor of the Fitter object is as follows (the function ``my_residuals`` is
not given here)::

   fa = {'x':x, 'y':y, 'err':err}
   fitobj = kmpfit.Fitter(residuals=my_residuals, resargs=fa, deriv=my_derivs)

The next code and plot show an example of finding and plotting best fit parameters given a Gauss
function as model. If you want to compare the speed between a fit with  explicit partial derivatives
and a fit using numerical derivatives, add a second Fitter object by omitting the ``deriv`` argument.
In our experience, the code with the explicit partial derivatives is about 20% faster because it
needs much less function calls to the residual function.

**Example: kmpfit_example_partialdervs.py - Finding best fit parameters for a Gaussian model**

.. plot:: EXAMPLES/kmpfit_example_partialdervs.py
   :include-source:
   :align: center





Standard errors of best-fit values
----------------------------------
   
With the estimation of errors on the best-fit parameters we get an idea how
good a fit is. Usually thse errors are called standard errors, but often
programs call these errors also standard deviations. For non linear least fit routines,
these errors are based on mathematical simplifications and are therefore often called
*asymptotic* or *approximate* standard errors.

The standard error (often denoted by SE) is a measure of the average amount that
the model over- or under-predicts.

According to [Bev]_ , the standard error is an uncertainty which corresponds to an
increase of :math:`\chi^2` = 1. That implies that if we we add the standard error to
its corresponding parameter, fix it in a second fit and fit again, the value of
:math:`\chi^2` will be increased with 1.

.. math::
   :label: bevington11_31

   \chi^2(p_i+\sigma_i) = \chi^2(p_i) + 1


The next example shows this behaviour. We tested it with the first parameter fixed and 
a second time with the second parameter fixed. The model is a straight line. If you run
the example you will see that it shows exactly the behaviour as in 
:eq:`bevington11_31`. This proves that the covariance matrix of *kmpfit* can be used to
derive standard errors.
Note the use of the ``parinfo`` attribute of the *Fitter* object to fix 
parameters. One can use an index to set values for one parameter or one can set
the values for all parameters. These values are given as a Python dictionary.
An easy way to create a dictionary is to use Python's ``dict()`` function.

.. literalinclude:: EXAMPLES/kmpfit_errors_chi2delta.py

The results for an arbitrary run::

   ======== Results kmpfit for Y = A + B*X =========
   Params:         [2.0104270702631712, 2.94745915643011]
   Errors from covariance matrix         :  [ 0.05779471  0.06337059]
   Uncertainties assuming reduced Chi^2=1:  [ 0.04398439  0.04822789]
   Chi^2 min:      56.7606029739
   
   Fix first parameter and set its value to fitted value+error
   Params:         [2.0682217814912143, 2.896736695408106]
   Chi^2 min:      57.7606030002
   Reduced Chi^2:  0.583440434345
   Errors from covariance matrix         :  [ 0.          0.03798767]
   
   Fix second parameter and set its value to fitted value+error
   Params:         [1.9641675954511788, 3.0108297500339498]
   Chi^2 min:      57.760602835
   Reduced Chi^2:  0.583440432677
   Errors from covariance matrix         :  [ 0.0346452  0.       ]


In the (astronomical) literature one often lists best-fit parameters with
the standard errors derived from the covariance matrix (in fact they are the
square root of the diagonal values of this matrix). At least if the fit uses weights.
If no weights are used, we need to rescale the covariance elements.

Assume we have N data points and each data point has an idividual error of
:math:`\sigma_i`. Then Chi square (which we try to minimalize in our fits) is
defined as:

.. math::
   :label: bevington8_3
   
   \chi^2 = \sum\limits_{i=0}^N {\left(\frac{\Delta y_i}{\sigma_i}\right)}^2 = \sum\limits_{i=0}^N \left[\frac{({y_i-y_{i}model)}^2}{\sigma_i^2} \right]

The sample variance of the fit is defined as:

.. math::
   :label: bevington8_29
   
   s^2 = \frac{1}{N-n} \sum\limits_{i=0}^N ({y_i-y_{i}model)}^2

*N* is the number of data points and *n* is the number of (free) parameters.
The number *N-n* is also called the *number of degrees of freedom*.
Kmpfit returns this number as attribute ``dof``.

We know that the best experimental estimate of the *parent standard deviation* 
:math:`\sigma` is given by the experimental sample standard deviation *s*.
If the standard deviations :math:`\sigma_i` for the data points :math:`y_i`
are all equal then :math:`\sigma_i=\sigma` and therefore:

.. math::
   :label: bevington8_30
   
   \sigma_i^2 = \sigma^2 \approx s^2


Combining :eq:`bevington8_29` and :eq:`bevington8_30` we find:

.. math::
   :label: bevington_combined

   \sigma_i^2 = \sigma^2 \approx s^2 = \frac{\chi^2}{N-n} \equiv \chi_{\nu}

Then the variance in a fitted parameter is approximately the value from the
covariance matrix diagonal times the variance *s* derived with :math:`\sigma_i=1`.
If a parameter :math:`p_j` has covariance error :math:`C_{ii}` then:

.. math::
   :label: bevington_scalederror

   \sigma_{p_j} \approx s^2 C_{ii}

The uncertainties from the covariance matrix in *kmpfit* are called ``xerror``.
The scaled uncertainties are used in unweighted fits are called ``stderr``.

Next code example is a small script that shows that the scaled error estimates
are realistic if we compare them to errors found with a bootstrap method.

.. literalinclude:: EXAMPLES/kmpfit_unweighted_bootstrap.py


References
----------
   
.. [Bev] Data Reduction and Error Analysis for the Physical Sciences,
   Philip R. Bevington, 1969, McGraw-Hill


