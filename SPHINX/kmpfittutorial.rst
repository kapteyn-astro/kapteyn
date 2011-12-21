.. _kmpfit_tutorial:
   
Least squares fitting with kmpfit
===================================

.. highlight:: python
   :linenothreshold: 10


Introduction
------------

In this tutorial we try to show how flexible the least squares fit routine in
:mod:`kmpfit` is. We added practical examples which are explained with
the necessary theoretical background. The fit routine *kmpfit* has many similar
aspects in common with SciPy's *leastsq* function,
but its interface is a more friendly and flexible.

A least squares fit method is an algorithm that minimizes a so called
*objective function* for N data points :math:`(x_i,y_i), i=0, ...,N-1`.
These data points are measured and often :math:`y_i` has a measurement error
that is much smaller than the error in :math:`x_i`. Then we call *x* the
independent and *y* the dependent variable. In this tutorial we will
also deal with examples where the errors are comparable.

The method of least squares adjusts the parameters of a model function
*f(parameters, independent_variable)* by finding a minimum of a so called
*objective function*. This objective function is a sum of values:

.. math::
   :label: Objective_function

   S = \sum\limits_{i=0}^{N-1} r_i^2

Objective functions are also called *merit* functions.
Least squares routines also predict what the range of best-fit
parameters will be if we repeat the experiment, which produces the
data points, many times. But it can do that only for objective functions
if they return the (weighted) sum of squared residuals (WSSR). The
least squares fitting procedure is then a maximum-likelihood estimation (MLE)
And the objective function *S* is also called chi square (:math:`\chi^2`).

If we define :math:`\mathbf{p}` as the set of parameters and take *x* for the independent data
then we define a residual as the difference between the actual dependent variable
:math:`y_i` and the value given by the model:

.. math::
   :label: Residuals_function

   r(\mathbf{p}, (x_i,y_i)) = y_i - f(\mathbf{p},x_i)

One is not restricted to one independent (*explanatory*) variable. For example, 
for a plane the dependent (*response*) variable :math:`y_i`
depends on two independent variables :math:`(x1_i,x2_i)`

*kmpfit* needs a specification of the residuals function :eq:`Residuals_function`.
It defines the objective function itself by squaring the residuals and summing them
afterwards.

.. note::

   For *kmpfit*, you need only to specify a residuals function.
   The least squares fit method in *kmpfit* does the squaring and summing
   of the residuals.



A Basic example
-------------------

In this section we explain how to setup a residuals function for *kmpfit*.
We use vectorized functions written with :term:`NumPy`.

The residual function
+++++++++++++++++++++++++

Assume we have data for which we know that the relation between X
and Y is a straight line with offset *a* and slope *b*,
then a model :math:`f(\mathbf{p},\mathbf{x})`
could be written in Python as::

   def model(p, x):
      a,b = p
      y = a + b*x
      return y

Parameter ``x`` is a NumPy array and ``p`` is an array with model
parameters *a* and *b*. This function calculates response Y values
for a given set of parameters and an array with explanatory X values.

Then it is simple to define the residuals function :math:`r(\mathbf{p}, (x_i,y_i))`
which calculates the
residuals between data points and model::

   def residuals(p, data):
      x, y = data
      return y - model(p,x)

This residuals function has always two parameters.
The first one is ``p`` which is an array
with parameter values in the order as defined in your model, and ``data``
which is an object that stores the data arrays that you need in your residual function.
The object could be anything but a list or tuple is often most practical to store
the required data. We will explain a bit more about this object when we discuss
the constructor of a *Fitter* object.
We need not worry about the sign of the residuals because the
fit routine calculates the the square of the residuals itself.

Of course we can combine both
functions ``model`` and ``residuals`` in one function.
This is a bit more efficient in Python,
but usually it is handy to have the model function available if you  need
to plot the model using different sets of best-fit parameters.

.. note::

   A residuals function should always return a NumPy double-precision floating-point number
   array (i.e. dtype='d'). If your data in argument ``data`` is a list or is an array
   with single precision floating point numbers, you need to convert the result to
   the required type.

.. note::

   It is also possible to write residual functions that represent objective
   functions used in orthogonal fit procedures
   where both variables **x** and **y** have errors.



Artificial data for experiments
+++++++++++++++++++++++++++++++

For experiments with least square fits, it is often convenient to start with artificial data
which resembles the model with certain parameters, and add some Gaussian distributed
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


Initial parameter estimates
++++++++++++++++++++++++++++

Now we have to tell the constructor of the `Fitter` object what the 
residuals function is and which arrays the residuals function needs.
To create a Fitter object we use the line::

   fitobj = kmpfit.Fitter(residuals=residuals, data=(x,y))


Least squares fitters need initial estimates of the model parameters.
As you probably know, our problem is an example of 'linear regression' and this
category of models have best fit parameters that can be calculated analytically.
Then the fit results are not very sensitive to the initial values you supply.
So set the values of our initial parameters in the model (a,b) to (0,0). Use these values 
in the call to :meth:`Fitter.fit`. The result of the fit is stored in attributes 
of the Fitter object (`fitobj`). We show the use of attributes
`status`, `message`, and `params`. This last attribute stores the 'best fit' parameters,
it has the same type as the sequence with the initial parameter (i.e. NumPy array, list or tuple)::

   paramsinitial = (0.0, 0.0)
   fitobj.fit(params0=paramsinitial)
   if (fitobj.status <= 0):
      print 'Error message = ', fitobj.message
   else:
      print "Optimal parameters: ", fitobj.params

Below we show a complete example. If you run it, you should get a plot like the one
below the source code. It will not be exactly the same because we used a random number generator
to add some noise to the data.

**Example: kmpfit_example_simple.py - Simple use of kmpfit**

.. plot:: EXAMPLES/kmpfit_example_simple.py
   :include-source:
   :align: center



Function ``simplefit()``
-----------------------------

For simple fit problems we provided a simple interface.
It is a method which is used  as follows:

>>> p0 = (0,0)
>>> fitobj = kmpfit.simplefit(model, p0, x, y, err=err, xtol=1e-8)
>>> print fitobj.params

Argument ``model`` is a function, just like the model in the previous section. 
``p0`` is a sequence with initial values with length equal to the number of parameters
that defined in your model. Argument ``x`` and ``y`` are the arrays or lists that represent 
you measurement data. Argument ``err`` is an array with 1 :math:`\sigma` errors,
one for each data point. Then you can enter values to tune the fit routine
with keyword arguments (e.g. *gtol*, *xtol*, etc.).
In the next example we demonstrate how to use lists for your data points, 
how to make an unweighted fit and how to print the right parameter uncertainties.
For an explanation of parameter uncertainties, see section :ref:`standard_errors`.


The advantages of this method:

  * You need only to worry about a model function
  * No need to create a *Fitter* object first
  * Direct input of relevant arrays
  * As a result you get a Fitter object with all the attributes
  * It is (still) possible to tune the fit routine with keyword arguments,
    no limitations here.
 
**Example: kmpfit_example_easyinterface.py - Simple interface**

.. literalinclude:: EXAMPLES/kmpfit_example_easyinterface.py


Explicit partial derivatives
-----------------------------

Gaussian profiles
++++++++++++++++++

There are many examples where an astronomer needs to know the characteristics of a Gaussian profile.
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
Craig Markwardt's non-linear least squares curve fitting routines for IDL called MPFIT.
It uses the Levenberg-Marquardt technique to solve the least-squares problem, 
which is a particular strategy for iteratively searching for the best fit. 


Partial derivatives for a Gaussian
---------------------------------------------

In the documentation of the IDL version of *mpfit.pro*, the author states that it
is often sufficient and even faster to allow the fit routine to calculate the
derivatives numerically. With explicit partial derivatives
we usually gain an increase in speed of about 20%, at least for fitting Gaussian profiles.
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
   * a1, a2, ... an
     -References to data in the ``data`` argument in the constructor of the Fitter object.
   * dflags            
     -List with booleans. One boolean for each model parameter.
     If the value is ``True`` then an explicit partial derivative is
     required. The list is generated by the fit routine.

There is no need to process the ``dflags`` list in your code. There is no problem if 
you return all the derivatives even when they are not necessary.

.. note::

   A function which returns derivatives should create its own work array to store the 
   calculated values. The shape of the array should be (parameter_array.size, x_data_array.size).

The function ``my_derivs`` is then::

   def my_derivs(p, data, dflags):
      #-----------------------------------------------------------------------
      # This function is used by the fit routine to find the values for
      # the explicit partial derivatives. Argument 'dflags' is an array
      # with booleans. If an element is True then an explicit partial
      # derivative is required.
      #-----------------------------------------------------------------------
      x, y, err = data
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
      return pderiv/-err

Note that all the values per parameter are stored in a row. A minus sign is added to 
to the error array to fulfill the 
requirement in equation :eq:`dervresidual`.
The constructor of the Fitter object is as follows (the function ``my_residuals`` is
not given here)::


   fitobj = kmpfit.Fitter(residuals=my_residuals, deriv=my_derivs, data=(x, y, err))


The next code and plot show an example of finding and plotting best fit parameters given a Gauss
function as model. If you want to compare the speed between a fit with  explicit partial derivatives
and a fit using numerical derivatives, add a second Fitter object by omitting the ``deriv`` argument.
In our experience, the code with the explicit partial derivatives is about 20% faster because it
needs considerably fewer function calls to the residual function.

**Example: kmpfit_example_partialdervs.py - Finding best fit parameters for a Gaussian model**

.. plot:: EXAMPLES/kmpfit_example_partialdervs.py
   :include-source:
   :align: center



.. _standard_errors:

Standard errors of best-fit values
----------------------------------
   
With the estimation of errors on the best-fit parameters we get an idea how
good a fit is. Usually these errors are called standard errors, but often
programs call these errors also standard deviations. For nonlinear least-squares routines,
these errors are based on mathematical simplifications and are therefore often called
*asymptotic* or *approximate* standard errors.

The standard error (often denoted by SE) is a measure of the average amount that
the model over- or under-predicts.

According to [Bev]_ , the standard error is an uncertainty which corresponds to an
increase of :math:`\chi^2` = 1. That implies that if we we add the standard error
:math:`\sigma_i` to
its corresponding parameter, fix it in a second fit and fit again, the value of
:math:`\chi^2` will be increased by 1.

.. math::
   :label: bevington11_31

   \chi^2(p_i+\sigma_i) = \chi^2(p_i) + 1

The next example shows this behaviour. We tested it with the first parameter fixed and 
a second time with the second parameter fixed. The example also shows how to set
parameters to 'fixed' in *kmpfit*.
The model is a straight line. If you run
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
   Errors from covariance matrix         :  [ 0.          0.03798767]
   
   Fix second parameter and set its value to fitted value+error
   Params:         [1.9641675954511788, 3.0108297500339498]
   Chi^2 min:      57.760602835
   Errors from covariance matrix         :  [ 0.0346452  0.       ]

As you can see, the value of chi-square has increased with ~1. 


In the (astronomical) literature one often lists best-fit parameters with
the standard errors derived from the covariance matrix (in fact they are the
square root of the diagonal values of this matrix).
But note that this is only a good approximation of the Bootstrap standard errors
if you are sure that you are using weights and that
these weights are derived from real measurement errors (:math:`w_i=1/\sigma_i^2`)
and these errors should be normally distributed.
If you fit is unweighted or your weights are relative weights, then one needs
to rescale the covariance elements. 

Assume we have N data points and each data point has an individual error of
:math:`\sigma_i`. Then chi-square (which we try to minimalize in our fits) is
defined as:

.. math::
   :label: bevington8_3
   
   \chi^2 = \sum\limits_{i=0}^{N-1} {\left(\frac{\Delta y_i}{\sigma_i}\right)}^2 = \sum\limits_{i=0}^{N-1} \left[\frac{({y_i-ymodel_{i})}^2}{\sigma_i^2} \right]

The estimated sample variance of the fit does not include weighting and
is defined as:

.. math::
   :label: bevington8_29
   
   s^2 = \frac{1}{N-n} \sum\limits_{i=0}^{N-1} ({y_i-ymodel_{i})}^2

*N* is the number of data points and *n* is the number of (free) parameters.
The number *N-n* is also called the *number of degrees of freedom*.
Kmpfit returns this number as attribute ``dof``.

Unweighted fit
+++++++++++++++++

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

   \sigma_i^2 = \sigma^2 \approx s^2 = \frac{\chi^2}{N-n} \equiv \chi_{\nu}^2

Then the variance in a fitted parameter is approximately the value from the
covariance matrix diagonal times the variance *s* derived with :math:`\sigma_i=1`.
If a parameter :math:`p_j` has covariance error :math:`C_{jj}` then:

.. math::
   :label: bevington_scalederror

   \sigma_{p_j} \approx \sqrt{s^2 C_{jj}}

The uncertainties from the covariance matrix in *kmpfit* are called ``xerror``.
The scaled uncertainties are used in unweighted fits are called ``stderr``.


The next code example is a small script that shows that the scaled error estimates
are realistic if we compare them to errors found with a bootstrap method.
The plot shows the fit and the bootstrap distributions of parameter *A* and *B*.
We will explain the Bootstrap Method in the next section.


**Example: kmpfit_unweighted_bootstrap_plot.py - How to deal with unweighted fits**

.. plot:: EXAMPLES/kmpfit_unweighted_bootstrap_plot.py
   :include-source:
   :align: center



Bootstrap Method
++++++++++++++++++

We need to discuss the bootstrap method, that we used in the last script, in some detail.
Bootstrap is a tool which estimates standard errors of parameter estimates
by generating synthetic data sets with samples drawn with replacement from
the measured data and repeating the fit process with this synthetic data.


Your data realizes a set of best-fit parameters, say :math:`p_{(0)}`.
This data set is one
of many different data sets that represent the 'true' parameter set :math:`p_{true}` . 
Each data set will
give a different set of fitted parameters :math:`p_{(i)}`. These parameter sets follow
some probability distribution in the *n* dimensional space of all possible parameter sets.
To find the uncertainties in the fitted parameters we need to know the distribution 
of :math:`p_{(i)}-p_{true}` [Num]_. In Monte Carlo simulations of synthetic data sets
we assume that the shape of the distribution of Monte Carlo set :math:`p_{(i)}-p_{0}` is equal to 
the shape of the real world set :math:`p_{(i)}-p_{true}`

The *Bootstrap Method* [Num]_ uses the data set that you used to find the best-fit parameters.
We generate different synthetic data sets, all with *N* data points, by drawing 
randomly *N* data points, with replacement from the original data.
In Python we realize this as follows::

   indx = randint(0, N, N)    # Do the re-sampling using an RNG
   xr[:] = x[indx]
   yr[:] = y[indx]
   ery[:] = err[indx]

We create an array with randomly selected array indices in the range 0 to *N*.
This index array is used to create new arrays which represent our synthetic data.
Note that for the copy we used the syntax xr[:] with the colon, because we want to
be sure that we are using the same array ``xr``, ``yr`` and ``ery`` each time, because
the fit routine expects the data in these arrays (and not copies of them with the same name).
The synthetic data arrays will consist of about 37 percent duplicates. With these 
synthetic arrays we repeat the fit and find our :math:`p_{(i)}`. If we repeat this
many times (let's say 1000), then we get the distribution we needed. The standard
deviation of this distribution (i.e. for one parameter), gives the uncertainty.

.. note::

   The bigger the data set, the higher the number of bootstrap trials should be
   to get accurate statistics. The best way to find a minimum number is to plot
   the Bootstrap results as in the example.





Weighted fits
+++++++++++++

In the literature we find analytical expressions for the standard errors
of weighted fits for standard linear regression or orthogonal regression.
How should we interpret these errors? For instance in Numerical Recipes, [Num]_
we find the expressions for the best fit parameters of a model :math:`y=a+bx`

Define:

.. math::
   :label: numrep_linear1

   S \equiv \sum_{i=0}^{N-1} \frac{1}{\sigma_i^2}\ \ S_x \equiv \sum_{i=0}^{N-1} \frac{x_i}{\sigma_i^2} \ \ S_y \equiv \sum_{i=0}^{N-1} \frac{y_i}{\sigma_i^2}\\
   S_{xx} \equiv \sum_{i=0}^{N-1} \frac{x_i^2}{\sigma_i^2}\ \ S_{xy} \equiv \sum_{i=0}^{N-1} \frac{x_iy_i}{\sigma_i^2}

Then the system is rewritten into the simple equations:

.. math::
   :label: numrep_linear


   aS + bS_x = S_y\ \ \ aS_x + bS_{xx} = S_{xy}

.. math::
   :label: numrep_linear3

   \Delta \equiv SS_{xx} - (S_x)^2


The solutions for *a* and *b* are:

.. math::
   :label: numrep_linear4

   a = \frac{S_{xx}S_y - S_xS_{xy}}{\Delta}\\
   b = \frac{S_{}S_{xy} - S_xS_{y}}{\Delta}

and the standard errors are:

.. math::
   :label: numrep_standarderrors

   \sigma_a^2 = \frac{S_{xx}}{\Delta}\\
   \sigma_b^2 = \frac{S}{\Delta}


It is easy to demonstrate that these errors are the same as those we find with
*kmpfit* in attribute ``xerror``, which are square-root diagonal values of
the covariance matrix in attribute ``covar``.


.. literalinclude:: EXAMPLES/kmpfit_linearreg.py


.. note::
   
   In the example output of the program you can verify that if we scale
   the measurement errors with factor 10, the covariance errors are also
   scaled with a factor 10, while the errors in attribute ``stderr``
   are unaltered and therefore not sensitive to the absolute values
   of the measurement errors.
   
.. note::

   The script above demonstrates (at least for one model) that analytical
   errors are the same as those we find as with *kmpfit* in attribute
   ``xerror`` which are the square root of the diagonal elements of the
   covariance matrix (attribute ``covar``).
   The discussion about which standard errors to use (scaled or unscaled),
   also applies to the analytically derived standard errors.

.. note::

   The uncertainties given in attribute ``xerror`` and ``stderr`` are the same,
   only when :math:`\chi_{\nu}^2 = 1`



**Weights from measurement errors**

In fits with weights, one often uses the individual standard errors of the data
points as weights.
In equation :eq:`bevington8_3` we see that the weights :math:`w_i` are defined as
:math:`w_i = 1/\sigma_i^2`. This choice has the practical advantage that scaling of the
measurement errors does not change the values of the best-fit parameters.
If we use real measurement errors and and use weighted fit procedures then
the value of :math:`\chi_{\nu}^2` can be used as a *goodness of fit* estimator.
If we assume a correct model and its value is smaller than 1,
we underestimated our errors in the data points. If we assume a correct model
and its value is (much) bigger than 1, we probably overestimated
these errors.
Alper, [Alp]_ states that for some combinations of model, data and weights,
*the standard error estimates from diagonal elements of the covariance
matrix neglect the interdependencies between parameters and lead
to erroneous results*. Often the measurement errors are difficult to obtain precisely,
sometimes these errors are not normally distributed.


**Relative weights**

If the weights are relative weights, we don't have any idea how well chi-square
estimates the sample variance. Then we scale the standard errors in the same way
as in the unweighted fit. Implicitly we assume a perfect fit (:math:`\chi_{\nu}^2 = 1`).
This sounds reasonable because these scaled errors in attribute ``stderr`` do not
change when we scale the weights. Scaling the weights with scale factor *q*
will scale the reduced chi-square with :math:`1/q^2` (:eq:`bevington8_3`), but
scale the covariance errors with :math:`q^2`. According to our correction formula
in :eq:`bevington_scalederror`, this implies that the standard error does
not change. This is the expected behaviour if we work with relative weights.
If you examine the example output of the script, you can verify this behaviour.

.. note::

   Some conclusions:

   * For unweighted fits, the standard errors on the best-fit parameters are
     the scaled errors given in attribute ``Fitter.stderr``
   * For unweighted fits, the standard errors from ``Fitter.stderr`` are comparable to
     errors we find with Monte Carlo simulations.
   * For weighted fits with real measurement errors we use the standard errors
     derived from the covariance matrix given in attribute ``Fitter.xerror``
   * For weighted fits with relative weights we use the standard errors
     given in attribute ``Fitter.stderr`` assuming :math:`\chi^2_{\nu} \approx 1`
     but even then they can differ significantly from the values we find with
     Monte Carlo simulations.


Fitting data when both variables have uncertainties
----------------------------------------------------

Sometimes your data contains errors in the *response* (dependent) variable
y (i.e. we have values for :math:`\sigma_y`) and in the *explanatory* (independent) variable x
(i.e. we have values for :math:`\sigma_x`).
In the next sections we describe a method to use *kmpfit* for this category
of least squares fit problems.


Orthogonal Distance Regression (ODR)
+++++++++++++++++++++++++++++++++++++


Assume we have a model function *f(x)* and on that curve we have a
data point :math:`(\hat{x},\hat{y}) = (\hat{x},f(\hat{x}))` which has the shortest distance to a
data point :math:`(x_i,y_i)`. The distance between those points is:

.. math::
   :label: orthdistance

   D_i(\hat{x}) = \sqrt{{(x_i-\hat{x})}^2 + {(y_i-f(\hat{x}))}^2}

or more general with weights in :math:`\hat{x},\hat{y}`

.. math::
   :label: weighted_orthdistance

   D_i(\hat{x}) = \sqrt{w_{xi}{(x_i-\hat{x})}^2 + w_{yi}{(y_i-f(\hat{x}))}^2}

The problem with this distance function is that it is not usable as an
:term:`Objective Function`
because we don't have the model values for :math:`\hat{x}`. But there is a
condition that can be used to express :math:`\hat{x}` in known variables :math:`x_i`
and :math:`y_i`
Orear [Ore]_ showed that for any model *f(x)* for which

.. math::
   :label:  Taylor

   f(\hat{x}) = f(x_i)+(\hat{x}-x_i)f^{\prime}(x_i)

is a good approximation, we can find an expression for a usable objective function.
:math:`D_i(\hat{x})` has a minimum for :math:`\frac{\partial{D}}{\partial{\hat{x}}} = 0`.
Insert :eq:`Taylor` in :eq:`weighted_orthdistance` and take the derivative to find
the condition for the minimum:

.. math::
   :label: derivative

   \frac{\partial{D}}{\partial{\hat{x}}} = \frac{\partial{}}{\partial{\hat{x}}}\sqrt{w_{xi}{(x_i-\hat{x})}^2 + w_{yi}{(y_i-[f(x_i)+(\hat{x}-x_i)f^{\prime}(x_i)])}^2} = 0

Then one derives:

.. math::
   :label: extra_condition

   -2w_x(x_i-\hat{x})-2w_y\left( y_i-\left [ f(x_i)-(x_i-\hat{x}){f^{\prime}}(x_i)\right]\right ) {f^{\prime}}(x_i) = 0

so that:

.. math::
   :label: extra_condition2

   (x_i-\hat{x}) = \frac{-w_y\big( y_i- f(x_i)\big){f^{\prime}}(x_i) }{w_x+w_y{f^{\prime}}^2(x_i)}

If you substitute this in :eq:`weighted_orthdistance`,
then (after a lot of re-arranging) one finds for the objective function:

.. math::
   :label: Orear

   D_i^2(\hat{x}) \approx \frac{w_{xi}w_{yi}}{w_{xi}+w_{yi}{f^{\prime}}^2(x_i) }{(y_i-f(x_i))}^2

If we use statistical weighting with
weights :math:`w_{xi}=1/{\sigma_{xi}}^2` and :math:`w_{yi}=1/{\sigma_{yi}}^2`,
we can write this as:

.. math::
   :label: common_distance

   \boxed{\chi^2 = \sum\limits_{i=0}^{N-1} D_i^2 = \frac{{\big(y_i-f(x_i)\big)}^2}{\sigma^2_{yi}+\sigma^2_{xi}{f^{\prime}}^2(x_i)}}


Effective variance
+++++++++++++++++++

The method in the previous section can also be explained in another way:
Clutton [Clu]_ shows that for a model function *f*,
the effect of a small error :math:`\delta x_i` in :math:`x_i` is to change the
measured value :math:`y_i` by an amount :math:`f^\prime (x_i) \delta x_i` and that
as a result, the *effective variance* of a data point *i* is:

.. math::
   :label: clutton

   var(i) = var(y_i) + var(f^\prime(x_i)) = \sigma_{y_i}^2 + {f^\prime}^2(x_i) \sigma_{x_i}^2


Best parameters for a straight line
++++++++++++++++++++++++++++++++++++

Equation :eq:`common_distance` can be used to create an objective function.
We sow this for a model which represents a straight line :math:`f(x)=a+bx`.
For a straight line the Taylor approximation :eq:`Taylor` is exact.
This can be seen as follows:
With :math:`f^\prime(x) = b`. The relation :math:`f(x) = f(x_i)+(x-x_i)f^{\prime}(x_i)`
is equal to :math:`f(x) = f(x_i)+(x-x_i)b = a+bx_i+bx-bx_i = a+bx`.

The objective function, chi-square, that needs to be minimized for a straight line
is then:

.. math::
   :label: errorinxandy

   \chi^2 = \sum\limits_{i=0}^{N-1} D_i^2 = \sum\limits_{i=0}^{N-1} \frac{{(y_i-a-bx_i)}^2}{\sigma_{y_i}^2 + \sigma_{x_i}^2 b^2 }

This formula seems familiar. It resembles an ordinary least squares objective
function but with 'corrected' weights in Y.
A suitable residuals function for *kmpfit* is the square root of this
objective function::

   def residuals(p, data):
      a, b = p
      x, y, ex, ey = data
      w = ey*ey + b*b*ex*ex
      wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
      d = wi*(y-model(p,x))
      return d



Pearson's data
++++++++++++++++

Another approach to find the best fit parameters for orthogonal fits of straight lines
starts with the observation that best (unweighted) fitting straight lines for given data points
go through the centroid of the system. This applies to regression of y on x,
regression of x on y and also for the result of an orthogonal fit.

.. note::

   Unweighted best fitting straight lines for given data points
   go through the centroid of the system.

If we express our straight line as :math:`y=b+\tan(\theta)x` and substitute the
coordinates of the centroid :math:`(\bar{x}, \bar{y})`, we get the expression
for a straight line:

.. math::
   :label: straightline

   \tan(\theta)x - y + \bar{y}-\tan(\theta)\bar{x} = 0

For a line :math:`ax+by+c=0` we know that the distance of a data point
:math:`(x_i,y_i)` to this line is given by: :math:`(ax_i+by_i+c)/\sqrt{(a^2+b^2)}`.
If we use this for :eq:`straightline` then we derive an expression for
the distance *D*:

.. math::
   :label: disttan

   D_i = \left[\tan(\theta)x_i - y_i + \bar{y}-\tan(\theta)\bar{x}\right]\cos(\theta)

For an objective function we need to minimize:

.. math::
   :label: disttan2

   \sum\limits_{i=0}^{N-1} D_i^2 =\sum\limits_{i=0}^{N-1}  {\left[\tan(\theta)x_i - y_i + \bar{y}-\tan(\theta)\bar{x}\right]}^2\cos(\theta)^2

To minimize this we set the first partial derivative with respect to :math:`\theta`
to 0 and find the condition:

.. math::
   :label: conditiontan

   \tan(2\theta) = \frac{\sum\limits_{i=0}^{N-1}(y_i-\bar{x})(y_i-\bar{x})}{\sum\limits_{i=0}^{N-1}{(y_i-\bar{y})}^2 - \sum\limits_{i=0}^{N-1}{(x_i-\bar{x})}^2}


Fitting problems like the ones we just described are not new. In 1901, Karl Pearson
published an article [Pea]_ in which he discussed a problem "where
the :term:`Independent Variable` is subject to as much deviation or error
as the :term:`Dependent Variable`.
He derived the same best-fit angle :eq:`conditiontan` in a different way (using correlation
ellipsoids). Pearson writes it as:

.. math::
   :label: Pearson

   \tan(2\theta) = \frac{2r_{xy}\sigma_x\sigma_y}{\sigma_x^2-\sigma_y^2}

where :math:`r_{xy}` is called the Pearson product-moment correlation coefficient.
Using the same variables he writes for the slope :math:`b_1` of a regression of y on x
and the slope :math:`b_2` for a regression of x on y:

.. math::
   :label: Pearsonsslope

   b_1 = \frac{r_{xy}\sigma_y}{\sigma_x},\ \ b_2 = \frac{r_{xy}\sigma_x}{\sigma_y}

with:

.. math::
   :label: Pearsonscorrcoeff

   r_{xy} = \frac{\sum\limits^{N-1}_{i=0}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum\limits^{N-1}_{i=0}(x_i - \bar{x})^2} \sqrt{\sum\limits^{N-1}_{i=0}(y_i - \bar{y})^2}}

With :eq:`Pearson` and :eq:`Pearsonsslope` we get the well known relation between
the slopes of the two regression lines and the correlation coefficient:

.. math::
   :label: Pearsonsrelation

   r_{xy}^2 = b_1*b_2

and :eq:`Pearson` can be written as:

.. math::
   :label: Pearsonb1b2

   \boxed{\tan(2\theta) = \frac{2b_1b_2}{b_2-b_1}}

On page 571 in this article he presented a table
with data points. This table has been used many times in the literature to compare
different methods.

>>> x = numpy.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
>>> y = numpy.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])

So let's prove it works with a short program.
The script in the next example calculates Pearson's best fit slope using the analytical
formulas from this section. Then it shows how one can use *kmpfit* for a
regression of y on x and for a regression of x on y. In the latter case,
we swap the data arrays x and y in the initialization of *kmpfit*.
Note that for a plot we need to transform its offset and slope in the YX plane
to an offset and slope in the XY plane. If the values are :math:`(a,b)` in the
YX plane, then in the XY plane, the offset and slope will be :math:`(-a/b, 1/b)`.

.. literalinclude:: EXAMPLES/kmpfit_Pearsonshort.py

The most remarkable fact is that Pearson applied the 'effective variance'
method, formulated at a later date, to an unweighted orthogonal fit, as can
be observed in the second plot in the figure. Pearson's best-fit parameters
are the same as the best-fit parameters we find
with the effective variance method (look in the output below).
In an extended version of the program above, we added the effective variance
method and added the offset and slope for the bisector of the two regression lines
(y on x and x on y). The results are shown in the next figure.
Note that Pearson's best-fit line is **not** the same
as the bisector which has no relation to orthogonal fitting procedures.

.. note::

   Pearson's method is an example of an orthogonal fit procedure. It cannot
   handle weights nor does it give you estimates of the errors
   on the best-fit parameters. We discussed the method because it is
   historically important and it proves that *kmpfit* can be used for its
   implementation.

.. note::

   In the example we find best-fit values for the angle :math:`\theta` from
   which we derive the slope :math:`b = \tan(\theta)`. The advantage of this method
   is that it also finds fits for data points that represent vertical lines.

**Example:  kmpfit_Pearsonsdata - Pearsons data and method (1901)**

.. plot:: EXAMPLES/kmpfit_Pearsonsdata.py
   :align: center

The output of the ``kmpfit_Pearsonsdata.py`` is::


   Analytical solution
   ===================
   Best fit parameters: a=5.7840437745  b=-0.5455611975
   Pearson's Corr. coef:  -0.976475222675
   Pearson's best tan2theta, theta, slope:  -1.55350214417 -0.49942891481 -0.545561197521
   b1 (Y on X), slope:  -0.539577274984 -0.539577274984
   b2 (X on Y), slope -1.76713124274 -0.565888925403

   ======== Results kmpfit: effective variance =========
   Params:                  5.78404377469 -0.545561197496
   Covariance errors:       [ 0.68291482  0.11704321]
   Standard errors          [ 0.18989649  0.03254593]
   Chi^2 min:               0.618572759437
   Reduced Chi^2:           0.0773215949296

   ======== Results kmpfit Y on X =========
   Params:                  [5.7611851899974615, -0.4948059176648682]
   Covariance errors:       [ 0.59895647  0.10313386]
   Standard errors          [ 0.1894852   0.03262731]
   Chi^2 min:               0.800663522236
   Reduced Chi^2:           0.100082940279

   ======== Results kmpfit X on Y =========
   Params:                  (10.358385598025167, 5.2273490890768901)
   Covariance errors:       [ 0.94604747  0.05845157]
   Standard errors          [ 0.54162728  0.03346446]
   Chi^2 min:               2.62219628339
   Reduced Chi^2:           0.327774535424

   Least squares solution
   ======================
   a1, b1 (Y on X) 5.76118519 -0.539577274869
   a2, b2 (X on Y) 5.86169569507 -0.565888925412
   Best fit tan2theta, Theta, slope:  -1.5535021437 -0.499428914742 -0.545561197432
   Best fit parameters: a=5.7840437742  b=-0.5455611974
   Bisector through centroid a, b:  5.81116055121 -0.552659830161


Comparisons of weighted fits methods
+++++++++++++++++++++++++++++++++++++

York [Yor]_ added weights to Pearsons data. This data set is a standard for
comparisons between fit routines for weighted fits. Note that the weights are given as
:math:`w_{x_i}` which is equivalent to :math:`1/\sigma_{x_i}^2`.

>>> x = numpy.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
>>> y = numpy.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
>>> wx = numpy.array([1000.0,1000,500,800,200,80,60,20,1.8,1.0])
>>> wy = numpy.array([1,1.8,4,8,20,20,70,70,100,500])

This standard set is the data we used in the next example. This program compares
different methods. One of the methods is the approach of Williamson [Wil]_
using an implementation described in [Ogr]_.

**Example:  kmpfit_Pearsonsdata_compare - Pearson's data with York's weights**

.. plot:: EXAMPLES/kmpfit_Pearsonsdata_compare.py
   :align: center

Part of the output of this program is summarized in the next table.

Literature results:

=========================== ================ ================
Reference                    a                b
=========================== ================ ================
Pearson unweighted           5.7857           -0.546
Williamson                   5.47991022403    -0.48053340745
Reed                         5.47991022723    -0.48053340810
Lybanon                      5.47991025       -0.480533415
=========================== ================ ================


Practical results:

=========================== ================ ================
Method                       a                b
=========================== ================ ================
kmpfit unweighted            5.76118519259    -0.53957727555
kmpfit weights in Y only     6.10010929336    -0.61081295310
kmpfit effective variance    5.47991015994    -0.48053339595
ODR                          5.47991037830    -0.48053343863
Williamson                   5.47991022403    -0.48053340745
=========================== ================ ================

From these results we conclude that *kmpfit* with the effective variance
residuals function, is very well suited to perform weighted
orthogonal fits for a model that represents a straight line.
If you run the program, you can observe that also the uncertainties
match.


To study the effects of weights and to compare residual functions based on
a combination of :eq:`extra_condition2` and :eq:`weighted_orthdistance` and
on the effective variance formula in :eq:`common_distance` we made a small program
which produces random noise for the model data and random weights for
the measured data in both x an y.
It also compares the results of these methods with SciPy's ODR routine.
If you run the program you will observe that the three methods agree very well.


**Example:  kmpfit_errorsinXandYPlot - Comparing methods using random weights**

.. plot:: EXAMPLES/kmpfit_errorsinXandYPlot.py
   :align: center



Effective variance method for various models
+++++++++++++++++++++++++++++++++++++++++++++

Model :math:`f([a,b],x) = ax - b/x`
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

We used data from an experiment described in Orear's article [Ore]_ to test the
effective variance method.
Orear starts with a model :math:`f([a,b],x) = ax - b/x`. He
tried to minimize the objective function by an iteration using 
:eq:`extra_condition` with the derivative :math:`f^{\prime}([a,b],x) = a + b/x^2`
and calls this the exact solution. He also iterates
using the effective variance method as in :eq:`Orear` and find small differences
between these methods. This must be the result of an insufficient convergence
criterion or numerical instability because we don't find significant difference
using these methods in a program (see example below).
The corresponding residual function for the minimum distance expression is::

   def residuals3(p, data):
      # Minimum distance formula with expression for x_model
      a, b = p
      x, y, ex, ey = data
      wx = numpy.where(ex==0.0, 0.0, 1.0/(ex*ex))
      wy = numpy.where(ey==0.0, 0.0, 1.0/(ey*ey))
      df = a + b/(x*x)
      # Calculated the approximate values for the model
      x0 = x + (wy*(y-model(p,x))*df)/(wx+wy*df*df)
      y0 = model(p,x0)
      D = numpy.sqrt( wx*(x-x0)**2+wy*(y-y0)**2 )
      return D

The residual function for the effective variance is::
  
   def residuals(p, data):
      # Residuals function for data with errors in both coordinates
      a, b = p
      x, y, ex, ey = data
      w = ey*ey + ex*ex*(a+b/x**2)**2
      wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
      d = wi*(y-model(p,x))
      return d

The conclusion, after running the example, is that *kmpfit* in combination with
the effective variance method finds best-fit parameters that are better than
the published best-fit parameters (because a smaller value for the minimum
chi-square is obtained). The example shows that for data and model like Orear's,
the effective variance, which includes uncertainties both in x and y, produces
a better fit than an Ordinary Least-Squares (OLS) fit where we treat errors in x
as being much smaller than the errors in y.


**Example:  kmpfit_Oreardata - The effective variance method with Orear's data**

.. plot:: EXAMPLES/kmpfit_Oreardata.py
   :align: center


Model :math:`f([a,b,c],x) = ax^2+bx+c`
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                             

Applying the effective variance method for a parabola
we  an objective function:

.. math::
   :label: objective_function_parabola

   \chi^2 = \sum\limits_{i=0}^{N-1} \frac{{(y_i-a-bx_i)}^2}{\sigma_{y_i}^2 + \sigma_{x_i}^2 {(b+2cx)}^2 }

and we write the following residuals function
for *kmpfit*::

   def residuals(p, data):
      # Model: Y = a + b*x + c*x*x
      a, b, c = p
      x, y, ex, ey = data
      w = ey*ey + (b+2*c*x)**2*ex*ex
      wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
      d = wi*(y-model(p,x))
      return d

How good is our Taylor approximation here?
Using :math:`f(x) \approx f(x_i)+(x-x_i)(b+2cx_i)` we find that f(x) can be approximated
by: :math:`f(x) = a + bx + cx^2 - c(x-x_i)^2`. So this approximation works if
the difference between :math:`x_i` and :math:`x` remains small.
For *kmpfit* this implies that also the initial parameter estimates must be
of reasonable quality.
Using the code of residuals function above, we observed that this approach works
adequate. It is interesting to compare the results of *kmpfit* with the results
of Scipy's ODR routine. Often the results are comparable. That is, if we start
with model parameters ``(a, b, c) = (-6, 1, 0.5)`` and initial estimates
``beta0 = (1,1,1)``,
then *kmpfit* (with smaller tolerance than the default) obtains a smaller value for
chi square in 2 of 3 trials. With initial estimates ``beta0 = (1.8,-0.5,0.1)``
it performs worse with really wrong fits.

.. note::

   *kmpfit* in combination with the effective variance method is more sensitive
   to reasonable initial estimates than Scipy's ODR.


**Example:  kmpfit_ODRparabola - The effective variance method for a parabola**

.. plot:: EXAMPLES/kmpfit_ODRparabola.py
   :align: center


Model :math:`f([a,b,c],x) = a\sin(bx+c)`
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

If your model is a model that is not linear in its parameters, then the
effective variance method can still be applied.
If your model is given for example by :math:`f(x) = a\sin(bx+c)`, which
is not linear in parameter *b*,
then :math:`f^\prime(x) = ab\cos(bx+c)`
and the effective variance in relation :eq:`clutton` can be implemented as::

   def model(p, x):
      # Model: Y = a*sin(b*x+c)
      a,b,c = p
      return a * numpy.sin(b*x+c)

   def residuals(p, data):
      # Merit function for data with errors in both coordinates
      a, b, c = p
      x, y, ex, ey = data
      w1 = ey*ey + (a*b*numpy.cos(b*x+c))**2*ex*ex
      w = numpy.sqrt(numpy.where(w1==0.0, 0.0, 1.0/(w1)))
      d = w*(y-model(p,x))
      return d

In the next script we implemented the listed model and
residuals function. The results are compared with SciPy's ODR routine.
The same conclusion applies to these results as to the results of
the parabola in the previous section. 

.. note::

   *kmpfit* with effective variance can also be used for models
   that are not linear in their parameters.

**Example:  kmpfit_ODRsinus.py - Errors in both variables**

.. plot:: EXAMPLES/kmpfit_ODRsinus.py
   :align: center




Glossary
--------

.. glossary::

   Objective Function
      An *Objective Function* is a function associated with an optimization
      problem. It determines how good a solution is. In Least Squares fit
      procedures, it is this function that needs to be minimized.
   
   Independent Variable
      Usually the **x** in a measurement. It is also
      called the explanatory variable

   Dependent Variable
      Usually the **y** in a measurement. It is also
      called the response variable

   Numpy
      NumPy is the fundamental package needed for scientific computing with Python.
      See also information on the Internet at: `numpy.scipy.org <http://numpy.scipy.org/>`_

References
----------

.. [Alp] Alper, Joseph S., Gelb, Robert I., *Standard Errors and Confidence Intervals
   in Nonlinear Regression: Comparison of Monte Carlo and Parametric Statistics*,
   J. Phys. Chem., 1990, 94 (11), pp 4747–4751 (Journal of Physical Chemistry)

.. [Bev] Bevington, Philip R. , *Data Reduction and Error Analysis for the Physical Sciences*,
   1969, McGraw-Hill

.. [Clu] Clutton-Brock, *Likelihood Distributions for Estimating Functions
   When Both Variables Are Subject to Error*, Technometrics, Vol. 9, No. 2 (May, 1967), pp. 261-269

.. [Num] William H. Press, Saul A. Teukolsky, William T. Vetterling and Brian P. Flannery,
   *Numerical Recipes in C, The Art of Scientific Computing*,
   2nd edition, Cambridge University Press, 1992

.. [Ogr] Ogren, J., Norton, J.R., *Applying a Simple Linear Least-Squares
   Algorithm to Data with Uncertainties in Both Variables*,
   J. of Chem. Education, Vol 69, Number 4, April 1992 

.. [Ore] Orear, Jay, *Least squares when both variables have uncertainties*,
   Am. J. Phys. 50(10), Oct 1982

.. [Pea] Pearson, K. *On lines and planes of closest fit to systems
   of points in space*. Philosophical Magazine 2:559-572, 1901.
   A copy of this article can be found at:
   `http://stat.smmu.edu.cn/history <http://stat.smmu.edu.cn/history/pearson1901.pdf/>`_

.. [Yor] York, D. *Least-squares fitting of a straight line*,
   Canadian Journal of Physics. Vol. 44, p.1079, 1966

.. [Wil] Williamson, *Least-squares fitting of a straight line*,
   J.A., Can. J. Phys, 1968, 46, 1845-1847

.. [Wol] Wolberg, J., *Data Analysis Using the Method of Least Squares*,
   2006, Springer