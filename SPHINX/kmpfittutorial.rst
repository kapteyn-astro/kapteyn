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

Parameter ``x`` is a NumPy array and ``p`` is an array with parameters. This function
calculates y values for a given set of parameters and an array with x values.

Then it is simple to define a so called residual function which calculates the 
residuals between data points and model::

   def residuals(p, data):
      x, y = data
      return y - model(p,x)

This function has always two parameters. The first one is ``p`` which is an array
with parameter values in the order as defined in your model, and ``data``
which is an object that stores the data arrays that you need in your residual function.
The object could be anything but a list or tuple is often most practical to store the required 
NumPy arrays with data. We will explain a bit more about this object when we discuss
the constructor of a *Fitter* object.
We need not to worry about the sign of the residuals because the
fit routine calculates the the square of the residuals itself. Of course we can combine both
functions ``model`` and ``residuals`` in one function. This is a bit more efficient in Python,
but usually it is handy to have the model function available for if you want for 
instance a plot of the model with the best-fit parameters. 

.. note::

   A residuals function should always return a NumPy double-precision floating-point number
   array (i.e. dtype='d'). If your data in argument ``data`` is a list or is an array
   with single precision floating point numbers, you need to convert the result to
   the required type.


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
residuals function is and which arrays the residuals function needs 
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



Method ``simplefit()``
-----------------------

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


Gaussian profiles
-----------------

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


Explicit partial derivatives
----------------------------

In the documentation of the IDL version of *mpfit.pro*, the author states that it
is often sufficient and even faster to allow the fit routine to calculate the
derivatives numerically. However, when we work with *kmpfit*, we need an external function
to evaluate the residuals. Such functions delay the fit routine so in Python it is efficient 
to keep the number of function calls as low as possible. With explicit partial derivatives
we usually gain an increase in speed of about 20%, at least for fitting Gaussian profiles.
Probably the reduction in calls to the residuals function is bigger than the cost of calculating 
the explicit partial derivatives.
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
     -Arrays given in the ``data`` argument in the constructor of the Fitter object.
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
      return numpy.divide(pderiv, -err)

Note that all the values per parameter are stored in a row. With NumPy's ``divide`` we
divide each row elements-wise by the error. A minus sign is added to fulfill the 
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
increase of :math:`\chi^2` = 1. That implies that if we we add the standard error to
its corresponding parameter, fix it in a second fit and fit again, the value of
:math:`\chi^2` will be increased by 1.

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

Assume we have N data points and each data point has an individual error of
:math:`\sigma_i`. Then chi-square (which we try to minimalize in our fits) is
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
If a parameter :math:`p_j` has covariance error :math:`C_{jj}` then:

.. math::
   :label: bevington_scalederror

   \sigma_{p_j} \approx s^2 C_{jj}

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
-----------------

We need to discuss the bootstrap method that we used in the last script in some detail
Your data realized a set of best-fit parameters, say :math:`p_{(0)}`. This data set is one
of many different data sets that represent the 'true' parameter set :math:`p_{true}` . 
Each data set will
give a different set of fitted parameters :math:`p_{(i)}`. These parameter sets follow
some probability distribution in the *n* dimensional space of all possible parameter sets.
To find the uncertainties in the fitted parameters we need to know the distribution 
of :math:`p_{(i)}-p_{true}` [NumRep]_. In Monte Carlo simulations of synthetic data sets 
we assume that the shape of the distribution of Monte Carlo set :math:`p_{(i)}-p_{0}` is equal to 
the shape of the real world set :math:`p_{(i)}-p_{true}`

The *Bootstrap Method* [NumRep_] uses the data set that you used to find the best-fit parameters.
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
many times (let's say 5000), then we get the distribution we needed. The standard
deviation of this distribution (i.e. for one parameter), gives the uncertainty.

.. note::

   For unweighted fits (the errors on the data points are all set to 1), one should use 
   the scaled errors in attribute ``stderr``. They match with uncertainties of the
   parameters found with the Bootstrap Method.

For linear regression we have analytical expressions for the parameter uncertainties
and we can compare them with the errors from *kmpfit*. It can be demonstrated that these
analytical values are the same as the values derived from the covariance matrix 
(attribute ``xerrors``).

Weighted fits
+++++++++++++

In fits with weights, one often uses the individual standard errors of the data points as weights.
In equation :eq:`bevington8_3` we see that the weights :math:`w_i` are defined as
:math:`w_i = 1/\sigma_i`. If we use these weights, then the value of :math:`\chi^2_{\nu}` is
often smaller than 1, indicating that we underestimated our errors on the data points, or
it is much bigger than 1, indicating that we overestimated these errors.
The interpretation of the uncertainties derived from the covariance matrix is difficult then.
The value of :math:`\chi^2` can be used to derive a value for the goodness of fit, but 
the errors are not comparable with what we find if we apply bootstrapping to 
find the standard errors on the best-fit parameters.

.. note::

   Some conclusions:

   * For unweighted fits, the standard errors on the best-fit parameters are
     the scaled errors given in attribute ``Fitter.stderr``
   * For unweighted fits, the standard errors from ``Fitter.stderr`` are comparable to
     errors we find with Monte Carlo simulations.
   * For weighted fits we trust the errors in attribute ``Fitter.stderr`` 
     if :math:`\chi^2_{\nu} \approx 1`,  but even then they can differ significantly
     from the ones we find with Monte Carlo simulations.

References
----------
   
.. [Bev] Data Reduction and Error Analysis for the Physical Sciences,
   Philip R. Bevington, 1969, McGraw-Hill

.. [NumRep] Numerical Recipes in C, The Art of Scientific Computing,
   William H. Press, Saul A. Teukolsky, William T. Vetterling and Brian P. Flannery,
   2nd edition, Cambridge University Press, 1992