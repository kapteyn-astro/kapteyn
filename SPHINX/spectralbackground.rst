Background information spectral translations
===============================================

.. highlight:: python
   :linenothreshold: 10


Introduction
++++++++++++

This background information has been written for two reasons. First we wanted to get
some understanding of the conversions between spectral quantities and second,
we wanted to have some knowledge about legacy FITS headers (of which there must be
a lot) where applying the conversions of WCSLIB in the context of module :mod:`wcs`
without modifications will give wrong results.

.. warning::

   One needs to be aware of the fact that WCSLIB converts between frequencies
   and velocities in the same reference system while in legacy FITS headers it is
   common to give a topocentric reference frequency and a reference velocity in a
   different reference system. 

Alternate headers for a spectral line example
+++++++++++++++++++++++++++++++++++++++++++++

In "Representations of spectral coordinates in FITS" ([Ref3]_ ), section 10.1 
deals with an example of a VLA spectral line cube which is regularly sampled
in frequency (CTYPE3='FREQ'). The section describes how one can define
alternative FITS headers to deal with different velocity definitions. 
We want to examine this exercise in more detail than provided in the
article to illustrate how a FITS header can be modified and serve as an
alternate header.

The topocentric spectral properties in the FITS header from the paper are::

   CTYPE3= 'FREQ'
   CRVAL3=  1.37835117405e9
   CDELT3=  9.765625e4
   CRPIX3=  32
   CUNIT3= 'Hz'
   RESTFRQ= 1.420405752e+9
   SPECSYS='TOPOCENT'


.. note::

      For a pixel coordinate :math:`N`, reference pixel :math:`N_{ref}` with reference
      world coordinate :math:`W_{ref}` and a step size in
      world coordinates :math:`\Delta W`, the world coordinate :math:`W` is calculated with:

      .. math::
         :label: eq2
        
           W(N) = W_{ref} + (N - N_{ref}) * \Delta W  

      If *CTYPE* contains a code for a non linear conversion algorithm
      (as in CTYPE='VOPT-F2W') then this relation cannot be applied.
      
As stated in the note above, code for a conversion algorithm is important.
The statements can be verified with the following script::

   #!/usr/bin/env python
   from kapteyn import wcs
   
   Z0 = 9120000              # Barycentric optical reference velocity
   dZ0 = -2.1882651e+4       # Increment in barycentric optical velocity
   N = 32                    # Pixel coordinate of reference pixel
   
   header = {  'NAXIS'   :  1,
               'RESTWAV' :  0.211061140507,   # [m]
               'CTYPE1'  : 'VOPT',
               'CRVAL1'  :  Z0,               # [m/s]
               'CDELT1'  :  dZ0,              # [m/s]
               'CRPIX1'  :  N,
               'CUNIT1'  : 'm/s'
            }
   spec = wcs.Projection(header)
   print "From VOPT: Pixel, velocity wcs, velocity linear (%s)" % spec.units
   pixels = range(30,35)
   Vwcs = spec.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0, (Z0 + (p-N)*dZ0)/1000.0
   
   header = {  'NAXIS'   :  1,
               'CNAME1'  : 'Barycentric optical velocity',
               'RESTWAV' :  0.211061140507,   # [m]
               'CTYPE1'  : 'VOPT-F2W',
               'CRVAL1'  :  Z0,               # [m/s]
               'CDELT1'  :  dZ0,              # [m/s]
               'CRPIX1'  :  N,
               'CUNIT1'  : 'm/s'
            }
   spec = wcs.Projection(header)
   print "From VOPT-F2W: Pixel, velocity wcs, velocity linear (%s)" % spec.units
   pixels = range(30,35)
   Vwcs = spec.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0, (Z0 + (p-N)*dZ0)/1000.0
   
   # Output:
   #
   # From VOPT: Pixel, velocity wcs, velocity linear (m/s)
   # Conversion is linear; no differences
   # 30 9163.765302 9163.765302
   # 31 9141.882651 9141.882651
   # 32 9120.0 9120.0
   # 33 9098.117349 9098.117349
   # 34 9076.234698 9076.234698
   # From VOPT-F2W: Pixel, velocity wcs, velocity linear (m/s)
   # Conversion is not linear
   # 30 9163.77150335 9163.765302
   # 31 9141.88420123 9141.882651
   # 32 9120.0 9120.0
   # 33 9098.11889901 9098.117349
   # 34 9076.24089759 9076.234698


Relation optical velocity and barycentric/lsrk reference frequency
--------------------------------------------------------------------

Let's start to find the alternate header information for the header from article in
[Ref3]_ .
The extra information about the velocity there is that we have an optical barycentric 
velocity of 9120 km/s (as required by an observer) stored as an alternate FITS keyword CRVAL3Z.::
   
   CTYPE3Z= 'VOPT-F2W'
   CRVAL3Z=  9.120e+6      / [m/s]


The relation between frequency and optical velocity requires a rest frequency (RESTFRQ=).
The relation is:

.. math::
   :label: eq5

   Z = c\ \bigl(\frac{\lambda - \lambda_0}{\lambda_0}\bigr) =  c\ \bigl(\frac{\nu_0 - \nu}{\nu}\bigr)
   
We adopted variable Z for velocities following the optical definition.
The header tells us that equal steps in pixel coordinates are equal steps in frequency
and the formula above shows that these steps in terms of optical velocity is
depends on the frequency in a non-linear way. Therefore we set the conversion algorithm
to **F2W**  which indicates that there is a non linear conversion from frequency to wavelength
(optical velocities are associated with wavelength, see  [Ref3]_ .). Note that we can use wildcards
for the non linear conversion algorithm, so *CTYPE3Z='VOPT-???'* is also allowed in
our programs.


We can rewrite equation 1 into:
 
.. math::
   :label: eq10

   \nu = \frac{\nu_0}{(1+Z/c)}

If we enter the numbers we get a **barycentric** HI reference frequency:

.. math::
   :label: eq20
   
    \nu_b = \frac{1.420405752\times 10^9}{(1+9120000/299792458.0)} = 1378471216.43\ Hz

and we have part of a new alternate header::

   CTYPE3F= 'FREQ'
   CRVAL3F= 1.37847121643e+9 / [Hz]

So given an optical velocity in a reference system (in our case the barycentric system),
we can calculate which barycentric frequency we can use as a reference frequency.
For a conversion between a barycentric frequency and a barycentric velocity we
also need to know what the baraycentric frequency increment is.

Barycentric/lsrk frequency increments
--------------------------------------

.. image:: topocentriccorrection.png
   :width: 400
   :align: center

*fig.1 Overview velocities and frequencies of barycenter (B) and Earth (E) w.r.t. source.
The arrows represent velocities. The object and the Earth are moving. The longest arrow represents the
(relativistic) addition of two velocities*


Let's use index *b* for variables bound to the barycentric system and *e*
for the topocentric system.
This frequency, :math:`\nu_b` =1.37847121643 Ghz is greater than the reference frequency
:math:`\nu_e` at the observatory (FITS keyword `CRVAL3=` 1.37835117405 Ghz).

**The difference between frequencies in the topocentric and barycentric system
is caused by the difference between the velocities of reference frames B and E at 
the time of observation.**

This velocity is a *true* velocity. It is called the *topocentric correction*.

Let's try to find an expression for this topocentric correction in terms of frequencies.
The relation between a true velocity and
a shift in frequency is given by the formula

.. math::
   :label: eq30

   \nu = \nu_0\sqrt{\frac{1-v/c}{1+v/c}} = \nu_0\sqrt{\frac{c-v}{c+v}} = 
   \nu_0 \frac{c-v}{\sqrt{c^2-v^2}}
   
If we want to express the velocity in terms of frequencies, then this can be written as:

.. math::
   :label: eq40

   v = c\ \frac{\nu_0^2-\nu^2}{\nu_0^2+\nu^2}

For the radial velocities :math:`v_b` and :math:`v_e` we have:

.. math::
   :label: eq50

    v_b = c\ \frac{\nu_0^2-\nu_b^2}{\nu_0^2+\nu_b^2}=299792458.0 \ \frac{1420405752.0^2-1378471216.43^2}{1420405752.0^2+1378471216.43^2} = 8981342.29811\ m/s

and:

.. math::
   :label: eq60

   v_e = c\ \frac{\nu_0^2-\nu_e^2}{\nu_0^2+\nu_e^2}=299792458.0 \ \frac{1420405752.0^2-1378351174.05^2}{1420405752.0^2+1378351174.05^2} = 9007426.97201\ m/s

The relativistic addition of velocities in fig. 1. requires:

.. math::
   :label: eq70

   v_e = \frac{v_b + v _{t}}{1 + \frac{v_b v_{t}}{c^2}}

which gives the topocentric correction as:

.. math::
   :label: eq80

   v_t = \frac{v_e - v_{b}}{1 - \frac{v_b v_{e}}{c^2}}


With the numbers inserted we find:

.. math::
   :label: eq90

   v_t = \frac{9007426.97201 - 8981342.29811}{1 - \frac{8981342.29811\times 9007426.97201}{299792458.0^2}} = 26108.1743997\ m/s

If the FITS header has keywords with the position of the source, the time of observation and 
the location of the observatory than one can calculate the topocentric correction by hand.
This information was needed at the observatory to set a frequency for a given barycentric 
velocity. However many FITS files do not have enough information to calculate the topocentric correction.
Also it is not needed if one knows the shifted frequencies :math:`\nu_e` and :math:`\nu_b` , then
we can calculate the topocentric velocity without calculating the radial velocities.
This can be shown if we insert the expressions voor velocities :math:`v_e` and :math:`v_b` 
in the expression for :math:`v_t` . Then after some rearranging one finds:

.. math::
   :label: eq100

   v_t = c\ \frac{\nu_b^2-\nu_e^2}{\nu_b^2+\nu_e^2}

and with the numbers:

.. math::
   :label: eq110

   v_t = 299792458.0\ \frac{1378471216.43^2-1378351174.05^2}{1378471216.43^2+1378351174.05^2} = 26108.1743998\  m/s

which is consistent with :eq:`eq90`.

::

  VELOSYSZ=26108   / [m/s]


With a given topocentric correction and the reference frequency in the barycenter
we can reconstruct the reference frequency at the
observatory with :eq:`eq100` written as:

.. math::
   :label: eq120
   
   \nu_e =\nu_b\sqrt{\frac{c-v_t}{c+v_t}}
   
.. note::
   
   1) It is important to realize that the reference frequency at E is always smaller
   than the reference frequency at B because w.r.t. the source E moves faster than B.
   So if there is a change in the velocity of the source, the frequencies in B and E will change, but
   the topocentric correction keeps the same value and therefore the relation between
   the frequencies :math:`\nu_e` and :math:`\nu_b` remains the same (eq. :eq:`eq120`).


.. note::
   
   2) If we forget about the source and we have an *event on E* with a certain frequency
   than an *observer* in barycenter *B* will observe a *lower* frequency.
   This is because on the line that connects the source and B, the observatory at E moves away
   from B which decreases the remote frequency.


So if we change a frequency on E by tuning the receiver at the observatory at frequency 
:math:`\nu_e + \Delta \nu_e` ,
than the observer at B would observe a smaller frequency
:math:`\nu_b + \Delta \nu_b` .
The amount of the decrease is related to the topocentric correction as follows:

.. math::
   :label: eq130

   \nu_b+\Delta \nu_b = (\nu_e+\Delta \nu_e) \sqrt{\frac{c-v_t}{c+v_t}}
   
and therefore we can write for the frequency bandwidth in B:

.. math::
   :label: eq140

   \Delta \nu_b =\Delta \nu_e\sqrt{\frac{c-v_t}{c+v_t}}

The reason that this formula does not contradict eq. :eq:`eq120` (where the indices seems to be swapped)
is explained in note 2. Inserting the appropriate numbers:

.. math::
   :label: eq150

   \Delta \nu_b = 97656.25\ \frac{\sqrt{299792458.0-26108.1743998}}{\sqrt{299792458.0+26108.1743998}} = 97647.745732\ Hz

   
The increment in frequency therefore becomes 97.64775 khz::

   CDELT3F=  9.764775e+4 / [Hz]


So if we change CRVAL1 and CDELT1 in our demonstration script to the barycentric values,
we get the barycentric optical convention velocities for the pixels. As a check we listed 
the script and the value for pixel 32 which is exactly 9120 (km/s)::

   
   #!/usr/bin/env python
   from kapteyn import wcs
   header  = { 'NAXIS'  : 1,
               'CTYPE1' : 'FREQ',
               'CRVAL1' : 1378471216.4292786,
               'CRPIX1' : 32,
               'CUNIT1' : 'Hz',
               'CDELT1' : 97647.745732,
               'RESTFRQ': 1.420405752e+9
            }
   spec = wcs.Projection(header).spectra('VOPT-F2W')
   pixels = range(30,35)
   Vwcs = spec.toworld1d(pixels)
   print "Pixel, velocity (%s)" % spec.units
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0
   
   print "Pixel at velocity 9120 km/s: ", spec.topixel1d(9120000)
   # Output
   # Pixel, velocity (m/s)
   # 30 9163.77150423
   # 31 9141.88420167
   # 32 9120.0
   # 33 9098.11889856
   # 34 9076.2408967
   # Pixel at velocity 9120 km/s:  32.0

   
Note: A closure test is added with method `topixel1d()`

.. note::

      In the previous two sections we started with a topocentric frequency and
      a topocentric frequency increment and derived values for a barycentric frequency
      and a barycentric frequency increment. These values can be used
      to set an alternate header (barycentric frequency system 'F') for which we
      can convert between frequency and optical velocity.
      For GIPSY legacy headers these steps are used to convert between
      topocentric frequencies and velocities in another reference system,
      See :ref:`spectral_gipsy`

Increment in barycentric/lsrk optical velocity
-----------------------------------------------

The optical velocity was given by:

.. math::
   :label: eq160

   Z = c\ \bigl(\frac{\nu_0 - \nu}{\nu}\bigr) = c\ \bigl(\frac{\nu_0}{\nu} - 1\bigr)
   
Its derivative is:

.. math::
   :label: eq170
      
   \frac{dZ}{d\nu} = \frac{-c \nu_0}{\nu^2}

But for :math:`\nu` we have the expression:

.. math::
   :label: eq180

   \nu = \frac{\nu_0}{(1+\frac{Z}{c})}

so we end up with:

.. math::
   :label: eq190

   dZ = \frac{-c}{\nu_0}\ {\bigl(1+\frac{Z}{c}\bigr)}^2\ d\nu
   
With :math:`d\nu = \Delta \nu_b` and the given barycentric velocity
:math:`Z_b` = 9120000 m/s,
this gives an increment in optical velocity of:

.. math::
   :label: eq200

   dZ_b = \frac{-299792458.0}{1420405752.0}\ {\bigl(1+\frac{9120000.0}{299792458.0}\bigr)}^2\ 97647.745732 =-21882.651\ m/s

With these values we explained some other alternate header 
keywords in the basic spectral-line example::

   CDELT3Z= -2.1882651e+4  / [m/s]
   SPECSYSZ= 'BARYCENT'    / Velocities w.r.t. barycenter
   SSYSOBSZ= 'TOPOCENT'    / Observation was made from the 'TOPOCENT' frame


Barycentric/lsrk radio velocity
--------------------------------

For radio velocities one needs to apply the definition:

.. math::
   :label: eq210

   V_{radio} = V = c\ \bigl(\frac{\nu_0 - \nu}{\nu_0}\bigr)
   
and for the shifted frequency we derive from this equation:

.. math::
   :label: eq220

   \nu = \nu_0\  \bigl(1-\frac{V}{c}\bigr)
   
and the spectral translation code becomes:
`proj.spectra('VRAD')`

In the next code example we demonstrate for a barycentric radio velocity
*V* = 8850.750904 km/s how to calculate the barycentric velocities at arbitrary pixels.
This velocity is derived from the optical example in a way that shifted frequency and 
topocentric correction are the same. One can use the formula

.. math::
   :label: eq225

   \frac{V_b}{Z_b} = \frac{\nu_b}{\nu_0} 

to find the value of :math:`V_b = 1.37847121643*9120/1.420405752 = 8850.750904` km/s (with the frequencies in GHz and the velocity in km/s).
In a next section we will derive this value in another way see :eq:`eq230` and :eq:`eq240`
::

      
   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy as n
   
   c = 299792458.0       # Speed of light (m/s)
   f = 1.37835117405e9   # Topocentric reference frequency (Hz)
   df = 9.765625e4       # Topocentric frequency increment (Hz)
   f0 = 1.420405752e+9   # Rest frequency (Hz)
   V = 8850750.904       # Barycentric radio velocity (m/s)
   
   fb = f0*(1-V/c)
   print "Barycentric freq.: ", fb
   v = c * ((fb*fb-f*f)/(fb*fb+f*f))
   print "VELOSYSR= Topocentric correction:", v, "m/s"
   dfb = df*(c-v)/n.sqrt(c*c-v*v)
   print "CDELT3F= Delta in frequency in the barycentric frame eq.4): ", dfb
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : fb,
              'CRPIX1' : 32,
              'CUNIT1' : 'Hz',
              'CDELT1' : dfb,
              'RESTFRQ': 1.420405752e+9
         }
   line = wcs.Projection(header).spectra('VRAD')
   pixels =  range(30,35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000

   # Output:
   # Barycentric freq.:  1378471216.43
   # VELOSYSR= Topocentric correction: 26108.1745986 m/s
   # CDELT3F= Delta in frequency in the barycentric frame eq.4):  97647.745732
   #
   # Output Radio velocities (km/s)
   # 30 8891.97019316
   # 31 8871.36054858
   # 32 8850.750904
   # 33 8830.14125942
   # 34 8809.53161484


Frequency to Radio velocity
-----------------------------

From the definition of radio velocity:

.. math::
   :label: eq230

   V = c\ \bigl(\frac{\nu_0 - \nu}{\nu_0}\bigr)

we can find a radio velocity that corresponds to the value of the optical
velocity. This (barycentric) optical velocity (9120 Km/s) caused a shift of the rest frequency.
The new frequency became :math:`\nu_b` = 1.37847122\times 10^9 Hz.
If we insert this number in the equation above we find:

.. math::
   :label: eq240

   V_b = c\ \bigl(\frac{1420405752.0 - 1378471216.43}{1420405752.0}\bigr) = 8850750.90419\ m/s
   
The formula for a direct conversion from optical to radio velocity can be derived to 
insert the formula for the frequency shift corresponding to optical velocity, into
the expression for the radio velocity:

.. math::
   :label: eq250

   V = c\ \bigl(1 - \frac{1}{1+\frac{Z}{c}}\bigr)

With eq. :eq:`eq230` it is easy to find the increment of the velocity if the increment in frequency
at the reference frequency is given:

.. math::
   :label: eq260

   dV = \frac{-c}{\nu_0}\ d\nu

Note that this increment in frequency is the **increment in the barycentric system**!

Inserting the numbers with :math:`d\nu = \Delta \nu_b` we find:

.. math::
   :label: eq270

   dV_b = \frac{-299792458.0}{1420405752.0}\times 97647.7457312 = -20609.644582\ m/s

This gives us another two values for the alternate header keywords::

   CTYPE3R= 'VRAD'
   CRVAL3R= 8.85075090419e+6  / [m/s]
   CDELT3R= -2.0609645e+4     / [m/s]

Note that *CTYPE3R= 'VRAD'* indicates that the conversion between frequency and radio velocity
is linear.

The next script shows how we can use these new header values to get a list of 
radio velocities as function of pixel. We commented out the rest frequency. Its value
is not necessary because we can rewrite the formulas for the velocity in terms of
:math:`\nu/\nu_0` and :math:`\Delta \nu/\nu_0`
::

   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VRAD',
              'CRVAL1' : 8850750.904193053,
              'CRPIX1' : 32,
              'CUNIT1' : 'm/s',
              'CDELT1' : -20609.644582145629,
   #          'RESTFRQ': 1.420405752e+9
            }
   line = wcs.Projection(header)
   pixels =  range(30,35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   #
   # Output barycentric radio velocity in km/s:
   # 30 8891.97019336
   # 31 8871.36054878
   # 32 8850.75090419
   # 33 8830.14125961
   # 34 8809.53161503

Alternatively use the spectral translation method `spectra()`
with the values of the barycentric frequency and frequency increment
as follows to get (exactly) the same output::

   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : 1378471216.4292786,
              'CRPIX1' : 32,
              'CUNIT1' : 'Hz',
              'CDELT1' : 97647.745732,
              'RESTFRQ': 1.420405752e+9
            }
   line = wcs.Projection(header).spectra('VRAD')
   pixels =  range(30,35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   #
   # Output barycentric radio velocity in km/s:
   # 30 8891.97019336
   # 31 8871.36054878
   # 32 8850.75090419
   # 33 8830.14125961
   # 34 8809.53161503


Frequency to Apparent radial velocity
-------------------------------------

As written before, the relation between a true velocity and a shifted frequency is:

.. math::
   :label: eq300

   v = c\ \frac{\nu_0^2-\nu^2}{\nu_0^2+\nu^2}

Observed from the barycenter the source has a radial velocity:

.. math::
   :label: eq310

   v_b = 299792458.0 \ \frac{1420405752.0^2-1378471216.42927^2}{1420405752.0^2+1378471216.42927^2} = 8981342.29811\ m/s
   
::

   CTYPE3V= 'VELO-F2V'
   CRVAL3V= 8.98134229811e+6 / [m/s]

Note that *CTYPE3V= 'VELO-F2V'* indicates that we derived these velocities from a system in which
the frequency is linear with the pixel value.


For the increment of the apparent radial velocity we need to find the derivative of eq. :eq:`eq40`

.. math::
   :label: eq320

   \frac{dv}{d\nu} = c(\nu_0^2-\nu^2)\frac{d}{d\nu}{(\nu_0^2+\nu^2)}^{-1} + c{(\nu_0^2+\nu^2)}^{-1}\frac{d}{d\nu}(\nu_0^2-\nu^2)
   
This works out as:

.. math::
   :label: eq330

   dv = \frac{-4 c \nu \nu_0^2}{{(\nu_0^2+\nu^2)}^2}\ d\nu
   
and with the appropriate numbers inserted for :math:`d\nu = \Delta \nu_b`

and :math:`\nu = \nu_b`:

.. math::
   :label: eq340

   dv_b = \frac{-4 \times 299792458.0\times  1378471216.4292786\times 1420405752.0^2}{{(1420405752.0^2+1378471216.4292786^2)}^2}\  97647.745732 = -21217.55136
   
which reveals the value of another keyword from the header in the article's example::

   CDELT3V= -2.1217551e+4  / [m/s]

Sometimes you might encounter an alternative formula that doesn't list the frequency.
It uses eq. :eq:`eq30` to express the frequency in terms of the radial velocity and the rest
frequency.

.. math::
   :label: eq350

   \nu = \nu_0\sqrt{\frac{1-v/c}{1+v/c}}
   
If you insert this into:

.. math::
   :label: eq360

   dv = \frac{-4 c \nu \nu_0^2}{{(\nu_0^2+\nu^2)}^2}\ d\nu
   
then after some calculations you end up with the expression:

.. math::
   :label: eq370

   dv = \frac{-c}{\nu_0}\  \sqrt{(1-\frac{v}{c})}\ {(1+\frac{v}{c})}^{\frac{3}{2}}\  d\nu 

If you insert v = 8981342.29811 (m/s) in this expression you will get exactly the same
radial velocity increment (-2.1217551e+4 m/s).


We found a true radial velocity and 
calculated the increment for this radial velocity. With a short script and
a minimal header we demonstrate how to use WCSLIB to get
a radial velocity for an arbitrary pixel::

         
   #!/usr/bin/env python
   from kapteyn import wcs
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VELO-F2V',
              'CRVAL1' : 8981342.2981121931,
              'CRPIX1' : 32,
              'CUNIT1' : 'm/s',
              'CDELT1' : -21217.5513673598,
              'RESTFRQ': 1.420405752e+9
            }
   line = wcs.Projection(header)
   pixels = range(30,35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   # Output:
   # 30 9023.78022672
   # 31 9002.56055595
   # 32 8981.34229811
   # 33 8960.12545322
   # 34 8938.9100213

How can this work?
From eq. :eq:`eq350` and eq. :eq:`eq360` it is obvious that WCSLIB can calculate the reference frequency from
the reference radial velocity. For this reference frequency and the increment
in radial velocity it can calculate the increment in frequency at this reference frequency.
Then we have all the information to use eq. :eq:`eq350` to calculate radial
velocities for different frequencies (i.e. different pixels). Note that the step in 
frequency is linear and the step in radial velocity is **not** (which explains the
extension 'F2V' in the CTYPE keyword).

Next script and header is an alternative to get exactly the same results. The header lists the barycentric 
frequency and frequency increment. We need a spectral translation with 
method `spectra()` to tell WCSLIB to calculate radial velocities::


   #!/usr/bin/env python
   from kapteyn import wcs
   header  = { 'NAXIS'  : 1,
               'CTYPE1' : 'FREQ',
               'CRVAL1' : 1378471216.4292786,
               'CRPIX1' : 32,
               'CUNIT1' : 'Hz',
               'CDELT1' : 97647.745732,
               'RESTFRQ': 1.420405752e+9
            }
   line = wcs.Projection(header).spectra('VELO-F2V')
   pixels = range(30,35)
   Vwcs = line.toworld1d(pixels)
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   # Output:
   # 30 9023.78022672
   # 31 9002.56055595
   # 32 8981.34229811
   # 33 8960.12545322
   # 34 8938.9100213


Frequency to Wavelength
------------------------

The rest wavelength is given by the relation:

.. math::
   :label: eq380
   
   \lambda_0 = \frac{c}{\nu_0}

Inserting the right numbers we find:

.. math::
   :label: eq390

   \lambda_0 = \frac{299792458.0}{1420405752.0} = 0.211061140507\ m


For the barycentric wavelength we need to insert the barycentric frequency.

.. math::
   :label: eq400

   \lambda = \frac{299792458.0}{1378471216.43} = 0.217481841062\ m

The increment in wavelength as function of the increment in 
(barycentric) frequency is:

.. math::
   :label: eq410

   d\lambda = \frac{-c}{\nu^2} d\nu
   
With the right numbers:

.. math::
   :label: eq420

   d\lambda = \frac{-299792458.0}{1378471216.43^2}\ 97647.745732 = -1.54059158176\times 10^{-5}\ m
 
This gives us the alternate header keywords::

   RESTWAVZ= 0.211061140507  / [m]

::

   CTYPE3W= 'WAVE-F2W'
   CRVAL3W=  0.217481841062  / [m]
   CDELT3W= -1.5405916e-05   / [m]
   CUNIT3W=  'm'
   RESTWAVW= 0.211061140507  / [m]

Note that CTYPE indicates that there is a non linear conversion from frequency
to wavelength.

From the standard definition of optical velocity:

.. math::
   :label: eq430

   Z = c\ \frac{\lambda-\lambda_0}{\lambda_0}


it follows that the increment in optical velocity as function of increment of wavelength
is given by:

.. math::
   :label: eq440

    dZ = \frac{c}{\lambda_0}\ d\lambda

Then with the numbers we find:

.. math::
   :label: eq450

   dZ_b = \frac{299792458.0}{0.211061140507}\times -1.54059158176\times 10^{-5}= -21882.6514422\ m/s
   
which is the increment in optical velocity earlier given for CDELT3Z.

This is one of the possible conversions between wavelength and velocity. Others are listed 
in `scs.pdf <scs.pdf>`_ table 3 of E.W. Greisen et al. page 750.


Conclusions
-----------

* Note that the inertial system is set by a (FITS) header using a special keyword 
  (e.g. VELREF=) or it is coded in the CTYPEn keyword. It doesn't change anything in
  the calculations above. Conversions between inertial reference systems is not possible because
  headers do (usually) not contain the relevant information to calculate the topocentric
  correction w.r.t. that system (one needs time of observation, position of observatory 
  and position of the observed source).

* From a header with CTYPEn='FREQ' we can derive optical, radio and apparent radial velocities
  with method *spectra()*:
   
   * *proj = wcs.Projection(header).spectra('VOPT-F2W')*
   * *proj = wcs.Projection(header).spectra('VRAD')*
   * *proj = wcs.Projection(header).spectra('VELO-F2V')*
   
   This applies also to alternate axis descriptions. So if CTYPE1='VRAD' one can derive 
   one of the other velocity definitions by adding the *spectra()* method with
   the appropriate argument.

   Here is an example::

      #!/usr/bin/env python
      from kapteyn import wcs
      wcs.debug = True
      header = { 'NAXIS'  : 1,
               'CTYPE1' : 'VRAD',
               'CRVAL1' : 8850750.904193053,
               'CRPIX1' : 32,
               'CUNIT1' : 'm/s',
               'CDELT1' : -20609.644582145629,
               'RESTFRQ': 1.420405752e+9
               }
      line = wcs.Projection(header).spectra('VOPT-F2W')
      pixels = range(30,35)
      Vwcs = line.toworld1d(pixels)
      for p,v in zip(pixels, Vwcs):
         print p, v/1000
      # Output:
      # Velocities in km/s converted from 'VRAD' to 'VOPT-F2W'
      # 30 9163.77150423
      # 31 9141.88420167
      # 32 9120.0
      # 33 9098.11889856
      # 34 9076.2408967

   Note that the rest frequency is required now.

   Note also that we added statement *wcs.debug = True* to get some debug
   information from WCSLIB.

* Axis types 'FREQ-HEL' and 'FREQ-LSR' (AIPS definitions) are recognized by WCSLIB
  and are treated as 'FREQ'. No conversions are done. Internally the keyword *SPECSYS=*
  gets a value.


The complete alternate axis descriptions
-----------------------------------------

In this section we summarize the alternate axis descriptions and we add
a small script that proves that these descriptions are consistent::

   CNAME=  'Topocentric Frequency. Basic header'
   CTYPE3= 'FREQ'
   CRVAL3=  1.37835117405e9
   CDELT3=  9.765625e4
   CRPIX3=  32
   CUNIT3= 'Hz'
   RESTFRQ= 1.420405752e+9
   SPECSYS='TOPOCENT'

   CNAME3Z= 'Barycentric optical velocity'
   RESTWAVZ= 0.211061140507   / [m]
   CTYPE3Z= 'VOPT-F2W'
   CRVAL3Z=  9.120e+6         / [m/s]
   CDELT3Z= -2.1882651e+4     / [m/s]
   CRPIX3Z=  32
   CUNIT3Z= 'm/s'
   SPECSYSZ='BARYCENT'        / Velocities w.r.t. barycenter
   SSYSOBSZ='TOPOCENT'        / Observation was made from the 'TOPOCENT' frame
   VELOSYSZ= 26108            / [m/s]
   
   CNAME3F= 'Barycentric frequency'
   CTYPE3F= 'FREQ'
   CRVAL3F=  1.37847121643e+9 / [Hz]
   CDELT3F=  9.764775e+4      / [Hz]
   CRPIX3F=  32
   CUNIT3F= 'Hz'
   RESTFRQF= 1.420405752e+9
   SPECSYSF='BARYCENT'
   SSYSOBSF='TOPOCENT'
   VELOSYSF= 26108            / [m/s]
   
   CNAME3R= 'Barycentric radio velocity'
   CTYPE3R= 'VRAD'
   CRVAL3R=  8.85075090419e+6 / [m/s]
   CDELT3R= -2.0609645e+4     / [m/s]
   CRPIX3R=  32
   CUNIT3R= 'm/s'
   RESTFRQR= 1.420405752e+9
   SPECSYSR='BARYCENT'
   SSYSOBSR='TOPOCENT'
   VELOSYSR= 26108            / [m/s]
   
   CNAME3V= 'Barycentric apparent radial velocity'
   RESTFRQV= 1.420405752e+9   / [Hz]
   CTYPE3V= 'VELO-F2V'
   CRVAL3V=  8.98134229811e+6 / [m/s]
   CDELT3V= -2.1217551e+4     / [m/s]
   CRPIX3V=  32
   CUNIT3V= 'm/s'
   SPECSYSV='BARYCENT'
   SSYSOBSV='TOPOCENT'
   VELOSYSV= 26108            / [m/s]

   CNAME3W= 'Barycentric wavelength'
   CTYPE3W= 'WAVE-F2W'
   CRVAL3W=  0.217481841062   / [m]
   CDELT3W= -1.5405916e-05    / [m]
   CRPIX3W=  32
   CUNIT3W= 'm'
   RESTWAVW=  0.211061140507  / [m]
   SPECSYSW='BARYCENT'
   SSYSOBSW='TOPOCENT'
   VELOSYSW= 26108            / [m/s]
   

To check the validity and completeness of these alternate axis descriptions, 
we wrote a small script that loops over all the mnemonic letter codes in a header that
is composed from the header fragments above. We only changed axisnumber 3 to 1.
The output is the same within the boundaries of the given precision of the numbers.
To change the axis description in a header we use the *alter* parameter
when we create the projection object.
 
Parameter *alter* is an optional
letter from 'A' through 'Z', indicating an alternative WCS axis description::

   #!/usr/bin/env python
   from kapteyn import wcs
   header = {  'NAXIS'    :  1,
               'CTYPE1'   : 'FREQ',
               'CRVAL1'   :  1378471216.4292786,
               'CRPIX1'   :  32,
               'CUNIT1'   : 'Hz',
               'CDELT1'   :  97647.745732,
               'RESTFRQ'  :  1.420405752e+9,
               'CNAME1Z'  : 'Barycentric optical velocity',
               'RESTWAVZ' :  0.211061140507,   # [m]
               'CTYPE1Z'  : 'VOPT-F2W',
               'CRVAL1Z'  :  9.120e+6,         # [m/s]
               'CDELT1Z'  : -2.1882651e+4,     # [m/s]
               'CRPIX1Z'  :  32,
               'CUNIT1Z'  : 'm/s',
               'SPECSYSZ' : 'BARYCENT',        # Velocities w.r.t. barycenter,
               'SSYSOBSZ' : 'TOPOCENT',        # Observation was made from the 'TOPOCENT' frame,
               'VELOSYSZ' :  26108,            # [m/s]
               'CNAME1F'  : 'Barycentric frequency',
               'CTYPE1F'  : 'FREQ',
               'CRVAL1F'  :  1.37847121643e+9, # [Hz]
               'CDELT1F'  :  9.764775e+4,      # [Hz]
               'CRPIX1F'  :  32,
               'CUNIT1F'  : 'Hz',
               'RESTFRQF' :  1.420405752e+9,
               'SPECSYSF' : 'BARYCENT',
               'SSYSOBSF' : 'TOPOCENT',
               'VELOSYSF' :  26108,            # [m/s]
               'CNAME1W'  : 'Barycentric wavelength',
               'CTYPE1W'  : 'WAVE-F2W',
               'CRVAL1W'  :  0.217481841062,   # [m]
               'CDELT1W'  : -1.5405916e-05,    # [m]
               'CRPIX1W'  :  32,
               'CUNIT1W'  : 'm',
               'RESTWAVW' :  0.211061140507,   # [m]
               'SPECSYSW' : 'BARYCENT',
               'SSYSOBSW' : 'TOPOCENT',
               'VELOSYSW' :  26108,            # [m/s]
               'CNAME1R'  : 'Barycentric radio velocity',
               'CTYPE1R'  : 'VRAD',
               'CRVAL1R'  :  8.85075090419e+6, # [m/s]
               'CDELT1R'  : -2.0609645e+4,     # [m/s]
               'CRPIX1R'  :  32,
               'CUNIT1R'  : 'm/s',
               'RESTFRQR' :  1.420405752e+9,
               'SPECSYSR' : 'BARYCENT',
               'SSYSOBSR' : 'TOPOCENT',
               'VELOSYSR' :  26108,            # [m/s]
               'CNAME1V'  : 'Barycentric apparent radial velocity',
               'CTYPE1V'  : 'VELO-F2V',
               'CRVAL1V'  :  8.98134229811e+6, # [m/s]
               'CDELT1V'  : -2.1217551e+4,     # [m/s]
               'CRPIX1V'  :  32,
               'CUNIT1V'  : 'm/s',
               'RESTFRQV' :  1.420405752e+9,   # [Hz]
               'SPECSYSV' : 'BARYCENT',
               'SSYSOBSV' : 'TOPOCENT',
               'VELOSYSV' :  26108             # [m/s]
            }
   
   # Loop over all the alternative headers
   for alt in ['F', 'Z', 'W', 'R', 'V']:
      spec = wcs.Projection(header, alter=alt).spectra('VOPT-F2W')
      pixels = range(30,35)
      Vwcs = spec.toworld1d(pixels)
      cname = header['CNAME1'+alt]             # Just a header text
      print "VOPT-F2W from %s" % (cname,)
      print "Pixel, velocity (%s)" % spec.units
      for p,v in zip(pixels, Vwcs):
         print p, v/1000.0
   # Output
   # VOPT-F2W from Barycentric frequency
   # Pixel, velocity (m/s)
   # 30 9163.77150598
   # 31 9141.88420246
   # 32 9119.99999984
   # 33 9098.11889745
   # 34 9076.24089463
   # VOPT-F2W from Barycentric optical velocity
   # Pixel, velocity (m/s)
   # 30 9163.77150335
   # 31 9141.88420123
   # 32 9120.0
   # 33 9098.11889901
   # 34 9076.24089759
   # VOPT-F2W from Barycentric wavelength
   # Pixel, velocity (m/s)
   # 30 9163.77150495
   # 31 9141.88420213
   # 32 9120.0000002
   # 33 9098.1188985
   # 34 9076.24089638
   # VOPT-F2W from Barycentric radio velocity
   # Pixel, velocity (m/s)
   # 30 9163.77150512
   # 31 9141.88420211
   # 32 9120.0
   # 33 9098.11889812
   # 34 9076.24089581
   # VOPT-F2W from Barycentric apparent radial velocity
   # Pixel, velocity (m/s)
   # 30 9163.77150347
   # 31 9141.88420129
   # 32 9120.0
   # 33 9098.11889894
   # 34 9076.24089746


Alternative conversions
+++++++++++++++++++++++++

Conversion between radio and optical velocity
-----------------------------------------------

In the next two section we give some formula's that could be handy if you want to verify
numbers. They are not used in WCSLIB.

With the definitions for radio and optical velocity it is easy to derive:

.. math::
   :label: eq500

   \frac{V}{Z} = \frac{\nu}{\nu_0}
   
This can be verified with:
   
   * Z = 9120000.00000 m/s
   * V = 8850750.90419 m/s
   * :math:`\nu_0` = 1420405752.00 Hz
   * :math:`\nu_b` = 1378471216.43 Hz

Both ratio's are equal to 1.030421045482.

Conversion between radial velocity and optical/radio velocity
---------------------------------------------------------------

It is possible to find a relation between the true velocity and the optical velocity
using eq. :eq:`eq10` and eq. :eq:`eq50`.
The radial velocity can be written as:

.. math::
   :label: eq510

   \frac{v}{c} = \frac{\frac{\nu_0^2}{\nu^2}-1}{\frac{\nu_0^2}{\nu^2}+1}
   
The frequency shift for an optical velocity is:

.. math::
   :label: eq520

   \frac{\nu_0}{\nu} = \bigl(1+\frac{Z}{c}\bigr)

Then:

.. math::
   :label: eq530

   \frac{v}{c} = \frac{{(1+Z/c)}^2-1}{{(1+Z/c)}^2+1} = \frac{Z^2+2cZ}{Z^2+2cZ+2c^2}
   
This equation is used in AIPS memo 27 [Aipsmemo]_ to relate an optical velocity to a true radial velocity.
If we insert :math:`Z_b` = 9120000 (m/s) then we find :math:`v_b` = 8981342.29811 (m/s) as expected
(eq. :eq:`eq50`, :eq:`eq310`)

For radio velocities we find in a similar way:

.. math::
   :label: eq540

   \frac{\nu_0}{\nu} = \frac{1}{\bigl(1-\frac{V}{c}\bigr)}
   
which gives the relation between radial velocity and radio velocity:

.. math::
   :label: eq550

   \frac{v}{c} = \frac{2cV-V^2}{V^2-2cV+2c^2}
   
If we substitute the calculated barycentric radio velocity :math:`V_b` = 8850750.90419 (m/s)
then one finds again: :math:`v_b` = 8981342.29811 (m/s) (see also (eq. :eq:`eq50`, :eq:`eq310`)
Note that the last formula is equation 4 in AIPS memo 27 [Aipsmemo]_
Non-Linear Coordinate Systems in AIPS. However that formula lacks a minus sign
in the nominator and therefore does not give a correct result.

Legacy headers
++++++++++++++


.. _spectral_gipsy:

A recipe for modification of Nmap/GIPSY FITS data
--------------------------------------------------

For FITS headers produced by Nmap/GIPSY we don't have an increment
in velocity available so we cannot use them as input for WCSLIB (otherwise we
would treat them like the FELO axis recognized by AIPS). The Python interface to
WCSLIB applies a conversion for these headers before they are
processed by WCSLIB. From the previous steps we can summarize how
the data in the Nmap/GIPSY FITS header is changed:

   * The extension in CTYPEn is '-OHEL', '-OLSR', 'RHEL' or 'RLSR'
   * The velocity is retrieved from FITS keyword VELR= (always in m/s) or DRVALn= (in units of DUNITn)
   * Convert reference frequency to a frequency in Hz.
   * Calculate the reference frequency in the barycentric system using eq. :eq:`eq10`
     if the velocity is optical and eq. :eq:`eq220` if the velocity is a radio velocity.
   * Calculate the topocentric velocity using eq. :eq:`eq100`
   * Convert frequency increment to an increment in Hz
   * Calculate the increment in frequency in the selected reference system (HEL, LSR)
   * Change CRVALn and CDELTn to the barycentric values
   * Create a projection object with spectral translation, e.g. **proj.spectra('VOPT-F2W')**

The Python interface allows for an easy implementation for these special exceptions.
Here is a script that uses the new facility. The conversion here is triggered by the CTYPE
extension **OHEL**. So as long this is unique to GIPSY spectral axes, you are save to
use it. Note that we converted the frequencies to optical, radio and radial velocities.
This is added value to the existing GIPSY implementation where these conversions are not
possible. These WCSLIB conversions are explained in previous sections ::


   #!/usr/bin/env python
   from kapteyn import wcs
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ-OHEL',
              'CRVAL1' : 1.37835117405e9,
              'CRPIX1' : 32,
              'CUNIT1' : 'Hz',
              'CDELT1' : 9.765625e4,
              'RESTFRQ': 1.420405752e+9,
              'DRVAL1' : 9120000.0,
   #          'VELR'   : 9120000.0
              'DUNIT1' : 'm/s'
            }
   proj = wcs.Projection(header)
   pixels = range(30,35)
   
   voptical = proj.spectra('VOPT-F2W')
   Vwcs = voptical.toworld1d(pixels)
   print "\nPixel, optical velocity (%s)" % voptical.units
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0
   
   vradio = proj.spectra('VRAD')
   Vwcs = vradio.toworld1d(pixels)
   print "\nPixel, radio velocity (%s)" % vradio.units
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0
   
   vradial = proj.spectra('VELO-F2V')
   Vwcs = vradial.toworld1d(pixels)
   print "\nPixel, apparent radial velocity (%s)" % vradial.units
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0

   # Output:
   # Pixel, optical velocity (m/s)
   # 30 9163.77150423
   # 31 9141.88420167
   # 32 9120.0
   # 33 9098.11889856
   # 34 9076.2408967
   #
   # Pixel, radio velocity (m/s)
   # 30 8891.97019336
   # 31 8871.36054878
   # 32 8850.75090419
   # 33 8830.14125961
   # 34 8809.53161503
   #
   # Pixel, apparent radial velocity (m/s)
   # 30 9023.78022672
   # 31 9002.56055595
   # 32 8981.34229811
   # 33 8960.12545322
   # 34 8938.9100213


.. note::

   Comment: Note that changing DRVAL1 to VELR gives the same output. Both are recognized as keywords
   that store a velocity. The value in VELR should always be in m/s.
   Note also how we created different sub-projections (one for each type of velocity)
   from the same main projection. All these objects can coexist.
 
AIPS axis type FELO
--------------------
Next script and output shows that with the optical reference velocity and the corresponding
increment in velocity (CDELT3Z), we can get velocities without spectral translation. 
WCSLIB recognizes the axis type 'FELO' which is regularly
gridded in frequency but expressed in velocity units in the optical convention. It is therefore not a
surprise that the output is the same as the list with optical velocities in one of the previous
scripts. 

So if we have a barycentric reference velocity and a barycentric velocity increment, then
according to the formulas above it is easy to retrieve the values for the barycentric 
reference frequency and the barycentric frequency increment::

   #!/usr/bin/env python
   from kapteyn import wcs
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FELO-HEL',
              'CRVAL1' : 9120,
              'CRPIX1' : 32,
              'CUNIT1' : 'km/s',
              'CDELT1' : -21.882651442,
              'RESTFRQ': 1.420405752e+9
            }
   proj = wcs.Projection(header)
   pixels = range(30,35)
   Vwcs = proj.toworld1d(pixels)
   print "Pixel, velocity (%s)" % "km/s"
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0

   # FELO-HEL is equivalent to VOPT-F2W
   header['CTYPE1'] = 'VOPT-F2W'
   proj = wcs.Projection(header)
   pixels = range(30,35)
   Vwcs = proj.toworld1d(pixels)
   print "Pixel, velocity (%s)" % "km/s"
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0

   header['CTYPE1'] = 'FELO'
   proj = wcs.Projection(header)
   pixels = range(30,35)
   Vwcs = proj.toworld1d(pixels)
   print "Pixel, velocity (%s)" % proj.units
   Zr = header['CRVAL1']
   dZr = header['CDELT1']
   refpix = header['CRPIX1']
   for p,v in zip(pixels, Vwcs):
      print p, v/1000.0, Zr + (p-refpix) * dZr
   #
   # Output:
   # With CTYPE='FELO-HEL': Pixel, velocity (km/s)
   # 30 9163.77150423
   # 31 9141.88420167
   # 32 9120.0
   # 33 9098.11889857
   # 34 9076.24089671
   #
   # With CTYPE='VOPT-F2W': Pixel, velocity (km/s)
   # 30 9163.77150423
   # 31 9141.88420167
   # 32 9120.0
   # 33 9098.11889857
   # 34 9076.24089671
   #
   # With CTYPE='FELO' (i.e. Linear)
   # 30 9.16376530288 9163.76530288
   # 31 9.14188265144 9141.88265144
   # 32 9.12 9120.0
   # 33 9.09811734856 9098.11734856
   # 34 9.07623469712 9076.23469712

Despite the units in the header (km/s) we changed the output from m/s to km/s
because the output is always
in m/s (check attribute *proj.units*) so we divide it by 1000 to get km/s.

In this script we demonstrate the use of a special axis type which originates from classic AIPS.
It is called 'FELO'. WCSLIB (and not our Python interface) recognizes this type as an
**optical velocity** and performs
the necessary internal conversions)::

   if (strcmp(wcs->ctype[i], "FELO") == 0) {
      strcpy(wcs->ctype[i], "VOPT-F2W");

and the extensions are translated into values for FITS keyword *SPECSYS*::
       
   if (strcmp(scode, "-LSR") == 0) {
      strcpy(wcs->specsys, "LSRK");
   } else if (strcmp(scode, "-HEL") == 0) {
      strcpy(wcs->specsys, "BARYCENT");
   } else if (strcmp(scode, "-OBS") == 0) {
      strcpy(wcs->specsys, "TOPOCENT");


**Conclusions**


    * The extension HEL or LSR after FELO in *CTYPE1* is not used in the calculations. But when you omit
      a valid extension the axis will be treated as a linear axis.
    * In the example above one can replace *FELO-HEL* in *CTYPE1*
      by FITS standard *VOPT-F2W* showing that FELO is in fact the same as *VOPT-F2W*.


AIPS axis type VELO
-------------------

An AIPS VELO axis (e.g. CTYPE3='VELO-HEL') is an axis that is regularly gridded in velocity.
It is recognized by WCSLIB and is treated as a radio velocity. The increment in
velocity is constant.
It allows spectral translations like:
*proj.spectra('VOPT-V2W')*, *proj.spectra('VRAD-V2F')* and *proj.spectra('FREQ-V2F')*.

This makes sense because type 'VELO' without extension tells us that the increment in velocity
is constant::

   #!/usr/bin/env python
   from kapteyn import wcs
   import math
   
   V0 = -.24300000000000E+06             # Radio vel in m/s
   dV = 5000.0                           # Delta in m/s
   f0 = 0.14204057583700e+10
   c  = 299792458.0                      # Speed of light in m/s
   crpix = 32
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VELO-HEL',
              'CRVAL1' : V0,
              'CRPIX1' : crpix,
              'CUNIT1' : 'm/s',
              'CDELT1' : dV,
              'RESTFRQ': f0
            }
   
   proj = wcs.Projection(header)
   V1 = proj.spectra('VOPT-V2W')
   
   pixels = range(30,35)
   V = proj.toworld1d(pixels)
   Z = V1.toworld1d(pixels)
   print "Pixel Vradial (or Vradio) in (km/s)  and Voptical (km/s)"
   for p,v,z in zip(pixels, V, Z):
      print "%4d %15f %15f" % (p, v/1000, z/1000)

   print "\nOptical velocities calculated from radial velocity with constant increment"
   v0 = V0
   for i in pixels:
      v1 = v0 + (i-crpix)*dV
      beta = v1/c
      frac = (1-beta)/(1+beta)
      f = f0 * math.sqrt(frac)
      Z = c* (f0-f)/f
      print "%4d %15f" % (i, Z/1000.0)
   
   header['CTYPE1'] = 'VELO-F2V'
   proj = wcs.Projection(header)
   pixels = range(30,35)
   V = proj.toworld1d(pixels)
   print "\nWith CTYPE='VELO-F2V': Pixel Vradio (km/s)"
   for p,v in zip(pixels, V):
      print "%4d %15f" % (p, v/1000)
   
   header['CTYPE1'] = 'VELO'
   proj = wcs.Projection(header)
   pixels = range(30,35)
   V = proj.toworld1d(pixels)
   print "\nWith CTYPE='VELO': Pixel Vradio (km/s)"
   for p,v in zip(pixels, V):
      print "%4d %15f" % (p, v/1000)

   # Output:
   # Pixel Vradial (or Vradio) in (km/s)  and Voptical (km/s)
   # 30     -253.000000     -252.893335
   # 31     -248.000000     -247.897507
   # 32     -243.000000     -242.901597
   # 33     -238.000000     -237.905603
   # 34     -233.000000     -232.909526
   #
   # Optical velocities calculated from radial velocity with constant increment
   # 30     -252.893335
   # 31     -247.897507
   # 32     -242.901597
   # 33     -237.905603
   # 34     -232.909526
   # 
   # With CTYPE='VELO-F2V': Pixel , v (km/s)
   # 30     -252.999833
   # 31     -247.999958
   # 32     -243.000000
   # 33     -237.999958
   # 34     -232.999833
   #
   # With CTYPE='VELO': Pixel , v (km/s)
   # 30     -253.000000
   # 31     -248.000000
   # 32     -243.000000
   # 33     -238.000000
   # 34     -233.000000

   
We used eq. :eq:`eq30` to calculate a frequency for a given radial velocity. This frequency
is used in eq. :eq:`eq5` to calculate the optical velocity. The script proves:

       * Axis VELO-HEL is processed as if it was a linear axis
       * Both the spectral translation *proj.spectra('VOPT-V2W')* as
         the formulas :eq:`eq30` and :eq:`eq5` give the same optical
         velocities.
       * A VELO axis followed by a valid code for a conversion algorithm,
         is a radial velocity where a constant frequency increment is assumed.
       * A VELO axis without a code for a conversion algorithm is processed
         as if it was a linear axis.

To summarize: 

  * **without extension** VELO is interpreted by WCSLIB as a radial velocity.
    The velocity increment is constant.
  * if the VELO type has extension HEL or LSR then WCSLIB interprets it as
    a linear transformation. The velocity is (usually) a radio
    velocity. In the AIPS cookbook we read:

      * 'VELO' - velocity (m/s) (radio convention, unless overridden by use of the VELDEF SHARED keyword) 

    VELDEF can either be 'radio' or 'optical'.
    
  * a VELO axis without extension represents a radial velocity with constant velocity
    increment. The table below shows possible conversions between VELO axes and
    other spectral axes.


Conversions for type VELO:

=======================  =====================================================================
CTYPE                    Can be converted with spectral translations
=======================  =====================================================================
VELO-HEL, VELO-LSR       linear transformation, but can be converted as a VELO axis
VELO                     WAVE-V2W  or FREQ-V2F or VOPT-V2F or VRAD-V2F
VELO-F2V                 WAVE-F2W  or FREQ or VOPT-F2W or VRAD
VELO-W2V                 WAVE  or FREQ-W2F or VOPT or VRAD-W2F
=======================  =====================================================================



.. note::

      From the WCSLIB API documentation:
      
      AIPS-convention celestial projection types, NCP and GLS, and spectral types,
      '{FREQ,FELO,VELO}-{OBS,HEL,LSR}' as in 
      'FREQ-LSR', 'FELO-HEL', etc., set in CTYPEia are translated on-the-fly by
      wcsset() but without modifying the relevant ctype[], pv[] or specsys members
      of the wcsprm struct. That is, only the information extracted from ctype[]
      is translated when wcsset() fills in wcsprm::cel (celprm struct) or wcsprm::spc (spcprm struct).

      On the other hand, these routines do change the values of wcsprm::ctype[],
      wcsprm::pv[], wcsprm::specsys and other wcsprm struct members as appropriate to
      produce the same result as if the FITS header itself had been translated.


Definitions and formulas from AIPS and GIPSY
--------------------------------------------

AIPS
*****

A radio velocity is defined by:

.. math::
   :label: eq600

   V = c \bigl( \frac{\nu_0 - \nu^{'}}{\nu_0} \bigr)
   
where :math:`\nu` is the Doppler shifted rest frequency, given by:

.. math::
   :label: eq610

   \nu' = \nu_0\sqrt{(\frac{c-v}{c+v})}
   
Equivalent to the relativistic addition of radial velocities we can derive a relation for 
radio velocities if the velocities in  given in different reference systems.
 
The addition of radial velocities is given in
AIPS memo 27 [Aipsmemo]_
Non-Linear Coordinate Systems in AIPS (Eric W. Greisen, NRAO) Greisen, is

.. math::
   :label: eq620
   
   v = \frac{v_s + v _{obs}}{1 + \frac{v_s v_{obs}}{c^2}}
   
To stay close to our previous examples and definitions we set :math:`v_s` 
which is the radial velocity of an object w.r.t. an inertial system, to
be equal to  :math:`v_b` (our inertial system in this case is barycentric).

The other velocity, :math:`v_{obs}` is equal to the topocentric correction: :math:`v_t`
and the result :math:`v = v_e`, the radial velocity of the object as we would observe
it on earth.
 
Then we get the familiar formula (eq. :eq:`eq70`):

.. math::
   :label: eq630

   v_e = \frac{v_b + v_t}{1 + \frac{v_b v_t}{c^2}}
   
With the relation between V and v and the relativistic addition of velocities we find that
the radio velocities in different systems are related according to the equation:

.. math::
   :label: eq640

   V_e = V_b + V_t - V_b  V_t/ c
   
(see also AIPS memo 27 [Aipsmemo]_ ).
The barycentric radio velocity was calculated in a previous section.
Its value was :math:`V_b` = 8850750.90404 m/s. With the topocentric reference frequency
1378351174.05 Hz we find :math:`V_e` = 8876087.18567 m/s. We know from fig. 1 that the
topocentric correction is positive. To calculate the corresponding radio velocity :math:`V_t`
we use:

.. math::
   :label: eq650

   V_t = c(\frac{\nu_b - \nu_e}{\nu_b}) = 299792458.0\times\frac{(1378471216.43-1378351174.05)}{1378471216.43}=26107.03781\ m/s
   
With these values for :math:`V_b` and :math:`V_t` you can verify that the
expression for :math:`V_e` is valid.

.. math::
   :label: eq660

   V_e = 8850750.90404 +26107.03781  - \frac{8850750.90404 \times 26107.03781}{299792458.0} = 8876087.18567\ m/s
   
which is the value of :math:`V_e` that we found before using the topocentric reference frequency, so we can have
confidence in the relation for radio velocities as found in the AIPS memo [Aipsmemo]_ .

But this radio velocity :math:`V_e` (w.r.t. observer on Earth) for a pixel N is also given by the relation:

.. math::
   :label: eq670
   
   V_e(N) = -\frac{c}{\nu_0}(\nu_e(N)-\nu_0) = -\frac{c}{\nu_0}(\nu_e+\delta_{\nu}(N-N_{\nu})-\nu_0)

It is important to emphasize the meaning of the variables:
         
  *  :math:`\nu_e` = topocentric reference frequency).
  *  :math:`\delta_\nu` = the increment in frequency per pixel in the topocentric system
  *  :math:`N_\nu` = the frequency reference pixel
  *  :math:`N` = the pixel

If we use the previous formulas we can also write:

.. math::
   :label: eq680

   V_e(N_V) = V'_b + V_t - V'_b  V_t/ c

.. math::
   :label: eq690

   V_e(N_V) =  -\frac{c}{\nu_0}(\nu_e+\delta_{\nu}(N_V-N_{\nu})-\nu_0)
   
The velocity :math:`V^{'}_b` is the barycentric reference velocity
at velocity reference pixel :math:`N_V`.


From these relations we observe:

.. math::
   :label: eq700

   V_b(N) = \frac{V_e(N)-V_t}{1-\frac{V_t}{c}}
   
and from eq. :eq:`eq680` with :math:`V^{'}_b = V_b(N_V)`:

.. math::
   :label: eq710

   V_t = \frac{V_e(N_V)-V_b(N_V)}{1-\frac{V_b(N_V)}{c}}
   
Using also the equations with the frequencies, we can derive the following expression
for :math:`V_b(N)`:

.. math::
   :label: eq720

   V_b(N) = V_b(N_V) - \frac{\delta_\nu \bigl(c-V_b(N_V)\bigr)(N-N_V)}{\nu_e + \delta_\nu (N_V-N_\nu)}
   
or in an alternative notation:

.. math::
   :label: eq730

   V_b(N) = V_b(N_V) + \delta_V (N-N_V)
   
Note that in AIPS memo 27 [Aipsmemo]_ the variable :math:`V_R` is used for :math:`V_b(N_V)` and
:math:`V_R` and :math:`N_V` are stored in AIPS headers as alternative
reference information (if frequency is in the main axis description).

The difference between the velocity and frequency reference pixel can be expressed in terms of
the radio velocities  :math:`V_b(N_V)` and :math:`V_b(N_\nu)`.
It follows from eq. :eq:`eq720`) that for :math:`N = N_{\nu}` and a little rearranging:

.. math::
   :label: eq740

   N_V - N_{\nu} = \frac{\nu_e\ \bigl[V_b(N_{\nu}) - V_b(N_V)\bigr]}{\delta_\nu\ \bigl[c-V_b(N_{\nu})\bigr]}
   
We conclude that either one calculates (barycentric) radio velocities using the reference frequency 
and the frequency increment from the header, or one calculates these velocities using
a reference velocity and a velocity increment from the header. 

Note that we assumed that the frequency increment in the barycentric system 
is the same as in the the system of the observer, which is not correct.
However the differences are small (less than 0.01% for 100 pixels from the reference pixel
for typical observations as in our examples). 

For optical velocities Greisen derives:

.. math::
   :label: eq750

   Z_e = Z_b + Z_{t} +  Z_b Z_{t} / c

and:

.. math::
   :label: eq760

   Z_b(N) = Z_b(N_V) - \frac{\delta_\nu\ \bigl(c+Z_b(N_V)\bigr)\ (N-N_Z)}{\nu_e + \delta_\nu (N-N_\nu)}

The difference between the velocity and frequency reference pixels in terms of
optical velocity is:

.. math::
   :label: eq770

   N_Z - N_{\nu} = \frac{\nu_e\ \bigl[Z_b(N_{\nu}) - Z_b(N_Z)\bigr]}{\delta_\nu\ \bigl[c+Z_b(N_{\nu})\bigr]}


Next script demonstrates how we reconstruct the topocentric optical velocity and the reference pixel for
that velocity as it is used in the AIPS formula. Then we compare the output of the WCSLIB method
and the AIPS formula::


   #!/usr/bin/env python
   from kapteyn import wcs
   import numpy
   
   c   = 299792458.0           # m/s From literature
   f0  = 1.42040575200e+9      # Rest frequency HI (Hz)
   fR  = 1.37835117405e+9      # Topocentric reference frequency (Hz)
   dfR = 9.765625e+4           # Increment in topocentric frequency (Hz)
   fb  = 1.3784712164292786e+9 # Barycentric reference frequency (Hz)
   dfb = 97647.745732          # Increment in barycentric frequency (Hz)
   Zb  = 9120.0e+3             # Barycentric optical velocity in m/s
   Nf  = 32                    # Reference pixel for frequency
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : fb,
              'CRPIX1' : Nf,
              'CUNIT1' : 'Hz',
              'CDELT1' : dfR,
              'RESTFRQ': f0
            }
   line = wcs.Projection(header).spectra('VOPT-F2W')
   pixels = numpy.array(range(30,35))
   Vwcs = line.toworld1d(pixels) / 1000
   print """Optical velocities from WCSLIB with spectral
   translation and with barycentric ref. freq. (km/s):"""
   for p,v in zip(pixels, Vwcs):
      print p, v
   
   # Select an arbitrary velocity reference pixel
   Nz = 44.0
   # then calculate corresponding velocity
   Zb2 = (fR*Zb-dfR*c*(Nz-Nf))/(fR+dfR*(Nz-Nf))
   print "Zb(Nz) =", Zb2
   dN = fR*(Zb-Zb2)/(dfR*(c+Zb2))
   Nz = dN + Nf
   print "Closure test for selected reference pixel: Nz=", Nz
   
   print "\nOptical velocities using AIPS formula (km/s):"
   Zs = Zb2 - dfR*(c+Zb2)*(pixels-Nz)/(fR+dfR*(pixels-Nf)) 
   Zs /= 1000
   for p,z in zip(pixels, Zs):
      print p, z
   
   fx = fR + dfR*(Nz-Nf)
   dZ = -dfR*(c+Zb2) / fx
   print "Velocity increment: ", dZ
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'VOPT-F2W',
              'CRVAL1' : Zb2,
              'CRPIX1' : Nz,
              'CUNIT1' : 'm/s',
              'CDELT1' : dZ,
              'RESTFRQ': f0
            }
   line2 = wcs.Projection(header)
   Vwcs = line2.toworld1d(pixels) / 1000
   print """\nOptical velocities from WCSLIB without spectral
   translation with barycentric Z (km/s):"""
   for p,v in zip(pixels, Vwcs):
      print p, v
   # Output:
   # Optical velocities from WCSLIB with spectral
   # translation and with barycentric ref. freq. (km/s):
   # 30 9163.77531689
   # 31 9141.88610773
   # 32 9120.0
   # 33 9098.11699305
   # 34 9076.23708621
   # Zb(Nz) = 8857585.54671
   # Closure test for selected reference pixel: Nz= 44.0
   #
   # Optical velocities using AIPS formula (km/s):
   # 30 9163.77912988
   # 31 9141.88801395
   # 32 9120.0
   # 33 9098.11508736
   # 34 9076.23327538
   # Velocity increment:  -21849.2948239
   #
   # Optical velocities from WCSLIB without spectral
   # translation with barycentric Z (km/s):
   # 30 9163.77912988
   # 31 9141.88801395
   # 32 9120.0
   # 33 9098.11508736
   # 34 9076.23327538


Note that we used the topocentric frequency increment in the WCSLIB call
for a better comparison with the AIPS formula. The output of velocities with the AIPS formula
is exactly the same as WCSLIB with optical velocities using the velocity increment
calculated with the AIPS method (as to be expected). And these velocities 
are very close to the velocities calculates with WCSLIB using the barycentric
frequency that corresponds to the given optical velocity. The differences can be explained
with the fact that the different methods are used to calculate a velocity increment.

What did we prove with this script? We selected an arbitrary pixel as reference pixel for
the velocity. This velocity has a relation with the initial optical velocity (9120 km/s)
through the difference in reference pixels. We calculated that velocity and showed that
the AIPS formula generates results that are almost equal to WCSLIB with the barycentric
reference frequency. If we use the AIPS formulas to calculate a velocity increment, 
we can use the values in WCSLIB if we set CTYPE to 'VOPT-F2W'. This generates exactly
the same results as with the AIPS formula for velocities. So in frequency mode 
WCSLIB calculates topocentric frequencies (and *topocentric* velocities
if we use a spectral translation method) 
and in velocity mode it calculates barycentric velocities. AIPS axis type FELO can be
used as input for WCSLIB without modification.

**Conclusions**


   * In AIPS the reference pixel for the reference velocity differs from the
     frequency reference pixel. There is a relation between this reference velocity and 
     the barycentric velocity and these reference pixels. To us it is not clear 
     what this reference velocity represents and why it is not changed to a velocity at
     the same reference pixel as the frequency.
   * In the AIPS approach it is assumed that the increment in frequency is the same in
     different reference systems. This assumption is not correct, but the deviations are 
     usually very small.


GIPSY
***** 

The formulas used in GIPSY to convert frequencies to velocities are described in section:
`spectral coordinates <http://www.astro.rug.nl/~gipsy/pguide/coordinates.html>`_
in the GIPSY programmers guide.
There is a formula for radio velocities and one for optical velocities.
Both formulas are derived from the standard formulas but the result is 
split into the reference velocity and a part that is a function of the increment in 
frequency.

**Radio:**

.. math::
   :label: eq800

   V = -c (\frac{\nu'-\nu_0}{\nu_0})

The frequencies in the FITS and GIPSY headers are observed frequencies.

Assume for pixel :math:`N`:  :math:`\nu' = \nu + \Delta \nu + (N-N_{\nu}) \delta \nu`


with :math:`\nu` the topocentric reference value and :math:`\Delta \nu` the difference between
the topocentric and barycentric frequencies. Then we can write for the radio
velocity:

.. math::
   :label: eq810

   V(N) = -c\bigl(\frac{\nu+\Delta \nu-\nu_0}{\nu_0}\bigr) -\frac{c}{\nu_0}(N-N_{\nu}) \delta_{\nu}

But the first part of the right hand side of the formula is the definition of
:math:`V_b` for the Nmap/GIPSY case of a velocity in the inertial system (the inertial system
for the velocity is coded
in keyword CTYPE=, e.g. OHEL or OLSR). So the formula becomes:

.. math::
   :label: eq820

   V_b(N) = V_b(N_\nu) -\frac{c}{\nu_0}(N-N_{\nu}) \delta_{\nu}

which is as expected. The increment in radio velocity was derived before (in eq. :eq:`eq260`) and the increment
in velocity is linear with the increment in frequency.

We show the comparison between the WCSLIB method and the Nmap/GIPSY method::


   #!/usr/bin/env python
   import numpy as n
   from kapteyn import wcs
   
   c   = 299792458.0           # m/s From literature, not as used in GIPSY
   f0  = 1.42040575200e+9      # Rest frequency HI (Hz)
   fR  = 1.37835117405e+9      # Topocentric reference frequency (Hz)
   dfR = 9.765625e+4           # Increment in topocentric frequency (Hz)
   fb  = 1.37847121643e+9      # Barycentric reference frequency (Hz)
   dfb = 97647.745732          # Increment in barycentric frequency (Hz)
   VR  = 8.85075090419e+6      # Barycentric reference velocity
   Nf  = 32                    # Reference pixel for frequency
   
   header = { 'NAXIS'  : 1,
              'CTYPE1' : 'FREQ',
              'CRVAL1' : fb,
              'CRPIX1' : Nf,
              'CUNIT1' : 'Hz',
              'CDELT1' : dfb,
              'RESTFRQ': f0
            }
   line = wcs.Projection(header).spectra('VRAD')
   pixels = range(30,35)
   Vwcs = line.toworld1d(pixels)
   print "\nMethod used in WCSLIB with barycentric reference frequency (km/s):"
   for p,v in zip(pixels, Vwcs):
      print p, v/1000
   
   
   print "\nGIPSY formula (km/s):"
   for n in pixels:
      Vs = VR - dfR*c*(n-Nf)/f0
      print n, Vs/1000.0
   # Output:
   # Method used in WCSLIB with barycentric reference frequency (km/s):
   # 30 8891.97019321
   # 31 8871.36054862
   # 32 8850.75090404
   # 33 8830.14125946
   # 34 8809.53161488
   #
   # GIPSY formula (km/s):
   # 30 8891.9737832
   # 31 8871.36234369
   # 32 8850.75090419
   # 33 8830.13946469
   # 34 8809.52802518

To be consistent with previous sections we selected the barycentric radio velocity from
`CRVAL3R= 8.85075090419e+6`. For WCSLIB we used the barycentric increment in frequency.
If we replace this by the topocentric frequency increment, then the result agree slightly better,
but are less correct.

**Optical:**

For the optical velocity we get after applying the same procedure:

.. math::
   :label: eq850

   Z = -c (\frac{\nu'-\nu_0}{\nu'})
   
Again, assume for pixel :math:`N`: 
:math:`\nu' = \nu_e + \Delta \nu + (N-N_{\nu}) \delta \nu`

with :math:`\nu` the topocentric reference value and :math:`\Delta \nu` the difference between
the topocentric and barycentric frequencies.

Inserting this and rearranging the equation gives:

.. math::
   :label: eq860

   Z_b(N) = -c\bigl(\frac{\nu_e+\Delta \nu-\nu_0}{\nu_e+\Delta \nu}\bigr) +c \nu_0\ (\frac{1}{\nu_e+\Delta +(N-N_{\nu}) \delta_{\nu}} - \frac{1}{\nu_e+\Delta \nu})

with:
  * :math:`Z_b(N)` is the barycentric optical velocity for pixel :math:`N`,
  * :math:`\nu_e` is the topocentric reference frequency,
  * :math:`\delta_{\nu}` is the increment in topocentric frequency
  * :math:`\Delta \nu` is the difference between barycentric and topocentric reference frequency.

We recognize in the first part of the right side of the equation the optical 
velocity in our inertial system, then:

.. math::
   :label: eq870

   Z_b(N) = Z_b(N\nu) + (\frac{1}{\nu'_e+(N-N_\nu) \delta_{\nu}} - \frac{1}{\nu'_e} )\ \nu_0 c
   
with :math:`Z_b(N_{\nu)}` the barycentric reference velocity from the (FITS) header.

But for the velocity increment we still have :math:`\nu'_e` in our formula
so in fact we are dealing with :math:`\nu_e + \Delta \nu` which is the barycentric reference
frequency. For Nmap/GIPSY data there is no barycentric frequency available but the 
topocentric frequency is used instead. Later we show why this is a reasonable approximation.
Next example code shows the actual output of the GIPSY formula with both
the observed frequency and with the barycentric reference frequency. The differences are small.
These numbers can be compared with WCSLIB output with barycentric frequency en frequency increment::

   #!/usr/bin/env python
   import numpy as n
   from kapteyn import wcs
   
   c   = 299792458.0           # m/s From literature, not as used in GIPSY
   f0  = 1.42040575200e+9      # Rest frequency HI (Hz)
   fR  = 1.37835117405e+9      # Topocentric reference frequency (Hz)
   dfR = 9.765625e+4           # Increment in topocentric frequency (Hz)
   fb  = 1.37847121643e+9      # Barycentric reference frequency (Hz)
   dfb = 97647.745732          # Increment in barycentric frequency (Hz)
   ZR  = 9120000               # Barycentric optical reference velocity 
   Nf  = 32                    # Reference pixel for frequency
   
   header = {'NAXIS'  : 1,
             'CTYPE1' : 'FREQ',
             'CRVAL1' : fb,
             'CRPIX1' : Nf,
             'CUNIT1' : 'Hz',
             'CDELT1' : dfb,
             'RESTFRQ': f0
            }
   line = wcs.Projection(header).spectra('VOPT-F2W')
   pixarray = numpy.array(range(30,35))
   Vwcs = line.toworld1d(pixarray)
   print "\nMethod used in WCSLIB with barycentric reference frequency (km/s):"
   for p,v in zip(pixarray, Vwcs):
      print p, v/1000
   
   Zs = ZR + c*f0*(1/(fR+(pixarray-Nf)*dfR)-1/fR)
   Zsb = ZR + c*f0*(1/(fb+(pixarray-Nf)*dfb)-1/fb)
   print "\nGIPSY formula with topo and bary data (km/s):"
   for p, w1, w2 in zip(pixarray, Zs, Zsb):
      print p, w1/1000.0, w2/1000.0
   # Output:
   # Method used in WCSLIB with barycentric reference frequency (km/s):
   # 30 9163.77150407
   # 31 9141.88420151
   # 32 9119.99999984
   # 33 9098.1188984
   # 34 9076.24089654
   #
   # GIPSY formula with topo and bary data (km/s):
   # 30 9163.78294266 9163.77150423
   # 31 9141.88992021 9141.88420167
   # 32 9120.0 9120.0
   # 33 9098.11318138 9098.11889856
   # 34 9076.22946368 9076.24089671

This output proves that the Nmap/GIPSY formula for optical velocities (Zs in script) is a good approximation
if we have only the observed frequencies available. The formula gives exact results (Zsb in script) 
if we insert the barycentric frequency and frequency increment. What remains is the question how good
the approximation is...

If we calculate the derivative of the optical velocity at the reference pixel, we find:

.. math::
   :label: eq880

   \frac{dZ}{d\nu_e} \approx c \nu_0\ \Bigl(\frac{2(N-N_{\nu})\delta_{\nu}}{\nu_e^3}\Bigr)
   
So with the difference between topocentric and barycentric frequency this becomes:

.. math::
   :label: eq890

   {\Delta Z} \approx c \nu_0\ \Bigl(\frac{2(N-N_{\nu})\delta_{\nu}}{\nu_e^3}\Bigr) \Delta \nu

With:

   * :math:`\nu_e` = 1.37835117405e9 Hz
   * :math:`\delta_{\nu}` = 9.765625e4 Hz
   * :math:`\nu_b` = 1.37847121643e+9 Hz
   * :math:`c` = 299792458.0 m/s
   * :math:`\nu_0` = 0.14204057583700e+10 Hz
   * :math:`\Delta \nu = \nu_b - \nu_e` = 120042.38 Hz
   
we calculate for the pixel next to the reference pixel this becomes 0.0038 km/s
(in our previous example we find a difference of 0.0057 km/s).
After 100 pixels the difference is 0.38 km/s which seems acceptable for most radio
observations.

**Conclusions**
   
  * The GIPSY formulas assume constant frequency increments in different reference systems
    which gives small deviations from the result with WCSLIB.
  * The formula for the optical velocity is an approximation. The deviations are small but
    depend on the pixel i.e. :math:`(N-N_{\nu})`. This approximation is not necessary because
    when the optical velocity in the barycenter is given, then 
    one can calculate the barycentric reference frequency (see eq. :eq:`eq10`)
    and use that frequency in the GIPSY formula to get the exact result.




Header items in a (legacy) WSRT FITS file
-----------------------------------------

Program *nmap* (part of NEWSTAR which is a package specifically developed
to reduce WSRT data)
is/was used to create FITS files with WSRT image data. 
We investigated the meaning or interpretation of the various FITS header items. 
The program generated it own descriptors related to velocities and frequencies.
For example:
   

    * VEL: Velocity (m/s)
    * VELC: Velocity code
        *  0=continuum,
        *  1=heliocentric radio
        *  2=LSR radio
        *  3=heliocentric optical
        *  4=LSR optical
    * VELR: Velocity at reference frequency (FRQC)
    * INST: Instrument code (0=WSRT, 1=ATCA)
    * FRQ0: Rest frequency for line (MHz) 
    * FRQV: Real frequency for line (MHz) 
    * FRQC: Centre frequency for line (MHz) 

One of functions in *nmap* is called *nmawfh.for*.
It writes a FITS header using the values in the *nmap* descriptors.


The value of *CRVAL3* is set to *FRQV* if the velocity code is one of
combinations of optical and radio velocity with heliocentric or local standard of rest
reference systems (i.e. RHEL, RLSR, OHEL, OLSR).

The value of *CRPIX3* is equal to *FRQV* -lowest frequency divided by the
channel separation. 'lowest frequency' is the frequency of the input channel with
the lowest frequency.

  * The value for FITS keyword VEL= is equal to *nmap* descriptor VEL, the centre velocity in m/s
  * The value for FITS keyword VELR= is equal to *nmap* descriptor VELR, the Reference velocity
  * The value for FITS keyword FREQR= is equal to *nmap* descriptor FRQC, the Reference frequency (Hz)
  * The value for FITS keyword FREQ0= is equal to *nmap* descriptor FRQ0, the Rest frequency (Hz)

::

   VEL            !CENTRE VELOCITY (M/S)
   VELCODE        !VELOCITY CODE
   VELR           !REFERENCE VELOCITY (M/S)
   FREQR          !REFERENCE FREQUENCY (HERTZ)
   FREQ0          !REST FREQUENCY (HERTZ)


WCSlib in a GIPSY task
+++++++++++++++++++++++

GIPSY (`Groningen Image Processing SYstem <https://www.astro.rug.nl/~gipsy>`_ )
is one of the oldest image processing and data analysis systems. Python can be used
to create GIPSY tasks. The Kapteyn Package is integrated in GIPSY. Here we
give a small example how to use both.

Assuming you have a data set with three axes and the last axis is the spectral axis,
the next script is a very small GIPSY program that asks the user for the name of this set 
and then calculates the optical velocities for a number of pixels in the neighborhood of
the reference pixel (CRPIX3).

GIPSY data sets consist of two files. One file contains the image data.
The other is called the descriptor. The descriptor contains FITS header 
items (e.g. CRVAL1=) and GIPSY specific keywords but not only attached to the
set but also to subsets (slices) of the data. Not only planes or lines can have
their own header but even pixels can. The script below reads it information from 
top level (which hosts the global description of the data cube itself)::
   
   #!/usr/bin/env python
   from gipsy import *
   from kapteyn import wcs
   
   init()
   
   while True:
      try:
         set = Set(usertext('INSET=', 'Input set'))
         break
      except:
         reject('INSET=', 'Cannot open set')
   
   proj = wcs.Projection(set).sub((3,))
   s = "Ref. freq at that pixel: %f Hz" % (set['CRVAL3'],)
   anyout(s)
   s = "Velocity: %f m/s" % (set['DRVAL3'],)
   anyout(s)
   
   crpix = set['CRPIX3']
   
   proj2 = proj.spectra('VOPT-F2W')
   for i in range(-2,+3):
      world = proj2.toworld((crpix+i,))[0]/1000.0   #  to world coordinates
      anyout(str(world)+' km/s')
   
   finis()

This little GIPSY task simulates the functionality of GIPSY task *COORDS*
which lists world coordinates for data slices. The two most important differences between
this task and  *COORDS* are:
      
      * With WCSlib it is simple to change the
        output velocity to radio or radial by changing the spectral translation.
      * The Python interface to WCSlib prepares
        the GIPSY header information to give correct barycentric or lsrk velocities (i.e. it
        also converts the frequency increment to the barycentric or lsrk system).

Read more about GIPSY tasks written in Python in
`Python recipes for GIPSY <https://www.astro.rug.nl/~gipsy/python/recipes/pythonrep.php>`_





References
----------

.. [Aipsmemo] `AIPS memo 27 <_static/aips27.ps>`_  Non-Linear Coordinate Systems in AIPS (Eric W. Greisen, NRAO)

        
