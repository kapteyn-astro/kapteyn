Tutorial tabarray module
==========================

.. highlight:: python
   :linenothreshold: 2


Introduction
------------

Many applications send output of numbers to plain text (:term:`ASCII`)
files in a rectangular form. I.e. they store human readable
numbers in one ore more columns in one or more rows. In one of our
example figures to illustrate the use of graticules
we plotted coastline data from a text file with
coordinates in longitude and latitude in a :mod:`wcs` supported projection.

If you want to plot such data or you need data to do calculations, then you
need a function or method to read the data into for example NumPy arrays.
Package SciPy provides a function *read_array()* in module :mod:`scipy.io.array_import`.
It has all necessary features to do the job but it is very slow when it needs to read
big files with many numbers.

We wrote a fast version in module :mod:`tabarray` which is also part of
the Kapteyn Package  Its speed is comparable to
a well known module that is no longer supported, called *TableIO*.
The module interfaces with C and is written in *Cython*.
Such interfaces improve the speed of reading the data with a factor of 5-10 compared
to Python based solutions.
Module :mod:`tabarray` has a simple interface and an object oriented interface.
For the simple interface functions we used the same function names as *TableIO*.
This will simplify the migration from *TableIO* to :mod:`tabarray`.


Simple interface functions
--------------------------

Function readColumns
....................

A typical example of a text data file is given below.
It has 3 columns and several rows of which a number of rows represent a comment.
To experiment with tabarray functions and methods you can copy this data and store it
as `testdata.txt` on disk.::
   
   ! ASCII file 'testdata.txt' 12-09-2008
   !
   ! X    |   Y   |    err
   23.4    -44.12   1.0e-3
   19.32   0.211    0.332
   # Next numbers are include as
   -22.2   44.2     3.2
   1.2e3   800      1

Assuming you have some knowledge of the contents and structure of the file,
it is easy to read it into NumPy arrays. We use variables *x*, *y* and *err*
to represent the columns. The comment characters are '#' and '!' and are
included in the comment string which is the second parameter of the
:func:`tabarray.Readcolumns` function. The file on disk is identified by its name.
There is no need to open it first. Use the commands given below to read all
the data from our test file.

>>> from kapteyn import tabarray
>>> x,y,err = tabarray.readColumns('testtable.txt','#!')
>>> print x
[   23.4 ,    19.32,   -22.2 ,  1200.  ]

All numbers are converted to floating point.
Blank lines at the end of a file are ignored.
Blank lines in the middle of a file are treated as comment lines.
Suppose you want to read only the second and third column,
then one needs to specify the columns. The first column has index 0.

>>> y,err = tabarray.readColumns('testtable.txt','#!', cols=(1,2))
>>> print err
[  1.00000000e-03   3.32000000e-01   3.20000000e+00   1.00000000e+00]

.. note::

   Column and row numbers start with 0. The last row or last column
   is addressed with -1.

To make a selection of rows you can specify the rows parameter.
Rows are given as a sequence and the first row in a file has index 0.
Suppose you want to read the last two rows from the last two columns
in the text file together with the first row, then we could write:

>>> x,y = tabarray.readColumns('testtable.txt','#!', cols=(1,2), rows=(2,3,0))
>>> print x
[  44.2   800.    -44.12]

To read only the last row in your data you should use `rows=(-1,)`.

If you know beforehand which lines of the data files should be read,
you can set the converter to read only the lines in parameter *lines*.
For a big text file (called *satview.txt*) containing longitudes and latitudes of positions
in two columns, we are only interested in the first 1000 lines containing
relevant data. Then the *lines* parameter saves time.
So we use the following command:

>>> lons, lats = tabarray.readColumns('satview.txt','s', lines=(0,1000))

Comment lines in this *satview.txt* file do not start with a common
comment character, instead it starts with the word 'segment' so our
comment character becomes 's'.


Function writeColumns
.....................

One dimensional array data can also be written back to a file on disk.
The function for writing data is called :func:`tabarray.writeColumns`.
Its first argument is the name of the file. The second is a sequence
with columns. With the columns 'x' and 'y' from the *testtable.txt* file
in the previous section,
we want to write a new file where column 'y' is the first column and
column 'x' is the second.
Here is the code to do this:

>>> x,y,err = tabarray.readColumns('testtable.txt','#!')
>>> tabarray.writeColumns('testout.txt', (y,x)) 
# Contents on disk is:
     -44.12       23.4
      0.211      19.32
       44.2      -22.2
        800       1200

The columns are one dimensional NumPy arrays.
This implies that we can do some array arithmetic on the columns.
We could have changed our columns to:

>>> tabarray.writeColumns('testout.txt', (y*y,x*y,x*x))
# Contents on disk is:
    1946.57   -1032.41     547.56
   0.044521    4.07652    373.262
    1953.64    -981.24     492.84
     640000     960000   1.44e+06

which makes this function very powerful.

It is common practice to start text data file with some comments.
The next code shows how to write a date and the name of the author in a new
file with function :func:`tabarray.writeColumns`. The comments parameter
is a list with strings. Each string is written on a new line at the start of the text file.

>>> when = datetime.datetime.now().strftime("Created at: %A (%a) %d/%m/%Y")
>>> author = 'Created by: Kapteyn'
>>> tabarray.writeColumns('testout.txt', (y*y,x*y,x*x), comment=[when, author])

The header of the file will look similar to this::

   # Created at: Thursday (Thu) 18/09/2008
   # Created by: Kapteyn


Tabarray objects and methods
----------------------------

Reading data and making selections
..................................

A *tabarray* object is created with method :meth:`tabarray.tabarray`.
Again we want to read the data from file 'testtable.txt'.

>>> t = tabarray.tabarray('testtable.txt', '#!')
>>> print t
[[  2.34000000e+01  -4.41200000e+01   1.00000000e-03]
 [  1.93200000e+01   2.11000000e-01   3.32000000e-01]
 [ -2.22000000e+01   4.42000000e+01   3.20000000e+00]
 [  1.20000000e+03   8.00000000e+02   1.00000000e+00]]

Selections are made with methods :meth:`tabarray.rows` and :meth:`tabarray.columns`.

.. warning:: 

   The *rows()* method needs to be applied before the *columns()* method because
   for the latter, the array *t* is transposed and its row information is
   changed.
     
With this knowledge we can combine the methods in one statement
to read a selection of lines and a selection of columns into NumPy arrays.

>>> x,y = t.rows((2,3)).columns((1,2))
>>> print x
[  44.2  800. ]
>>> print y
[ 3.2  1. ]

If you want to select rows in a NumPy vector that is already filled with
data from disk after applying the lines and/or rows parameters you still
can extract data using NumPy indexing:

>>> lines = [0,1,3]
>>> print err[lines]
[ 0.001  0.332  1.   ]


Messy files
...........

ASCII text readers should be flexible and robust.
Examine the contents of the next ASCII data file (which we stored
on disk as *messyascii.txt*)::

   
   ! Very messy data file
   
   23.343, 34.434, 1e-20
   10, 20, xx
   
   
   2 4      600
   -23.23, -0.0002, -3x7
      # Some comment
   
   40, 50.0, 70.2


It contains blank lines at the end and between the data and it has
three different separators (spaces, comma's and tabs). Also it contains
data that cannot be converted to numbers. Instead of an exception we want
the converter to substitute a user given value for a string that could not
be converted to a number. Assume that a user wants -999 for those bad entries,
then the numbers should be read by:

>>> t= tabarray.tabarray('messyascii.txt','#!', sepchar=' ,\t', bad=-999)
>>> print t
[[  2.33430000e+01   3.44340000e+01   1.00000000e-20]
 [  1.00000000e+01   2.00000000e+01  -9.99000000e+02]
 [  2.00000000e+00   4.00000000e+00   6.00000000e+02]
 [ -2.32300000e+01  -2.00000000e-04  -9.99000000e+02]
 [  4.00000000e+01   5.00000000e+01   7.02000000e+01]]
>>> x,y = t.rows(range(1,4)).columns((1,2))  # Extract some rows and columns
>>> print x
[  2.00000000e+01   4.00000000e+00  -2.00000000e-04]
>>>print y   # Contains the 'bad' numbers
[-999.  600. -999.]

Note that we could have used function :func:`tabarray.readColumns` also
to get the same results:

>>> x,y = tabarray.readColumns('messyascii.txt','#!', sepchar=' ,/t', bad=-999, rows(range(1,4)), cols=(1,2))


.. note::

   Probably more useful as a bad number indicator is the 'Not a Number' (NaN) from
   NumPy. Use it as in: `bad=numpy.nan` and test on these numbers with NumPy's
   function: *isnan()*.


Glossary
--------

.. glossary::

   ASCII
      *American Standard Code for Information Interchange* is a character-encoding
      scheme based on the ordering of the English alphabet.