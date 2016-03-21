#----------------------------------------------------------------------
# FILE:    tabarray.py
# PURPOSE: This module provides a class which allows the user to read,
#          write and manipulate simple table-like structures.
#          It is based on NumPy and the table-reading part has been
#          optimized for speed.  When the flexibility of SciPy's
#          read_array() function is not needed, Tabarray can
#          be considered as an alternative. 
# AUTHOR:  J.P.Terlouw, University of Groningen, The Netherlands
# DATE:    September 29, 2008
# UPDATE:  October 9, 2008
# VERSION: 1.0
#
# (C) University of Groningen
# Kapteyn Astronomical Institute
# Groningen, The Netherlands
#----------------------------------------------------------------------

"""
================
Module tabarray
================

.. author:: Hans Terlouw <gipsy@astro.rug.nl>

.. highlight:: python
   :linenothreshold: 5

Module tabarray provides a class which allows the user to read,
write and manipulate simple table-like structures.
It is based on NumPy and the table-reading part has been
optimized for speed.  When the flexibility of SciPy's
read_array() function is not needed, Tabarray can
be considered as an alternative.

Class tabarray
--------------

.. autoclass:: tabarray(source[, comchar='#!', sepchar=' \\\\t', lines=None, bad=None, segsep=None])
   :members:


Functions
---------

.. autofunction:: readColumns

.. autofunction:: writeColumns
"""

import numpy, string, operator
from kapteyn.ascarray import ascarray

class tabarray(numpy.ndarray):

   """
Tabarray is a subclass of NumPy's ndarray.  It provides all of
ndarray's functionality as well as some extra methods and attributes. 

:param source:
   the object from which the tabarray object is constructed.  It can be a
   2-dimensional NumPy array, a list or tuple containing the table columns
   as 1-dimensional NumPy arrays, or a string with the name of a text file
   containing the table.  Only in the latter case the other arguments are
   meaningful.
:param comchar:
   a string with characters which are used to designate comments in the
   input file.  The occurrence of any of these characters on a line causes
   the rest of the line to be ignored.  Empty lines and lines containing
   only a comment are also ignored.
:param sepchar:
   a string containing the
   column separation characters to be used.  Columns are separated by any
   combination of these characters. 
:param lines:
   a two-element tuple or list specifying a range of lines
   to be read.  Line numbers are counted from one and the range is
   inclusive.  So (1,10) specifies the first 10 lines of a file.  Comment
   lines are included in the count.  If any element of the tuple or list is
   zero, this limit is ignored.  So (1,0) specifies the whole file, just
   like the default None. 
:param bad:
   is a number to be substituted for any field which cannot be
   decoded as a number.  The default None causes a ValueError exception to
   be raised in such cases.
:param segsep:
   a string containing segment separation characters. If any of
   these characters is present in a comment block, this comment block
   is taken as the end of the current segment. The default None indicates
   that every comment block will separate segments.
:raises:
   :exc:`IOError`, when the file cannot be opened.

   :exc:`IndexError`, when a line with an inconsistent number of fields is
   encountered in the input file.

   :exc:`ValueError`: when a field cannot be decoded as a number and
   no alternative value was specified.

**Attributes:**

.. attribute:: nrows

   the number of rows

.. attribute:: ncols

   the number of columns

.. attribute:: segments

   a list with slice objects which can be used to address the different
   segments from the table. Segments are parts of the table which
   are separated by comment blocks which meet the conditions specified
   by argument *segsep*. The following example illustrates how a program
   can iterate over all segments::

      from kapteyn.tabarray import tabarray

      coasts = tabarray('world.txt')

      for segment in coasts.segments:
         coast = coasts[segment]


**Methods:**

"""

   def __new__(cls, source, comchar='#!', sepchar=' \t', lines=None,
               bad=None, segsep=None):
      if isinstance(source, numpy.ndarray):
         return source.view(cls)
      elif isinstance(source, tuple) or isinstance(source, list):
         return numpy.column_stack(source).view(tabarray)
      else:
         arrayspec = ascarray(source, comchar, sepchar, lines, bad, segsep)
         array = arrayspec[0].view(cls)
         array.segments = arrayspec[1] 
         return array

   def __init__(self, source, comchar=None, sepchar=None, lines=None,
               bad=None, segsep=None):
      self.nrows, self.ncols = self.shape
      try:
         self.segments
      except:
         self.segments = [slice(0,self.nrows)]

   def __array_finalize__(self, obj):
      try:
         self.nrows, self.ncols = self.shape
      except:
         pass
      try:
         self.segments = [slice(0,self.nrows)]
      except:
         pass

   def columns(self, cols=None):
      """
:param cols:
   a tuple or list with the numbers (zero-relative) of the columns
   to be extracted.
:returns: a NumPy array.

Extract specified columns from a tabarray and return an array containing
these columns.  Cols is a tuple or list with the column numbers.  As the
first index of the resulting array is the column number, multiple
assignment is possible.  E.g., ``x,y = t.columns((2,3))`` delivers columns 2
and 3 in variables x and y.  Default: return all columns.

"""
      if cols is None:
         return self.T.view(numpy.ndarray)
      else:
         return self.take(cols, 1).T.view(numpy.ndarray)

   def rows(self, rows=None):
      """
:param rows:
   a tuple or list containing the numbers (zero-relative) of the rows
   to be extracted.
:return: a new tabarray.

This method extracts specified rows from a tabarray and returns a new tabarray. 
Rows is a tuple or list containing the row numbers to be extracted. 
Normal Python indexing applies, so (0, -1) specifies the first and the
last row.  Default: return whole tabarray. 
"""
      if rows is None:
         return self
      else:
         return self.take(rows, 0)

   def writeto(self, filename, rows=None, cols=None, comment=[], format=[]):
      """
Write the contents of a tabarray to a file.

:param filename:
   the name of the file to be written.
:param rows:
   a tuple or list with a selection of the rows (zero-relative) te be written.
   Default: all rows. 
:param columns:
   a tuple or list with a selection of the columns (zero-relative)
   to be written. Default: all columns. 
:param comment:
   a list with text strings which will be inserted as comments in the
   output file.  These comments will be prefixed by the hash character (#).
:param format:
   a list with format strings for formatting the output, one element per
   column, e.g., ``['%5d', ' %10.7f', ' %g']``. 



"""
      arrout = self.rows(rows)
      if cols is not None:
         arrout = arrout.take(cols, 1)
      f = open(filename, 'w')
      for line in comment:
         f.write('# %s \n' % line)
      columns = range(arrout.ncols)
      if not format:
         format = ['%10g ']*arrout.ncols
      for line in xrange(arrout.nrows):
         outline = ' '
         for column in columns:
            outline += format[column] % arrout[line, column]
         outline = outline.rstrip() + '\n'
         f.write(outline)
      f.close()

def readColumns(filename, comment='!#', cols='all', sepchar=', \t',
                rows=None, lines=None, bad=0.0,
                rowslice=(None,), colslice=(None,)):
   """
TableIO-compatible function for directly extracting table data from a file.

:param filename:
   a string with the name of a text file containing the table.
:param comment:
   a string with characters which are used to designate comments in the
   input file.  The occurrence of any of these characters on a line causes 
   the rest of the line to be ignored.  Empty lines and lines containing
   only a comment are also ignored. 
:param cols:
   a tuple or list with the column numbers or a scalar with one column number.
:param sepchar:
   a string containing the column separation characters to be used. 
   Columns are separated by any combination of these characters. 
:param rows:
   a tuple or list containing the row numbers to be extracted.
:param lines:
   a two-element tuple or list specifying a range of lines to be read. 
   Line numbers are counted from one and the range is inclusive.  So (1,10)
   specifies the first 10 lines of a file.  Comment lines are included in
   the count.  If any element of the tuple or list is zero, this limit is
   ignored.  So (1,0) specifies the whole file, just like the default None. 
:param bad:
   a number to be substituted for any field which cannot be decoded
   as a number.
:param rowslice:
   a tuple containing a Python slice indicating which rows should be selected.
   If this argument is used in combination with the argument *rows*, the latter
   should be expressed in terms of the *new* row numbers after slicing.
   Example: ``rowslice=(10, None)`` selects all rows, beginning with the
   eleventh (the first row has number 0) and ``rowslice=(10, 13)`` selects
   row numbers 10, 11 and 12.
:param colslice:
   a tuple containing a Python slice indicating which columns should be
   selected.
   If this argument is used in combination with the argument *cols*, the latter
   should be expressed in terms of the *new* column numbers after slicing.
   Selection is analogous to *rowslice*.

"""
   if cols=='all':
      cols = None
   if cols is not None and not operator.isSequenceType(cols):
      cols = [cols]
   return tabarray(filename, comment, sepchar=sepchar, lines=lines, bad=bad
                  )[slice(*rowslice),slice(*colslice)].rows(rows).columns(cols)

def writeColumns(filename, list, comment=[]):
   """
TableIO-compatible function for directly writing table data to a file.

:param filename:
   the name of the file to be written;
:param list:
   a list containing the columns to be written.
:param comment:
   a list with text strings which will be inserted as comments in the
   output file. These comments will be prefixed by the hash character (#).
"""
   tabarray(list).writeto(filename, comment=comment)

__version__ = '1.3'
__docformat__ = 'restructuredtext'
