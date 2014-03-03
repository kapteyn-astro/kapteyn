"""
================
Module ascarray
================

.. author:: Hans Terlouw <gipsy@astro.rug.nl>

This module is the base containing the function on which :mod:`tabarray`
has been built.

Function ascarray
-----------------

.. autofunction:: ascarray(filename[, comchar='#!', sepchar=', \\\\t', lines=None, bad=None, segsep=None])

"""

import numpy

# ==========================================================================
#                             Declarations
# --------------------------------------------------------------------------
 
ctypedef void FILE

cdef extern from "locale.h":
   cdef int LC_NUMERIC
   char *setlocale(int category, char *locale)

cdef extern from "numpy/arrayobject.h":
   cdef enum NPY_TYPES:
      NPY_DOUBLE
   ctypedef int npy_intp
   object PyArray_SimpleNewFromData(int nd, npy_intp* dims, NPY_TYPES type_num,
                                            void* data)
   void import_array()
                                                
cdef extern from "stdlib.h":
   ctypedef int size_t           
   void* malloc(size_t size)
   void *realloc(void *ptr, size_t size)
   void free(void* ptr)
   double strtod(char *nptr, char **endptr)

cdef extern from "stdio.h":
   int fseek(FILE *stream, long offset, int whence)
   FILE *fopen(char *filename, char *mode)
   int fclose(FILE *stream)
   long ftell(FILE *stream)
   char *fgets(char *s, int size, FILE *stream)
   enum codes:
      SEEK_SET
      SEEK_CUR
      SEEK_END
      
cdef extern from "string.h":
   char *strpbrk(char *s, char *accept)
   char *strtok(char *str, char *delim)
   int strlen(char *s)

import_array()

# ==========================================================================
#                             ascarray
# --------------------------------------------------------------------------
#  Read an ASCII table file and return its data as a NumPy array.
#
def ascarray(filename, char *comchar='#!', sepchar=', \t', lines=None,
             bad=None, segsep=None):
   """
Read an ASCII table file and return its data as a NumPy array.

:param source:
   a string with the name of a text file containing the table.
:param comchar:
   string with characters which are used to designate
   comments in the input file.  The occurrence of any of these
   characters on a line causes the rest of the line to be ignored.
   Empty lines and lines containing only a comment are also ignored.
   Default: '#!'.
:param sepchar:
   a string containing the column separation characters to be used.
   Columns are separated by any combination of these characters.
   Default: ', \\\\t'.
:param lines:
   a two-element tuple or list specifying a range of lines to be
   read.  Line numbers are counted from one and the range is
   inclusive.  So (1,10) specifies the first 10 lines of a file. 
   Comment lines are included in the count.  If any element of
   the tuple or list is zero, this limit is ignored. So (1,0)
   specifies the whole file, just like the default None. 
   Default: all lines.
:param bad:
   a number to be substituted for any field which cannot be
   decoded as a number. The default None causes a :exc:`ValueError`
   exception to be raised in such cases.
:param segsep:
   a string containing the segment separation characters. If any of
   these characters is present in a comment line, this comment block
   is taken as the end of the current segment. The default None indicates
   that every comment block will separate segments.
:returns:
   a tuple containing 1) a NumPy array containing the selected data from
   the table file, and 2) a list of slice objects, one for every segment.
:raises:
   :exc:`IOError`, when the file cannot be opened.

   :exc:`IndexError`, when a line with an inconsistent number of fields is
   encountered in the input file.

   :exc:`ValueError`: when a field cannot be decoded as a number and
   no alternative value was specified.

"""
   cdef FILE *f
   cdef char line[32768]
   cdef char *tokens[10000]
   cdef char *curline, *token, *comstart, *csep, *c_segsep=NULL
   cdef char *endptr
   cdef int i, column, ncols=0, lineno=0, lstart=0, lend=0
   cdef npy_intp nvalues=0
   cdef int filesize, maxitems=0, badflag
   cdef int *segments=NULL, maxseg=0, nseg=0, segfirst=0, segflag
   cdef double *data=NULL, badvalue=0.0
   setlocale(LC_NUMERIC, 'C')              # enforce decimal point, not comma
   f = fopen(filename, "r")
   if f==NULL:
      raise IOError, 'cannot open %s' % filename
   fseek(f, 0, SEEK_END)
   filesize = ftell(f)
   fseek(f, 0, SEEK_SET)
   sepchar += '\n'
   csep = sepchar
   badflag = bad is not None
   if badflag:
      badvalue = bad
   if lines:
      lstart, lend = lines
   if segsep is not None:
      c_segsep = segsep
   segments = <int*>malloc(sizeof(int))
   while fgets(line, 32768, f) != NULL:
      lineno += 1
      if lstart>0 and lineno<lstart:
         continue
      if lend>0 and lineno>lend:
         break
      segflag = (c_segsep == NULL or strpbrk(line, c_segsep) != NULL)
      comstart = strpbrk(line, comchar)
      if comstart!=NULL:
         comstart[0] = 0
      column = 0
      curline = line
      while 1:
         token = strtok(curline, csep)
         curline = NULL
         if token != NULL:
            tokens[column] = token
            column += 1
         else:
            if column>0:
               segfirst = 1
               if ncols>0:
                  if ncols!=column:
                     free(segments)
                     fclose(f)
                     raise IndexError, \
                        '%s, line %d: row width error' % (filename, lineno)
               else:
                  ncols = column
                  maxitems = <int>(1.3*ncols*filesize/strlen(line))+ncols
                  data = <double*>malloc(maxitems*sizeof(double))
               for i from 0 <= i <column:
                  if nvalues>=maxitems:
                     maxitems = <int>(1.3*maxitems)+ncols
                     data = <double*>realloc(data, maxitems*sizeof(double))
                  data[nvalues] = strtod(tokens[i], &endptr)
                  if endptr[0]!=0:
                     if badflag:
                        data[nvalues] = badvalue
                     else:
                        fclose(f)
                        free(segments)
                        raise ValueError, \
                         '%s, line %d, column %d: invalid number "%s"' \
                         % (filename, lineno, i+1, tokens[i])
                  nvalues += 1
            else:
               if segflag:
                  if segfirst:
                     if nseg >= maxseg:
                        maxseg += 1000
                        segments = <int*>realloc(segments, (maxseg+1)*sizeof(int))
                     segments[nseg] = nvalues/ncols  # number of rows up to here
                     nseg += 1
                     segfirst = 0
            break

   if segfirst:
      segments[nseg] = nvalues/ncols  # final number of rows
      nseg += 1

   seglist = []
   segstart = 0
   for i from 0 <= i <nseg:
      segend = segments[i]
      seglist.append(slice(segstart, segend))
      segstart = segend

   fclose(f)
   if not data:
      raise IndexError, '%s: no lines of data read' % filename
   data = <double*>realloc(data, nvalues*sizeof(double))
   array = PyArray_SimpleNewFromData(1, &nvalues, NPY_DOUBLE, data)
   array.shape = (nvalues/ncols, ncols)
   return (array, seglist)

__version__ = '1.2'
