.. automodule:: kapteyn.tabarray

   Example
   --------

   Suppose you have a file with catheti data from right-angled triangles
   and you want to compute the hypotenuses and write the result to a second
   file.  The input file may be as follows:

   .. highlight:: python   

   ::

      # Triangle data
      #
       3.0   4.0 ! classic example
       4.1   3.6
      10.0  10.0

   Then the following simple script will do the job:

   ::

      #!/usr/bin/env python
      import numpy
      from kapteyn.tabarray import tabarray

      x,y = tabarray('triangles.txt').columns()
      tabarray([x,y,numpy.sqrt(x*x+y*y)]).writeto('outfile.txt')

   leaving the following result in the output file:

   ::

          3      4      5
          4.1    3.6    5.45619
         10     10     14.1421
 
