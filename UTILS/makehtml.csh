#!/bin/csh
cd SPHINX
if ( ${#argv} == 1 ) then
   set flag = ${argv[1]}
else
   set flag = ''
endif
setenv PYTHONPATH ../PACKAGE/
sphinx-build ${flag} -b html  . HTML
cd ..
