#!/bin/csh
cd SPHINX
if ( ${#argv} == 1 ) then
   set flag = ${argv[1]}
else
   set flag = ''
endif
setenv PYTHONPATH ../PACKAGE
mkdir LATEX
sphinx-build ${flag} -b latex . LATEX
cd LATEX
make
cd ../..
