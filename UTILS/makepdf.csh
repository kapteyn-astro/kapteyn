#!/bin/csh
cd SPHINX
if ( ${#argv} == 1 ) then
   set flag = ${argv[1]}
else
   set flag = ''
endif
setenv PYTHONPATH ../PACKAGE
mkdir LATEX >& /dev/null
sphinx-build ${flag} -b latex . LATEX
cd LATEX
\cp ../sphinx.sty .    # modified version (see comment in file)
make
cd ../..
