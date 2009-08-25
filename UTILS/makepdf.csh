#!/bin/csh
cd SPHINX
setenv PYTHONPATH ../PACKAGE
sphinx-build -E -b latex . LATEX
cd LATEX
make
cd ../..
