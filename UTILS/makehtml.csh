#!/bin/csh
cd SPHINX
setenv PYTHONPATH ../PACKAGE/
sphinx-build -E -b html  . HTML
cd ..
