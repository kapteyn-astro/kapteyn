#!/bin/csh
cd DISTRIBUTION
\rm -rf build
python setup.py install --install-lib ../PACKAGE
cd ..
