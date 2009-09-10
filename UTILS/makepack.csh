#!/bin/csh
cd DISTRIBUTION
if ( ${#argv} >= 1 ) then
   if (${argv[1]} == "--clean") then
      echo cleaning build directory
      \rm -rf build
   endif
endif
python setup.py install --install-lib ../PACKAGE
cd ..
