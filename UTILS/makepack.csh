#!/bin/csh
cd DISTRIBUTION
\rm -rf build >& /dev/null
\rm -rf ../PACKAGE >& /dev/null
python setup.py install --install-lib ../PACKAGE
setenv PYTHONPATH ../PACKAGE
python ../UTILS/makebib.py
cd ..
