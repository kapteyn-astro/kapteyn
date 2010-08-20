#!/bin/csh
/bin/csh UTILS/makepdf.csh
cd DISTRIBUTION
\cp ../SPHINX/LATEX/kapteyn.pdf doc/
python setup.py sdist --formats=gztar,zip
cd ..
