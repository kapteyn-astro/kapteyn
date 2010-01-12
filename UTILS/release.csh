#!/bin/csh
if ( ${#argv} != 1 ) then
   echo "usage: release.csh <release home directory>"
   exit
else
   set destdir = ${argv[1]}
endif

UTILS/makepack.csh --clean

setenv PYTHONPATH ../PACKAGE/

mkdir ${destdir} >& /dev/null

cd SPHINX
sphinx-build -E -b html . ${destdir}
cd ..

UTILS/makedist.csh

\rm ${destdir}/kapteyn*tar.gz >& /dev/null
\mv DISTRIBUTION/dist/kapteyn*tar.gz ${destdir}
\mv DISTRIBUTION/doc/kapteyn.pdf ${destdir}
ln -s ${destdir}/kapteyn*tar.gz ${destdir}/kapteyn.tar.gz
