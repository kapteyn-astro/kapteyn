#!/bin/csh
if ( ${#argv} != 1 ) then
   echo "usage: clone.csh <destination directory>"
   exit
else
   set destdir = ${argv[1]}
endif
set scriptdir = `dirname $0`

echo ${scriptdir}
hg clone ${scriptdir}/../../KAPTEYN ${destdir}
cp -R ${scriptdir}/../../KAPTEYN/DISTRIBUTION/src/wcslib* ${destdir}/DISTRIBUTION/src/
