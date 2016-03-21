#!/bin/csh
if ( ${#argv} < 1 ) then
   echo "usage: clone.csh <destination directory>"
   exit
else
   set destdir = ${argv[1]}
endif

set scriptdir = `dirname $0`

hg clone ${scriptdir}/../../KAPTEYN ${destdir}
cp -R ${scriptdir}/../../KAPTEYN/DISTRIBUTION/src/wcslib* ${destdir}/DISTRIBUTION/src/

if ( ${#argv} == 2 && ${argv[2]} == "build") then
   cd ${destdir}
   UTILS/makepack.csh
endif
