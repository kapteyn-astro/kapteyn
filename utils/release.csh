#!/bin/csh
if ( ${#argv} != 1 ) then
   echo "usage: release.csh <release home directory>"
   exit
else
   set destdir = ${argv[1]}
endif

UTILS/makepack.csh --clean                    # build new private package

setenv PYTHONPATH ../PACKAGE/

mkdir ${destdir} >& /dev/null                 # try to make home directory

cd SPHINX
tar cfz examples.tar.gz EXAMPLES/*.py EXAMPLES/*.dat EXAMPLES/*.fits EXAMPLES/WDB
sphinx-build -E -b html . ${destdir}          # make html doc
\mv examples.tar.gz ${destdir}/_downloads/    # example sources
cd ..

UTILS/makedist.csh                            # make distro, including pdf doc

mkdir ${destdir}/OLD >& /dev/null
\rm ${destdir}/kapteyn.tar.gz >& /dev/null                  # old symbolic link
\mv ${destdir}/kapteyn*tar.gz ${destdir}/OLD/ >& /dev/null  # keep old distro
\mv DISTRIBUTION/dist/kapteyn*tar.gz ${destdir}             # new tar.gz distro
\mv DISTRIBUTION/dist/kapteyn*.zip ${destdir}               # new zip distro
\mv DISTRIBUTION/doc/kapteyn.pdf ${destdir}                 # new pdf doc
ln -s ${destdir}/kapteyn*tar.gz ${destdir}/kapteyn.tar.gz   # new symbolic link
echo "AddDefaultCharset utf-8" > ${destdir}/.htaccess
echo "Options Indexes\nIndexOptions FancyIndexing VersionSort XHTML" > ${destdir}/OLD/.htaccess
