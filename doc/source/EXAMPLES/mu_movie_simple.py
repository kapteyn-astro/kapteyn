#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib.pyplot import figure, show

fig = figure()
myCubes = maputils.Cubes(fig, toolbarinfo=True, printload=False,
                              helptext=False, imageinfo=True)

# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('ngc6946.fits')
naxis3 = fitsobject.hdr['NAXIS3']
# Note that slice positions follow FITS syntax, i.e. start at 1
slicepos = list(range(1,naxis3+1))

frame = fig.add_subplot(1,1,1)
vmin, vmax = fitsobject.get_dataminmax()
cube = myCubes.append(frame, fitsobject, (1,2), slicepos,
                      vmin=vmin, vmax=vmax, hasgraticule=True)

show()
