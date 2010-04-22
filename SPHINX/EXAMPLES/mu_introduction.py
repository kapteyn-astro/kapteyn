#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
fitsobject.set_limits(promptfie=maputils.prompt_box)
fitsobject.set_skyout(promptfie=maputils.prompt_skyout)
clipmin, clipmax = maputils.prompt_dataminmax(fitsobject)
   
# Get connected to Matplotlib
fig = plt.figure()
frame = fig.add_subplot(1,1,1)

# Create an image to be used in Matplotlib
annim = fitsobject.Annotatedimage(frame, clipmin=clipmin, clipmax=clipmax)
annim.Image()
annim.Graticule()
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

plt.show()
