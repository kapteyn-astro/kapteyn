#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt
   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
fitsobject.set_limits(promptfie=maputils.prompt_box)
fitsobject.set_skyout(promptfie=maputils.prompt_skyout)

   
# Get connected to Matplotlib
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
   
# Create an image to be used in Matplotlib
mplim = fitsobject.Annotatedimage(frame)
mplim.Image()
mplim.Graticule()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()
