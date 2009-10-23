#!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt
from kapteyn.mplutil import KeyPressFilter

KeyPressFilter.allowed = ['f', 'g']

# Get connected to Matplotlib
fig = plt.figure()
frame = fig.add_subplot(1,1,1)

#Create a container to store the annotated images
movieimages = maputils.MovieContainer()

# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('ngc6946.fits')

# Get a the range of channels in the data cube
n3 = fitsobject.hdr['NAXIS3']
ch = range(1,n3)
vmin, vmax = fitsobject.globalminmax()
print "Vmin, Vmax of data in cube:", vmin, vmax
cmap = None

# Start to build and store the annotated images
first = True
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
   # Set limits as in: fitsobject.set_limits(pxlim=(150,350), pylim=(200,350))
   mplim = fitsobject.Annotatedimage(frame, cmap=cmap, clipmin=vmin, clipmax=vmax)
   mplim.Image()
   mplim.plot()
   if first:
      mplim.interact_imagecolors()
      cmap = mplim.cmap
   movieimages.append(mplim, visible=first)
   first = False

movieimages.movie_events()

# Draw the graticule lines and plot WCS labels
grat = mplim.Graticule()
grat.plot(frame)

plt.show()

