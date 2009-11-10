#!/usr/bin/env python
from kapteyn import wcsgrat, maputils, ellinteract
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()


movieimages = maputils.MovieContainer()


# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')

#ch = [10,15,20,25,30,35]
#ch = range(15,85)
ch = [20, 30]
count = 0
vmin, vmax = fitsobject.get_dataminmax()
print "Vmin, Vmax:", vmin, vmax
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
#   fitsobject.set_limits()
   frame = fig.add_subplot(1,2,count+1)
   # Create an image to be used in Matplotlib
   mplim = fitsobject.Annotatedimage(frame)
   mplim.Image()
   count += 1
   # Draw the graticule lines and plot WCS labels
   mplim.Graticule()
   mplim.plot()
   movieimages.append(mplim)


# movieimages.movie_events()


shapes = ellinteract.Shapecollection(movieimages.mplim, fig, wcs=True)

plt.show()

