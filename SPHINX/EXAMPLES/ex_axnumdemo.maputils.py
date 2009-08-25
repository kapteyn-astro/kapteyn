#!/usr/bin/env python
import wcsgrat
import pyfits
import maputils
from matplotlib import pyplot as plt

def fx(x):
  return x/1000.0

usetex = True
plt.rc('text', usetex=usetex)
fig = plt.figure(figsize=(20.0/2.54,25/2.54))

fitsimage = maputils.FITSimage('rense.fits')

# Fist plot spatial map
fitsimage.setaxes(1,2, grids=30)
grat1 = wcsgrat.Graticule(fitsimage=fitsimage)
frame1 = fig.add_subplot(3,1,1)
image1 = maputils.MPLimage(frame1, fitsimage)
image1.plotim(interpolation="nearest")
gratplot1 = wcsgrat.Plotversion('matplotlib', fig, frame1)
gratplot1.addgraticule(grat1, labelsintex=usetex)
gratplot1.plot()


# Second plot VELO-DEC
fitsimage.setaxes(3,2, grids=51)
grat2 = wcsgrat.Graticule(fitsimage=fitsimage)

frame2 = fig.add_subplot(3,1,2)
image2 = maputils.MPLimage(frame2, fitsimage)
image2.plotim(interpolation="nearest")
gratplot2 = wcsgrat.Plotversion('matplotlib', fig, frame2)
gratplot2.addgraticule(grat2, labelsintex=usetex)
gratplot2.plot()


# Third plot VELO-RA
fitsimage.setaxes(3,1, grids=51)
grat3 = wcsgrat.Graticule(fitsimage=fitsimage)

frame3 = fig.add_subplot(3,1,3)
image3 = maputils.MPLimage(frame3, fitsimage)
image3.plotim(interpolation="nearest")
gratplot3 = wcsgrat.Plotversion('matplotlib', fig, frame3)
gratplot3.addgraticule(grat3, labelsintex=usetex)
gratplot3.plot()

plt.show()
