from kapteyn import wcsgrat
from matplotlib import pylab as plt
import pyfits

hdulist = pyfits.open('ngc6946.fits')
header = hdulist[0].header
fig = plt.figure(figsize=(20/2.54, 25/2.54))
fig.subplots_adjust(hspace=0.4)
labelx = -0.10         # Fixed position in x for all y labels
frame = fig.add_subplot(4,1,1)
frame.yaxis.set_label_coords(labelx, 0.5)

# Spatial map
grat = wcsgrat.Graticule(header)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.plot()

# Velocity - Dec
frame2 = fig.add_subplot(4,1,2)
frame2.yaxis.set_label_coords(labelx, 0.5)
grat2 = wcsgrat.Graticule(header, axnum=(3,2))
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame2)
gratplot.add(grat2)
gratplot.plot()

# Velocity - Dec (Version without offsets)
frame2a = fig.add_subplot(4,1,3)
frame2a.yaxis.set_label_coords(labelx, 0.5)
grat2a = wcsgrat.Graticule(header, axnum=(3,2), offsety=False)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame2a)
gratplot.add(grat2a)
gratplot.plot()

# Velocity - R.A.
frame3 = fig.add_subplot(4,1,4)
frame3.yaxis.set_label_coords(labelx, 0.5)
grat3 = wcsgrat.Graticule(header, axnum=(3,1))
grat3.setinsidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10, color='r')
grat3.setinsidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame3)
gratplot.add(grat3)
gratplot.plot()

plt.savefig('fig2.axnumdemo.png')
plt.show()

