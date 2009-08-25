from kapteyn import wcsgrat
from matplotlib import pylab as plt

header = {'NAXIS'  : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' :'RA---TAN',
          'CRVAL1' : 0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' :'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

grat = wcsgrat.Graticule(header)
x1 = 10; y1 = 1
x2 = 10; y2 = grat.pylim[1]

ruler1 = grat.ruler(x1, y1, x2, y2, 0.0, 1.0)
x1 = x2 = grat.pxlim[1]
ruler2 = grat.ruler(x1, y1, x2, y2, 0.5, 2.0, fmt='%3d', mscale=-2.5, fliplabelside=True)
ruler2.setp_labels(ha='left', va='center')
ruler3 = grat.ruler(23*15,30,22*15,15, 0.5, 2, world=True, fmt=r"$%6.0f^\prime$", 
                    fun=lambda x: x*60.0, mscale=4.5)
ruler3.setp_labels(color='r')
ruler4 = grat.ruler(1,800,800,800, 0.5, 2, fmt="%4.1f", addangle=90)
fig = plt.figure(figsize=(7,7))
frame = fig.add_axes(grat.axesrect, aspect=grat.aspectratio, adjustable='box', label='1')
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.add(ruler1)
gratplot.add(ruler2)
gratplot.add(ruler3)
gratplot.add(ruler4)
gratplot.plot()

plt.show()

