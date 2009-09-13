from kapteyn import wcsgrat
from matplotlib import pylab as plt

header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 100,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :80.0, 'CRPIX1' : 1, 
          'CUNIT1' : 'arcmin', 'CDELT1' : -0.5,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' :400.0, 'CRPIX2' : 1, 
           'CUNIT2' : 'arcmin', 'CDELT2' : 0.5,
          'CROTA2' : 30.0
         }

grat = wcsgrat.Graticule(header)
xmax = grat.pxlim[1]+0.5; ymax = grat.pylim[1]+0.5
ruler = grat.ruler(xmax,0.5, xmax, ymax, lambda0=0.5, step=5.0/60.0, 
                   fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
                   fliplabelside=True, color='r')
# The wcs input methods to convert between pixels and world
# coordinates expect input in degrees whatever the units in the
# header are (arcsec, arcmin).
ruler2 = grat.ruler(60/60.0,390/60.0,60/60.0,420/60.0, 
                    lambda0=0.0, step=5.0/60, world=True, 
                    fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", color='r')

fig = plt.figure(figsize=(7,7))
frame = fig.add_subplot(1,1,1)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add(grat)
gratplot.add(ruler)
gratplot.add(ruler2)

plt.show()
