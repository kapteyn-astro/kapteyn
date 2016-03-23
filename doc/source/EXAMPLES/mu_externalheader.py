from kapteyn import maputils
from matplotlib import pylab as plt
import numpy

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

# Overrule the header value for pixel size in y direction
header['CDELT2'] = 0.3*abs(header['CDELT1'])
fitsobj = maputils.FITSimage(externalheader=header)
figsize = fitsobj.get_figsize(ysize=7, cm=True)

fig = plt.figure(figsize=figsize)
print("Figure size x, y in cm:", figsize[0]*2.54, figsize[1]*2.54)
frame = fig.add_subplot(1,1,1)
mplim = fitsobj.Annotatedimage(frame)
gr = mplim.Graticule()
mplim.plot()

plt.show()
