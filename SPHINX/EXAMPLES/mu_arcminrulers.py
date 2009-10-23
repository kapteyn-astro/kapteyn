from kapteyn import maputils
from matplotlib import pyplot as plt

header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 100,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' : 80.0, 'CRPIX1' : 1, 
          'CUNIT1' : 'arcmin', 'CDELT1' : -0.5,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 400.0, 'CRPIX2' : 1, 
          'CUNIT2' : 'arcmin', 'CDELT2' : 0.5,
          'CROTA2' : 30.0
         }

f = maputils.FITSimage(externalheader=header)

fig = plt.figure(figsize=(7,7))
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)
grat = mplim.Graticule()

# Use pixel limits attributes of the FITSimage object

xmax = mplim.pxlim[1]+0.5; ymax = mplim.pylim[1]+0.5
grat.Ruler(xmax,0.5, xmax, ymax, lambda0=0.5, step=5.0/60.0, 
           fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
           fliplabelside=True, color='r')

# The wcs methods that convert between pixels and world
# coordinates expect input in degrees whatever the units in the
# header are (e.g. arcsec, arcmin).
grat.Ruler(60/60.0,390/60.0,60/60.0,420/60.0, 
           lambda0=0.0, step=5.0/60, world=True, 
           fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", color='r')
mplim.plot()

plt.show()
