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

fig = plt.figure()
frame = fig.add_axes([0.1,0.15,0.8,0.75])
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat.setp_ticklabel(plotaxis='bottom', rotation=90)
grat.setp_ticklabel(fmt='s') # Suppress the seconds in all labels

# Use pixel limits attributes of the FITSimage object

xmax = annim.pxlim[1]+0.5; ymax = annim.pylim[1]+0.5
annim.Ruler(x1=xmax, y1=0.5, x2=xmax, y2=ymax, lambda0=0.5, step=5.0/60.0, 
            fun=lambda x: x*60.0, fmt=r"%4.0f^\prime", 
            fliplabelside=True, color='r')

# The wcs methods that convert between pixels and world
# coordinates expect input in degrees whatever the units in the
# header are (e.g. arcsec, arcmin).
annim.Ruler(x1=60/60.0, y1=390/60.0, x2=60/60.0, y2=420/60.0, 
            lambda0=0.0, step=5.0/60, world=True, 
            fun=lambda x: x*60.0, fmt=r"%4.0f^\prime", color='g')

annim.Ruler(pos1='0h3m30s 6d30m', pos2='0h3m30s 7d0m', 
            lambda0=0.0, step=5.0, 
            units='arcmin', color='c')

annim.plot()
plt.show()
