from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

# Get a header and change some values
f = maputils.FITSimage("m101.fits")
header = f.hdr
header['CDELT1'] = 0.1
header['CDELT2'] = 0.1
header['CRVAL1'] = 285
header['CRVAL2'] = 20

# Use the changed header as external source for new object
f = maputils.FITSimage(externalheader=header)
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()

fn = 'WDB/smallworld.txt'
# Note that in this file the latitudes are in the first column
# (column 0). And the longitudes in the second (column=1)
xp, yp = annim.positionsfromfile(fn, 's', cols=[1,0])
annim.Marker(x=xp, y=yp, mode='pixels', marker=',', color='b')
annim.plot()
frame.set_title("Markers in the Carribbean")

plt.show()