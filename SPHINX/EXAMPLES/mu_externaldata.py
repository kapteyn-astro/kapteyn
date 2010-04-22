from kapteyn import maputils
from matplotlib import pylab as plt
import numpy

header = {'NAXIS'  : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' : 0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

nx = header['NAXIS1']
ny = header['NAXIS2']
sizex1 = nx/2.0; sizex2 = nx - sizex1
sizey1 = nx/2.0; sizey2 = nx - sizey1
x, y = numpy.mgrid[-sizex1:sizex2, -sizey1:sizey2]
edata = numpy.exp(-(x**2/float(sizex1*10)+y**2/float(sizey1*10)))

f = maputils.FITSimage(externalheader=header, externaldata=edata)
f.writetofits()
fig = plt.figure(figsize=(6,5))
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
mplim = f.Annotatedimage(frame, cmap='pink')
mplim.Image()
gr = mplim.Graticule()
gr.setp_gratline(color='y')
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()
