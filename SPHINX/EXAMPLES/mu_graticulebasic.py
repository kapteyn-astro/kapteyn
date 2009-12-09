from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('m101.fits')
annim = fitsobj.Annotatedimage()
grat = annim.Graticule()
annim.plot()

plt.show()

