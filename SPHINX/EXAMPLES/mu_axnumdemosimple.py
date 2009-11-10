from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
mplim = fitsobj.Annotatedimage()
graticule = mplim.Graticule()
mplim.plot()

plt.show()

