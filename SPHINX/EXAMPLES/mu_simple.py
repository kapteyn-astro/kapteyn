from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
mplim = f.Annotatedimage()
im = mplim.Image()
mplim.plot()

plt.show()

