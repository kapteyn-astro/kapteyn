from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

mplim = f.Annotatedimage()
cont = mplim.Contours()
mplim.plot()

print "Levels=", cont.clevels

plt.show()

