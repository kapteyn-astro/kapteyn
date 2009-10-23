from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(6,6))
frame = fig.add_axes([0,0,1,1])

mplim = f.Annotatedimage(frame, cmap="spectral", clipmin=1507, clipmax=10000)
im = mplim.Image()
print "min, max:", mplim.clipmin, mplim.clipmax
mplim.plot()

plt.show()

