from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((180,344), (180,344))

fig = plt.figure(figsize=(6,6))
frame = fig.add_axes([0,0,1,1])

mplim = fitsobj.Annotatedimage(frame, cmap="spectral", clipmin=10000, clipmax=15500)
im = mplim.Image(interpolation='spline36')
print "min, max:", mplim.clipmin, mplim.clipmax
mplim.plot()

plt.show()

