from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((180,344), (180,344))

fig = plt.figure(figsize=(4,4))
frame = fig.add_axes([0,0,1,1])

annim = fitsobj.Annotatedimage(frame, cmap="spectral", clipmin=10000, clipmax=15500)
annim.Image(interpolation='spline36')
print("clip min, max:", annim.clipmin, annim.clipmax)
annim.plot()

plt.show()

