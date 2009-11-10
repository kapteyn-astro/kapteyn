from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('example1test.fits')

fig = plt.figure(figsize=(5,5))
frame = fig.add_axes([0.1, 0.1, 0.8, 0.8])
mplim = fitsobj.Annotatedimage(frame)
mplim.set_aspectratio(1.2)
grat = mplim.Graticule()

mplim.plot()

plt.show()
