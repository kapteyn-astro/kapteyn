from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage('example1test.fits')

# Use the figure size as suggested by the Graticule object
fig = plt.figure(figsize=f.get_figsize())
frame = fig.add_axes([0.1,0.1,0.8,0.8])
mplim = f.Annotatedimage(frame)
grat = mplim.Graticule()

mplim.plot()

plt.show()
