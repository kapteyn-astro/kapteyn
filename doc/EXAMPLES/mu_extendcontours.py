from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
limy = limx=(160,360)
f.set_limits(limx,limy)
fig = plt.figure(figsize=(10,10))

frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours(filled=True)
mplim.Colorbar(clines=True, fontsize=8) # show only cont. lines
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

plt.show()
