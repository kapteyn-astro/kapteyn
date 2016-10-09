from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)

mycmlist = glob.glob("/home/gipsy/dat/lut/*.lut")
maputils.cmlist.add(mycmlist)
print("Colormaps: ", maputils.cmlist.colormaps)

mplim = f.Annotatedimage(frame)
ima = mplim.Image(cmap="mousse.lut")
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

