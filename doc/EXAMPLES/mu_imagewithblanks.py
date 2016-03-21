from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

f = maputils.FITSimage("blanksetmin32.fits")
#f = maputils.FITSimage("blankset16.fits")
f.set_imageaxes(1,2)

fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="mousse.lut", blankcolor='w')
mplim.Image()
#mplim.Image()
#mplim.set_blankcolor('c')
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

