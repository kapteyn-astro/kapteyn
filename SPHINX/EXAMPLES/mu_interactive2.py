"""Show interaction options"""
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
#f.set_limits((100,500),(200,400))

fig = plt.figure(figsize=(9, 7))
frame = fig.add_subplot(1, 1, 1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="mousse.lut")
mplim.cmap.set_bad('w')
ima = mplim.Image()
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

