from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
ima = mplim.Image()
mplim.Pixellabels()
mplim.plot()

mplim.interact_toolbarinfo()

plt.show()
