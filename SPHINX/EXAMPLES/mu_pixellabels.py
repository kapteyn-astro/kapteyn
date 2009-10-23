from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
#ima = mplim.Image(visible=False)
#mplim.Pixellabels(plotaxis=("bottom", "right"), color="r")
mplim.Pixellabels(plotaxis="bottom", color="r")
mplim.Pixellabels(plotaxis="right", color="b", markersize=10)
mplim.Pixellabels(plotaxis="top", color="g", markersize=-10)

mplim.plot()
plt.show()

