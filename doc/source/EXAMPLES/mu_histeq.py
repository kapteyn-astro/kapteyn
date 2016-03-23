from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

tpos = 1.02
f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(200,300), pylim=(200,300))
print(f.pxlim, f.pylim)

fig = plt.figure()
frame = fig.add_subplot(1,1,1)
t = frame.set_title("Original")
t.set_y(tpos)

mplim = f.Annotatedimage(frame)
ima = mplim.Image()
mplim.Pixellabels()
mplim.plot()

fig2 = plt.figure()
frame2 = fig2.add_subplot(1,1,1)
t = frame2.set_title("Histogram equalized")
t.set_y(tpos)

mplim2 = f.Annotatedimage(frame2)
ima2 = mplim2.Image(visible=True)
ima2.histeq()
mplim2.Pixellabels()
mplim2.plot()


fig3 = plt.figure()
frame3 = fig3.add_subplot(1,1,1)
t = frame3.set_title("Colors with LogNorm")
t.set_y(tpos)

mplim3 = f.Annotatedimage(frame3)
ima3 = mplim3.Image(norm=LogNorm())
mplim3.Pixellabels()
mplim3.plot()

"""
fig4 = plt.figure()
frame4 = fig4.add_subplot(1,1,1)
mplim4 = f.Annotatedimage(frame4)
ima4 = mplim4.Image()
ima4.blur_image(2,2)
mplim4.Pixellabels()
mplim4.plot()
"""


mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()

plt.show()

