from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(50,440), pylim=(50,450))

fig = plt.figure(figsize=(8,5.5))
frame = fig.add_axes((0.05, 0.1, 0.8, 0.7))
fig.text(0.5, 0.96, "Combination of plot objects", horizontalalignment='center',
         fontsize=14, color='r')

mplim = f.Annotatedimage(frame, clipmin=3000, clipmax=15000)
ima = mplim.Image(visible=False)
mplim.Pixellabels()

cont = mplim.Contours(levels=range(8000,14000,1000))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=2)

cb = mplim.Colorbar(clines=False, orientation='vertical', fontsize=8)

gr = mplim.Graticule()
gr.Insidelabels()

gr2 = mplim.Graticule(deltax=7.5/60, deltay=5.0/60,
                      skyout="galactic", 
                      visible=True)
gr2.setp_plotaxis(("top","right"), label="Galactic l,b", 
                  mode=maputils.native, color='g', visible=True)
gr2.setp_tick(wcsaxis=(0,1), color='g')
gr2.setp_gratline(wcsaxis=(0,1), color='g')
gr2.setp_plotaxis(("left","bottom"), mode=maputils.noticks, visible=False)
gr2.Ruler(150,100,150,330, step=1/60.0)
gr2.Ruler(102,59+50/60.0, 102+7.5/60,59+50/60.0,  world=True, step=1/60.0, color='r')

mplim.plot()
plt.show()

