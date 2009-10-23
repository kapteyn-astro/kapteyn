from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(50,440), pylim=(50,450))

fig = plt.figure(figsize=(12,8))
frame = fig.add_subplot(1,1,1)
fig.text(0.5, 0.96, "Two graticules", horizontalalignment='center',
         fontsize=14, color='r')

mplim = f.Annotatedimage(frame)
ima = mplim.Image(visible=False)
mplim.Pixellabels()

cont = mplim.Contours(levels=range(8000,14000,1000))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=2)

#cont2 = mplim.Contours(levels=(10000,11000,12000))
#cont2.setp_label(fontsize=14)

cb = mplim.Colorbar(clines=True, orientation='vertical', fontsize=8)

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

