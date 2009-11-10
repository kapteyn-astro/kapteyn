from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
limy = limx=(160,360)
f.set_limits(limx,limy)
rows = 3
cols = 2

fig = plt.figure(figsize=(8,10))

frame = fig.add_subplot(rows,cols,1)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours()
mplim.Colorbar(clines=True, fontsize=8,
               linewidths=3, visible=False) # show only cont. lines
mplim.plot()
# Levels only known after plotted
print "Proposed levels:", cont.clevels

frame = fig.add_subplot(rows,cols,2)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours(filled=True)
mplim.Colorbar(clines=True, fontsize=8) # show only cont. lines
mplim.plot()

frame = fig.add_subplot(rows,cols,3)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image()
cont = mplim.Contours(colors='w', linewidths=2)
mplim.Colorbar(clines=True, ticks=(4000,8000,12000))
mplim.plot()
mplim.interact_imagecolors()

frame = fig.add_subplot(rows,cols,4)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image()
cont = mplim.Contours(levels=(4000,6000,8000,10000,12000), 
                      colors=('r','b','y','g', 'c'))
mplim.Colorbar(clines=True, ticks=(4000,8000,12000), linewidths=2)
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

frame = fig.add_subplot(rows,cols,5)
mplim = f.Annotatedimage(frame, cmap="mousse.lut")
mplim.Image()
cont = mplim.Contours()
mplim.Colorbar(clines=True, orientation="horizontal", ticks=(4000,8000,12000))
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

# With given levels
frame = fig.add_subplot(rows,cols,6)
levels = (10000,11000,12000,13000)
mplim = f.Annotatedimage(frame, cmap="mousse.lut", 
                         clipmin=min(levels)-500,
                         clipmax=max(levels)+500)
mplim.Image()
cont = mplim.Contours(levels=levels)
mplim.Colorbar(clines=True, orientation="horizontal", 
               ticks=levels)
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

plt.show()

