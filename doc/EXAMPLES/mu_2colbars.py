from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")

fig = plt.figure()
frame1 = fig.add_subplot(1,2,1)

mplim = fitsobj.Annotatedimage(frame1, cmap="spectral")
mplim.Image()
units = r'$ergs/(sec.cm^2)$'
colbar = mplim.Colorbar(fontsize=6, orientation='horizontal')
colbar.set_label(label=units, fontsize=24)
mplim.plot()
mplim.interact_imagecolors()

frame2 = fig.add_subplot(1,2,2)
fitsobj = maputils.FITSimage("I1.fits")
#fitsobj.set_limits((30,150),(30,150))
mplim = fitsobj.Annotatedimage(frame2, cmap="spectral")
mplim.Image()
colbar = mplim.Colorbar(fontsize=6, orientation='horizontal')
mplim.plot()
mplim.interact_imagecolors()

plt.show()
