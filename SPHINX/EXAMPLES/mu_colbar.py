from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")

mplim = fitsobj.Annotatedimage(cmap="spectral")
mplim.Image()
units = r'$ergs/(sec.cm^2)$'
colbar = mplim.Colorbar(fontsize=8)
colbar.set_label(label=units, fontsize=24)
mplim.plot()
plt.show()
