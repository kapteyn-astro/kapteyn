from kapteyn import maputils
from matplotlib import pylab as plt

# Open FITS file and get header
f = maputils.FITSimage('mclean.fits')
f.set_imageaxes(3,2)
f.set_limits(pxlim=(35,45))

fig = plt.figure(figsize=f.get_figsize(xsize=15, cm=True))
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)
mplim.Pixellabels(plotaxis=("right","top"))

grat = mplim.Graticule(spectrans='WAVE-???')
grat.setp_tick(plotaxis=1, fun=lambda x: x*100, fmt="%.3f")
grat.setp_plotaxis(1, label="Wavelength (cm)")

mplim.plot()

plt.show()
