from kapteyn import maputils
from matplotlib import pylab as plt

# Make plot window wider if you don't see toolbar info

# Open FITS file and get header
f = maputils.FITSimage('mclean.fits')
f.set_imageaxes('freq','dec')
f.set_limits(pxlim=(35,45))

fig = plt.figure(figsize=f.get_figsize(ysize=12, cm=True))
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)

grat = annim.Graticule(spectrans='WAVE')
grat.setp_ticklabel(plotaxis='bottom', fun=lambda x: x*100, fmt="$%.3f$")
grat.setp_axislabel(plotaxis='bottom', label="Wavelength (cm)")
grat.setp_gratline(wcsaxis=(0,1), color='g')

annim.Pixellabels(plotaxis=("right","top"), gridlines=False)
annim.plot()
annim.interact_toolbarinfo()
plt.show()
