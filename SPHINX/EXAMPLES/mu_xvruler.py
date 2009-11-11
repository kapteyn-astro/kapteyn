from kapteyn import maputils
from matplotlib import pylab as plt

# Open FITS file and get header
f = maputils.FITSimage('ngc6946.fits')
f.set_imageaxes(3,2)

fig = plt.figure(figsize=f.get_figsize(xsize=15, cm=True))
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)

# Velocity - Dec
grat = mplim.Graticule()

xmax = grat.pxlim[1]+0.5; ymax = grat.pylim[1]+0.5
ruler = grat.Ruler(xmax,0.5, xmax, ymax, lambda0 = 0.5, step=5.0/60.0, 
                   fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
                   fliplabelside=True)
ruler.setp_line(lw=2, color='r')
ruler.setp_labels(clip_on=True, color='r')

ruler2 = grat.Ruler(0.5,0.5, xmax, ymax, lambda0 = 0.5, step=5.0/60.0, 
                    fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
                    fliplabelside=True)
ruler2.setp_line(lw=2, color='b')
ruler2.setp_labels(clip_on=True, color='b')
grat.setp_plotaxis("right", label="Offset (Arcsec)", visible=True)

mplim.plot()
mplim.interact_writepos()

plt.show()

