from kapteyn import maputils
from matplotlib import pylab as plt

# Open FITS file and get header
f = maputils.FITSimage('ngc6946.fits')
f.set_imageaxes(3,2)   # X axis is velocity, y axis is declination

fig = plt.figure(figsize=f.get_figsize(xsize=15, cm=True))
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)

# Velocity - Dec
grat = annim.Graticule()
grat.setp_axislabel("right", label="Offset (Arcmin.)", visible=True)

xmax = annim.pxlim[1]+0.5; ymax = annim.pylim[1]+0.5
ruler = annim.Ruler(x1=xmax, y1=0.5, x2=xmax, y2=ymax, 
                    lambda0 = 0.5, step=5.0/60.0, 
                    fun=lambda x: x*60.0, fmt="%4.0f^\prime", 
                    fliplabelside=True)
ruler.setp_line(lw=2, color='r')
ruler.setp_label(color='r')

ruler2 = annim.Ruler(x1=0.5, y1=0.5, x2=xmax, y2=ymax, lambda0 = 0.5, 
                     step=5.0/60.0, 
                     fun=lambda x: x*60.0, fmt="%4.0f^\prime", 
                     fliplabelside=True)
ruler2.setp_line(lw=2, color='b')
ruler2.setp_label(color='b')


annim.plot()
annim.interact_writepos()

plt.show()
