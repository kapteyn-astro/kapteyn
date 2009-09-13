from kapteyn.wcs import galactic, equatorial, fk4_no_e, fk5
from kapteyn import wcsgrat
from matplotlib import pylab as plt
import pyfits

# Open FITS file and get header
hdulist = pyfits.open('example1test.fits')
header = hdulist[0].header

# Initialize a graticule for this header and set some attributes
#grat = wcsgrat.Graticule(header, boxsamples=5000)
grat = wcsgrat.Graticule(header)
grat.setp_lineswcs0(color='g')                # Set properties of graticule lines in R.A.
grat.setp_lineswcs1(color='g')
grat.setp_tick(plotaxis=(0,1), markersize=-10, color='b', fontsize=14)

# Select another sky system for an overlay
skyout = galactic        # Also try: skyout = (equatorial, fk5, 'J3000')
grat2 = wcsgrat.Graticule(header, skyout=skyout, boxsamples=20000)
grat2.setp_plotaxis((wcsgrat.top, wcsgrat.right), label="Galactic l,b", 
                    mode=wcsgrat.bothticks, visible=True)
grat2.setp_plotaxis(wcsgrat.top, color='r')
grat2.setp_plotaxis((wcsgrat.left, wcsgrat.bottom), mode=wcsgrat.noticks, visible=False)
grat2.setp_lineswcs0(color='r')
grat2.setp_lineswcs1(color='r')

# Print coordinate labels inside the plot boundaries
ilabs1 = grat2.insidelabels(wcsaxis=0, color='m')
ilabs2 = grat2.insidelabels(wcsaxis=1)
pixellabels = grat.pixellabels(plotaxis=(2,3), gridlines=True, color='c', 
                               markersize=-3, fontsize=7)

# Plot this graticule with Matplotlib, i.e. create a figure, an 
# Axes instance (frame) and connect to Matplotlib
fig = plt.figure(figsize=(8,8))
frame = fig.add_subplot(1,1,1, adjustable='box', aspect=grat.aspectratio)
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add( [grat,grat2,ilabs1,ilabs2,pixellabels])

plt.show()
