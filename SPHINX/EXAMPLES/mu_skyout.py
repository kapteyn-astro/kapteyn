from kapteyn.wcs import galactic, equatorial, fk4_no_e, fk5
from kapteyn import maputils
from matplotlib import pylab as plt

# Open FITS file and get header
f = maputils.FITSimage('example1test.fits')

fig = plt.figure(figsize=(6,6))
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)

# Initialize a graticule for this header and set some attributes
grat = mplim.Graticule()
grat.setp_lineswcs0(color='g')    # Set properties of graticule lines in R.A.
grat.setp_lineswcs1(color='g')    # In Dec
grat.setp_tick(plotaxis=("left","bottom"), markersize=-10, 
                         color='b', fontsize=14)

# Select another sky system for an overlay
skyout = galactic        # Also try: skyout = (equatorial, fk5, 'J3000')
grat2 = mplim.Graticule(skyout=skyout, boxsamples=20000)
grat2.setp_plotaxis(("top", "right"), label="Galactic l,b", 
                     mode="all_ticks", visible=True)
grat2.setp_plotaxis("top", color='r')
grat2.setp_plotaxis(("left", "bottom"), 
                    mode="no_ticks", visible=False)
grat2.setp_lineswcs0(color='r')
grat2.setp_lineswcs1(color='r')

# Print coordinate labels inside the plot boundaries
grat2.Insidelabels(wcsaxis=0, color='m')
grat2.Insidelabels(wcsaxis=1)
mplim.Pixellabels(plotaxis=("right","top"), gridlines=True, 
                 color='c', markersize=-3, fontsize=7)

mplim.plot()
plt.show()
