from kapteyn.wcs import galactic, equatorial, fk4_no_e, fk5
from kapteyn import maputils
from matplotlib import pylab as plt

# Open FITS file and get header
f = maputils.FITSimage('example1test.fits')

fig = plt.figure(figsize=(6,6))
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)

# Initialize a graticule for this header and set some attributes
grat = annim.Graticule()
grat.setp_gratline(wcsaxis=[0,1],color='g') # Set graticule lines to green
grat.setp_ticklabel(plotaxis=("left","bottom"), color='b', 
                    fontsize=14, fmt="Hms")
grat.setp_tickmark(plotaxis=("left","bottom"), markersize=-10)

# Select another sky system for an overlay
skyout = galactic        # Also try: skyout = (equatorial, fk5, 'J3000')
grat2 = annim.Graticule(skyout=skyout, boxsamples=20000)
grat2.setp_axislabel(plotaxis=("top", "right"), label="Galactic l,b", 
                     visible=True)
grat2.set_tickmode(plotaxis=("top", "right"), mode="ALL")

grat2.setp_axislabel(plotaxis="top", color='r')
grat2.setp_axislabel(plotaxis=("left", "bottom"),  visible=False)
grat2.setp_ticklabel(plotaxis=("left", "bottom"),  visible=False)
grat2.setp_ticklabel(plotaxis=("top", "right"), fmt='Dms')
grat2.setp_gratline(color='r')

# Print coordinate labels inside the plot boundaries
grat2.Insidelabels(wcsaxis=0, color='m', constval=85, fmt='Dms')
grat2.Insidelabels(wcsaxis=1, fmt='Dms')
annim.Pixellabels(plotaxis=("right","top"), gridlines=True, 
                  color='c', markersize=-3, fontsize=7)

annim.plot()
plt.show()
