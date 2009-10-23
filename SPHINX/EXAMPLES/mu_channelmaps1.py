from kapteyn import maputils
from matplotlib import pylab as plt

# This is our function to convert velocity from m/s to km/s
def fx(x):
  return x/1000.0

# Create an object from the FITSimage class:
fitsobj = maputils.FITSimage('ngc6946.fits')


# We want to plot the image that corresponds to a certain velocity,
# let's say a radio velocity of 100 km/s
# Find the axis number that corresponds to the spectral axis:
specaxnum = fitsobj.proj.specaxnum
spec = fitsobj.proj.sub(specaxnum).spectra("VRAD-???")
channel = spec.topixel1d(100*1000.0)
channel = round(channel)                         # We need an integer pixel coordinate
vel = spec.toworld1d(channel)                    # Velocity near 100 km/s 

# Set for this 3d data set the image axes and the position
# for the slice, i.e. the frequency pixel
lonaxnum = fitsobj.proj.lonaxnum
lataxnum = fitsobj.proj.lataxnum
fitsobj.set_imageaxes(lonaxnum,lataxnum, slicepos=channel)

fig = plt.figure(figsize=(7,8))
frame = fig.add_axes([0.3,0.5,0.4,0.4])
mplim = fitsobj.Annotatedimage(frame)
mplim.Image()

# The FITSimage object contains all the relevant information 
# to set the graticule for this image
grat = mplim.Graticule()
ruler = grat.Ruler(-51.1916, 59.9283, -51.4877, 60.2821, 0.5, 0.05, fmt="%5.2f", world=True)

grat.setp_tick(plotaxis="right", color='r')
pixellabels = grat.Pixellabels(plotaxis=("right","top"), color='r', fontsize=7)
mplim.plot()

# First position-velocity plot at RA=51
fitsobj.set_imageaxes(lataxnum, specaxnum, slicepos=51)
frame2 = fig.add_axes([0.1,0.3,0.8,0.1])
mplim2 = fitsobj.Annotatedimage(frame2)
mplim2.set_aspectratio(0.15)
mplim2.Image()
grat2 = mplim2.Graticule()
grat2.setp_plotaxis("right", mode="native_ticks", label='Velocity (km/s)',
                    fontsize=9, visible=False)
grat2.setp_tick(plotaxis="right", fmt="%5g", fun=fx)
grat2.setp_plotaxis("bottom", label=r"Offset in latitude (arcmin) at $\alpha$ pixel 51",
                     fontsize=9)
grat2.setp_plotaxis("left", mode="no_ticks", visible=False)
mplim2.Pixellabels(plotaxis=("top", "left"))
mplim2.plot()

# Second position-velocity plot at DEC=51
fitsobj.set_imageaxes(lonaxnum, specaxnum, slicepos=51)
frame3 = fig.add_axes([0.1,0.1,0.8,0.1])
mplim3 = fitsobj.Annotatedimage(frame3)
mplim3.set_aspectratio(0.15)
mplim3.Image()

grat3 = mplim3.Graticule()
grat3.setp_plotaxis("right", mode="native_ticks", label='Velocity (km/s)', fontsize=9)
grat3.setp_tick(plotaxis="right", fmt="%5g", fun=fx)
grat3.setp_plotaxis("left",  mode="no_ticks", visible=False)
grat3.setp_plotaxis("bottom", label=r"Offset in longitude (arcmin) at $\delta$ pixel=51",
                    fontsize=9)
mplim3.Pixellabels(plotaxis=("top", "left"))
mplim3.plot()

# Adjust position of title
t = frame.set_title('NGC 6946 at %g km/s (channel %d)' % (vel/1000.0, channel))
t.set_y(1.1)

plt.show()
