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
annim = fitsobj.Annotatedimage(frame)
annim.Image()

# The FITSimage object contains all the relevant information 
# to set the graticule for this image
grat = annim.Graticule()
ruler = annim.Ruler(x1=-51.1916, y1=59.9283, x2=-51.4877, y2=60.2821, 
                    units='arcmin', step=3, mscale=5.0, 
                    color='w', world=True, ha='right')

grat.setp_tick(plotaxis="right", color='r')
pixellabels = annim.Pixellabels(plotaxis=("right","top"), color='r', fontsize=7)

# First position-velocity plot at RA=51
fitsobj.set_imageaxes(lataxnum, specaxnum, slicepos=51)
frame2 = fig.add_axes([0.1,0.3,0.8,0.1])
annim2 = fitsobj.Annotatedimage(frame2)
annim2.set_aspectratio(0.15)
annim2.Image()
grat2 = annim2.Graticule()
grat2.setp_axislabel(plotaxis="right", label='Velocity (km/s)',
                    fontsize=9, visible=True)
grat2.set_tickmode(plotaxis="right", mode="native_ticks")
grat2.setp_ticklabel(plotaxis="right", fmt="%+5g", fun=fx)
grat2.setp_axislabel("bottom", 
                     label=r"Offset in latitude (arcmin) at $\alpha$ = pixel 51",
                     fontsize=9)
grat2.setp_axislabel(plotaxis="left", visible=False)
grat2.set_tickmode(plotaxis="left", mode="no_ticks")
annim2.Pixellabels(plotaxis=("top", "left"))

# Second position-velocity plot at DEC=51
fitsobj.set_imageaxes(lonaxnum, specaxnum, slicepos=51)
frame3 = fig.add_axes([0.1,0.1,0.8,0.1])
annim3 = fitsobj.Annotatedimage(frame3)
annim3.set_aspectratio(0.15)
annim3.Image()
grat3 = annim3.Graticule()
grat3.setp_axislabel("right", 
                     label='Velocity (km/s)', fontsize=9, visible=True)
grat3.set_tickmode(plotaxis='right', mode="native_ticks")
# The next line forces labels to be right aligned, but one needs a shift
# in x to set the labels outside the plot
grat3.setp_ticklabel(plotaxis="right", fmt="%8g", fun=fx, ha="right", x=1.075)
grat3.setp_axislabel(plotaxis="left", visible=False)
grat3.set_tickmode(plotaxis="left",  mode="no_ticks")
grat3.setp_axislabel("bottom", 
                     label=r"Offset in longitude (arcmin) at $\delta$ = pixel 51",
                     fontsize=9)
annim3.Pixellabels(plotaxis=("top", "left"))

# Set title and adjust position of title
frame.set_title('NGC 6946 at %g km/s (channel %d)' % (vel/1000.0, channel), y=1.1)

maputils.showall()
