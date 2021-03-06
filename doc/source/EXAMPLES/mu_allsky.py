mu_annotatedcontours.py                                                                             0000644 0000764 0000144 00000001136 11256703432 015432  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(200,350), pylim=(200,350))

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
cont = mplim.Contours(levels=range(10000,16000,1000))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=3)

# Second contour set only for labels
cont2 = mplim.Contours(levels=(8000,9000,10000,11000))
cont2.setp_label(11000, colors='b', fontsize=14, fmt="%.3f")
cont2.setp_label(fontsize=10, fmt="$%g \lambda$")

mplim.plot()

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                  mu_arcminrulers.py                                                                                  0000644 0000764 0000144 00000002244 11257641734 014376  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 100,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' : 80.0, 'CRPIX1' : 1, 
          'CUNIT1' : 'arcmin', 'CDELT1' : -0.5,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 400.0, 'CRPIX2' : 1, 
          'CUNIT2' : 'arcmin', 'CDELT2' : 0.5,
          'CROTA2' : 30.0
         }

f = maputils.FITSimage(externalheader=header)

fig = plt.figure(figsize=(7,7))
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)
grat = mplim.Graticule()

# Use pixel limits attributes of the FITSimage object

xmax = mplim.pxlim[1]+0.5; ymax = mplim.pylim[1]+0.5
grat.Ruler(xmax,0.5, xmax, ymax, lambda0=0.5, step=5.0/60.0, 
           fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", 
           fliplabelside=True, color='r')

# The wcs methods that convert between pixels and world
# coordinates expect input in degrees whatever the units in the
# header are (e.g. arcsec, arcmin).
grat.Ruler(60/60.0,390/60.0,60/60.0,420/60.0, 
           lambda0=0.0, step=5.0/60, world=True, 
           fun=lambda x: x*60.0, fmt=r"$%4.0f^\prime$", color='r')
mplim.plot()

plt.show()
                                                                                                                                                                                                                                                                                                                                                            mu_axnumdemo.py                                                                                     0000644 0000764 0000144 00000003044 11275560675 013670  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj= maputils.FITSimage('ngc6946.fits')
newaspect = 1/5.0

fig = plt.figure(figsize=(20/2.54, 25/2.54))
labelx = -0.10         # Fix the  position in x for labels along y
frame = fig.add_subplot(4,1,1)

# fig 1. Spatial map
mplim1 = fitsobj.Annotatedimage(frame)
mplim1.Image()
graticule1 = mplim1.Graticule()

# fig 2. Velocity - Dec
frame2 = fig.add_subplot(4,1,2)
fitsobj.set_imageaxes(3,2)
mplim2 = fitsobj.Annotatedimage(frame2)
mplim2.Image()
graticule2 = mplim2.Graticule()

# fig 3. Velocity - Dec (Version without offsets)
frame3 = fig.add_subplot(4,1,3)
mplim3 = fitsobj.Annotatedimage(frame3)
mplim3.Image()
graticule3 = mplim3.Graticule(offsety=False)

# fig 4. Velocity - R.A.
frame4 = fig.add_subplot(4,1,4)
fitsobj.set_imageaxes(3,1)
mplim4 = fitsobj.Annotatedimage(frame4)
mplim4.Image()
graticule4 = mplim4.Graticule(offsety=False)
graticule4.setp_tick(plotaxis="left", fun=lambda x: x+360, fmt="$%.1f^\circ$")
graticule4.Insidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10, 
                        color='r', ha='right')
graticule4.Insidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')

mplim2.set_aspectratio(newaspect)
mplim3.set_aspectratio(newaspect)
mplim4.set_aspectratio(newaspect)

mplim1.plot()
mplim2.plot()
mplim3.plot()
mplim4.plot()

# Align left axis title in frame of graticule
graticule2.frame.yaxis.set_label_coords(labelx, 0.5)
graticule3.frame.yaxis.set_label_coords(labelx, 0.5)
graticule4.frame.yaxis.set_label_coords(labelx, 0.5)

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            mu_axnumdemosimple.py                                                                               0000644 0000764 0000144 00000000306 11275556763 015103  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
mplim = fitsobj.Annotatedimage()
graticule = mplim.Graticule()
mplim.plot()

plt.show()

                                                                                                                                                                                                                                                                                                                          mu_basic1.py                                                                                        0000644 0000764 0000144 00000000416 11275512203 013015  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)
annim.Image()
annim.Graticule()
annim.plot()
annim.interact_imagecolors()
plt.show()
                                                                                                                                                                                                                                                  mu_basic2.py                                                                                        0000644 0000764 0000144 00000000657 11275510470 013031  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits((200,400), (200,400))
fig = plt.figure()
frame = fig.add_subplot(2,1,1)
annim = f.Annotatedimage(frame)
annim.Image(interpolation="nearest")
annim.Graticule()
annim.plot()
frame = fig.add_subplot(2,1,2)
annim = f.Annotatedimage(frame)
annim.Image(interpolation="spline36")
annim.Graticule()
annim.plot()
plt.show()
                                                                                 mu_channelmaps1.py                                                                                  0000644 0000764 0000144 00000005441 11261101172 014222  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
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
                                                                                                                                                                                                                               mu_channelmosaic.py                                                                                 0000644 0000764 0000144 00000003136 11273361777 014501  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt

# This is our function to convert velocity from m/s to km/s
def fx(x):
  return x/1000.0

# Create an object from the FITSimage class:
fitsobj = maputils.FITSimage('ngc6946.fits')
specaxnum = fitsobj.proj.specaxnum
lonaxnum = fitsobj.proj.lonaxnum
lataxnum = fitsobj.proj.lataxnum
spec = fitsobj.proj.sub(specaxnum).spectra("VRAD-???")

start = 5; end = fitsobj.proj.naxis[specaxnum-1]; step = 4
channels = range(start,end,step)
nchannels = len(channels)

fig = plt.figure(figsize=(7,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

vmin, vmax = fitsobj.get_dataminmax()
print "Vmin, Vmax of data in cube:", vmin, vmax
cmap = 'spectral'             # Colormap
cols = 5
rows = nchannels / cols
if rows*cols < nchannels: rows += 1
for i, ch in enumerate(channels):
   fitsobj.set_imageaxes(lonaxnum, lataxnum, slicepos=ch)
   print "Min, max in this channel: ", fitsobj.get_dataminmax(box=True)
   frame = fig.add_subplot(rows, cols, i+1)
   mplim = fitsobj.Annotatedimage(frame, 
                                  clipmin=vmin, clipmax=vmax, 
                                  cmap=cmap)
   mplim.Image()

   vel = spec.toworld1d(ch)
   velinfo = "ch%d = %.1f km/s" % (ch, vel/1000.0)
   frame.text(0.98, 0.98, velinfo,
              horizontalalignment='right',
              verticalalignment='top',
              transform = frame.transAxes,
              fontsize=8, color='w',
              bbox=dict(facecolor='red', alpha=0.5))
   mplim.plot()
   if i == 0:
      cmap = mplim.cmap
   mplim.interact_imagecolors()

plt.show()


                                                                                                                                                                                                                                                                                                                                                                                                                                  mu_clown.py                                                                                         0000644 0000764 0000144 00000002203 11277033265 013001  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import fft, log, asarray, float64, abs, angle

#f = maputils.FITSimage("m101.fits")
f = maputils.FITSimage("cl2.fits")

fig = plt.figure(figsize=(8,8))
frame = fig.add_subplot(2,2,1)

mplim = f.Annotatedimage(frame, cmap="gray")
mplim.Image()
mplim.plot()

fftA = fft.rfft2(f.dat, f.dat.shape)
fftre = fftA.real
fftim = fftA.imag

frame = fig.add_subplot(2,2,2)
#f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftre)+1.0))
f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftA)+1.0))
mplim2 = f.Annotatedimage(frame, cmap="gray")
im = mplim2.Image()
mplim2.plot()

frame = fig.add_subplot(2,2,3)
f = maputils.FITSimage("m101.fits", externaldata=angle(fftA))
mplim3 = f.Annotatedimage(frame, cmap="gray")
im = mplim3.Image()
mplim3.plot()

frame = fig.add_subplot(2,2,4)
D = fft.irfft2(fftA)
f = maputils.FITSimage("m101.fits", externaldata=D.real)
mplim4 = f.Annotatedimage(frame, cmap="gray")
im = mplim4.Image()
mplim4.plot()

mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()
mplim4.interact_imagecolors()

plt.show()
                                                                                                                                                                                                                                                                                                                                                                                             mu_colbarframe.py                                                                                   0000644 0000764 0000144 00000001035 11276234770 014141  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fig = plt.figure(figsize=(5,7.5))
#fig = plt.figure()
frame = fig.add_axes((0.1, 0.2, 0.8, 0.8))
cbframe = fig.add_axes((0.1, 0.1, 0.8, 0.1))

annim = fitsobj.Annotatedimage(cmap="Accent", clipmin=8000, frame=frame)
annim.Image()
units = r'$ergs/(sec.cm^2)$'
colbar = annim.Colorbar(fontsize=8, orientation='horizontal', frame=cbframe)
colbar.set_label(label=units, fontsize=24)
annim.plot()
annim.interact_imagecolors()
plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   mu_colbar.py                                                                                        0000644 0000764 0000144 00000000460 11276234334 013123  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")

mplim = fitsobj.Annotatedimage(cmap="spectral")
mplim.Image()
units = r'$ergs/(sec.cm^2)$'
colbar = mplim.Colorbar(fontsize=8)
colbar.set_label(label=units, fontsize=24)
mplim.plot()
plt.show()
                                                                                                                                                                                                                mu_colbarwithlines.py                                                                               0000644 0000764 0000144 00000004060 11276236753 015061  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
limy = limx=(160,360)
f.set_limits(limx,limy)
rows = 3
cols = 2

fig = plt.figure(figsize=(8,10))

frame = fig.add_subplot(rows,cols,1)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours()
mplim.Colorbar(clines=True, fontsize=8,
               linewidths=3, visible=False) # show only cont. lines
mplim.plot()
# Levels only known after plotted
print "Proposed levels:", cont.clevels

frame = fig.add_subplot(rows,cols,2)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours(filled=True)
mplim.Colorbar(clines=True, fontsize=8) # show only cont. lines
mplim.plot()

frame = fig.add_subplot(rows,cols,3)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image()
cont = mplim.Contours(colors='w', linewidths=2)
mplim.Colorbar(clines=True, ticks=(4000,8000,12000))
mplim.plot()
mplim.interact_imagecolors()

frame = fig.add_subplot(rows,cols,4)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image()
cont = mplim.Contours(levels=(4000,6000,8000,10000,12000), 
                      colors=('r','b','y','g', 'c'))
mplim.Colorbar(clines=True, ticks=(4000,8000,12000), linewidths=2)
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

frame = fig.add_subplot(rows,cols,5)
mplim = f.Annotatedimage(frame, cmap="mousse.lut")
mplim.Image()
cont = mplim.Contours()
mplim.Colorbar(clines=True, orientation="horizontal", ticks=(4000,8000,12000))
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

# With given levels
frame = fig.add_subplot(rows,cols,6)
levels = (10000,11000,12000,13000)
mplim = f.Annotatedimage(frame, cmap="mousse.lut", 
                         clipmin=min(levels)-500,
                         clipmax=max(levels)+500)
mplim.Image()
cont = mplim.Contours(levels=levels)
mplim.Colorbar(clines=True, orientation="horizontal", 
               ticks=levels)
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                mu_contourlinestyles.py                                                                             0000644 0000764 0000144 00000000627 11277110526 015470  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
annim.Image(alpha=0.5)
cont = annim.Contours(linestyles=('solid', 'dashed', 'dashdot', 'dotted'),
                      linewidths=(2,3,4), colors=('r','g','b','m'))
annim.plot()

print "Levels=", cont.clevels

plt.show()

                                                                                                         mu_extendcontours.py                                                                                0000644 0000764 0000144 00000000731 11266623755 014756  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
limy = limx=(160,360)
f.set_limits(limx,limy)
fig = plt.figure(figsize=(10,10))

frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image(visible=False)
cont = mplim.Contours(filled=True)
mplim.Colorbar(clines=True, fontsize=8) # show only cont. lines
mplim.plot()
mplim.interact_imagecolors()
mplim.interact_toolbarinfo()

plt.show()
                                       mu_externaldata.py                                                                                  0000644 0000764 0000144 00000001717 11265633756 014354  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt
import numpy

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

nx = header['NAXIS1']
ny = header['NAXIS2']
sizex1 = nx/2.0; sizex2 = nx - sizex1
sizey1 = nx/2.0; sizey2 = nx - sizey1
x, y = numpy.mgrid[-sizex1:sizex2, -sizey1:sizey2]
edata = numpy.exp(-(x**2/float(sizex1*10)+y**2/float(sizey1*10)))

f = maputils.FITSimage(externalheader=header, externaldata=edata)
f.writetofits()
fig = plt.figure(figsize=(7,7))
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
mplim = f.Annotatedimage(frame, cmap='pink')
mplim.Image()
gr = mplim.Graticule()
gr.setp_gratline(color='y')
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()
                                                 mu_externalheader.py                                                                                0000644 0000764 0000144 00000001415 11275617362 014662  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt
import numpy

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

# Overrule the header value for pixel size in y direction
header['CDELT2'] = 0.3*abs(header['CDELT1'])
fitsobj = maputils.FITSimage(externalheader=header)
figsize = fitsobj.get_figsize(ysize=7, cm=True)

fig = plt.figure(figsize=figsize)
print "Figure size x, y in cm:", figsize[0]*2.54, figsize[1]*2.54
frame = fig.add_subplot(1,1,1)
mplim = fitsobj.Annotatedimage(frame)
gr = mplim.Graticule()
mplim.plot()

plt.show()
                                                                                                                                                                                                                                                   mu_fft.py                                                                                           0000644 0000764 0000144 00000004014 11270404463 012433  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import fft, log, abs, angle

f = maputils.FITSimage("m101.fits")

yshift = -0.1
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.01, bottom=0.1, right=1.0, top=0.98, 
                    wspace=0.03, hspace=0.16)
frame = fig.add_subplot(2,3,1)
frame.text(0.5, yshift, "M101", ha='center', va='center',
           transform = frame.transAxes)
mplim = f.Annotatedimage(frame, cmap="spectral")
mplim.Image()
mplim.plot()


fftA = fft.rfft2(f.dat, f.dat.shape)
frame = fig.add_subplot(2,3,2)
frame.text(0.5, yshift, "Amplitude of FFT", ha='center', va='center',
           transform = frame.transAxes)
f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftA)+1.0))
mplim2 = f.Annotatedimage(frame, cmap="gray")
im = mplim2.Image()
mplim2.plot()


frame = fig.add_subplot(2,3,3)
frame.text(0.5, yshift, "Phase of FFT", ha='center', va='center',
           transform = frame.transAxes)
f = maputils.FITSimage("m101.fits", externaldata=angle(fftA))
mplim3 = f.Annotatedimage(frame, cmap="gray")
im = mplim3.Image()
mplim3.plot()


frame = fig.add_subplot(2,3,4)
frame.text(0.5, yshift, "Inverse FFT", ha='center', va='center',
           transform = frame.transAxes)
D = fft.irfft2(fftA)
f = maputils.FITSimage("m101.fits", externaldata=D.real)
mplim4 = f.Annotatedimage(frame, cmap="spectral")
im = mplim4.Image()
mplim4.plot()

frame = fig.add_subplot(2,3,5)
Diff = D.real - mplim.data
f = maputils.FITSimage("m101.fits", externaldata=Diff)
mplim5 = f.Annotatedimage(frame, cmap="spectral")
im = mplim5.Image()
mplim5.plot()

frame.text(0.5, yshift, "M101 - inv. FFT", ha='center', va='center',
           transform = frame.transAxes)
s = "Residual with min=%.1g max=%.1g" % (Diff.min(), Diff.max())
frame.text(0.5, yshift-0.08, s, ha='center', va='center',
           transform = frame.transAxes, fontsize=8)

mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()
mplim4.interact_imagecolors()
mplim5.interact_imagecolors()

plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    mu_figsize.py                                                                                       0000644 0000764 0000144 00000001303 11257664733 013327  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt
import numpy

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

header['CDELT2'] = 0.3*abs(header['CDELT1'])
f = maputils.FITSimage(externalheader=header)

#figsize = f.get_figsize(ysize=21, cm=True)
figsize = f.get_figsize()

fig = plt.figure(figsize=figsize)
print figsize[0]*2.54, figsize[1]*2.54
frame = fig.add_subplot(1,1,1)
mplim = f.Annotatedimage(frame)
gr = mplim.Graticule()
mplim.plot()

plt.show()
                                                                                                                                                                                                                                                                                                                             mu_figuredemo.py                                                                                    0000644 0000764 0000144 00000000465 11277026110 014004  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('example1test.fits')

fig = plt.figure(figsize=(5,5))
frame = fig.add_axes([0.1, 0.1, 0.8, 0.8])
annim = fitsobj.Annotatedimage(frame)
annim.set_aspectratio(1.2)
grat = annim.Graticule()

annim.plot()

plt.show()
                                                                                                                                                                                                           mu_fitsutils.py                                                                                     0000644 0000764 0000144 00000001111 11252473371 013701  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('ngc6946.fits')

print("HEADER:\n")
print fitsobject.str_header()

print("\nAXES INFO:\n")
print fitsobject.str_axisinfo()

print("\nEXTENDED AXES INFO:\n")
print fitsobject.str_axisinfo(long=True)

print("\nAXES INFO for image axes only:\n")
print fitsobject.str_axisinfo(axnum=fitsobject.axperm)

print("\nAXES INFO for non existing axis:\n")
print fitsobject.str_axisinfo(axnum=4)

print("SPECTRAL INFO:\n")
fitsobject.set_imageaxes(axnr1=1, axnr2=3)
print fitsobject.str_spectrans()
                                                                                                                                                                                                                                                                                                                                                                                                                                                       mu_getfitsfile.py                                                                                   0000644 0000764 0000144 00000000434 11252473366 014173  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)

print("\nAXES INFO for image axes:\n")
print fitsobject.str_axisinfo(axnum=fitsobject.axperm)

print("\nWCS INFO:\n")
print fitsobject.str_wcsinfo()
                                                                                                                                                                                                                                    mu_getfitsimage.py                                                                                  0000644 0000764 0000144 00000000701 11261605161 014321  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
print fitsobject.str_axisinfo()
fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
fitsobject.set_limits(promptfie=maputils.prompt_box)
print fitsobject.str_spectrans()
fitsobject.set_spectrans(promptfie=maputils.prompt_spectrans)
fitsobject.set_skyout(promptfie=maputils.prompt_skyout)

print("\nWCS INFO:")
print fitsobject.str_wcsinfo()
                                                               mu_graticulebasic.py                                                                                0000644 0000764 0000144 00000000301 11277237537 014645  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('m101.fits')
annim = fitsobj.Annotatedimage()
grat = annim.Graticule()
annim.plot()

plt.show()

                                                                                                                                                                                                                                                                                                                               mu_graticule.py                                                                                     0000644 0000764 0000144 00000002426 11277357431 013651  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(50,440), pylim=(50,450))

fig = plt.figure(figsize=(8,5.5))
frame = fig.add_axes((0.05, 0.1, 0.8, 0.7))
fig.text(0.5, 0.96, "Combination of plot objects", horizontalalignment='center',
         fontsize=14, color='r')

annim = f.Annotatedimage(frame, clipmin=3000, clipmax=15000)
annim.Pixellabels(plotaxis='top', va='top')
annim.Pixellabels(plotaxis='right')

cont = annim.Contours(levels=range(8000,14000,1000))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=2)

cb = annim.Colorbar(clines=False, orientation='vertical', fontsize=8)

gr = annim.Graticule()
gr.Insidelabels()

gr2 = annim.Graticule(deltax=7.5/60, deltay=5.0/60,
                      skyout="galactic", 
                      visible=True)
gr2.setp_plotaxis(("top","right"), label="Galactic l,b", 
                  mode=maputils.native, color='g', visible=True)
gr2.setp_tick(wcsaxis=(0,1), color='g')
gr2.setp_gratline(wcsaxis=(0,1), color='g')
gr2.setp_plotaxis(("left","bottom"), mode=maputils.noticks, visible=False)
gr2.Ruler(150,100,150,330, step=1/60.0)
gr2.Ruler(102,59+50/60.0, 102+7.5/60,59+50/60.0,  world=True, step=1/60.0, color='r')

annim.plot()
plt.show()

                                                                                                                                                                                                                                          mu_histeq.py                                                                                        0000644 0000764 0000144 00000002155 11261152124 013147  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

tpos = 1.02
f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(200,300), pylim=(200,300))
print f.pxlim, f.pylim

fig = plt.figure()
frame = fig.add_subplot(1,1,1)
t = frame.set_title("Original")
t.set_y(tpos)

mplim = f.Annotatedimage(frame)
ima = mplim.Image()
mplim.Pixellabels()
mplim.plot()

fig2 = plt.figure()
frame2 = fig2.add_subplot(1,1,1)
t = frame2.set_title("Histogram equalized")
t.set_y(tpos)

mplim2 = f.Annotatedimage(frame2)
ima2 = mplim2.Image(visible=True)
ima2.histeq()
mplim2.Pixellabels()
mplim2.plot()


fig3 = plt.figure()
frame3 = fig3.add_subplot(1,1,1)
t = frame3.set_title("Colors with LogNorm")
t.set_y(tpos)

mplim3 = f.Annotatedimage(frame3)
ima3 = mplim3.Image(norm=LogNorm())
mplim3.Pixellabels()
mplim3.plot()

"""
fig4 = plt.figure()
frame4 = fig4.add_subplot(1,1,1)
mplim4 = f.Annotatedimage(frame4)
ima4 = mplim4.Image()
ima4.blur_image(2,2)
mplim4.Pixellabels()
mplim4.plot()
"""


mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                   mu_hurricane.py                                                                                     0000644 0000764 0000144 00000004370 11300034642 013631  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

def plotcoast(fn, pxlim, pylim, col='k'):
   t = tabarray.tabarray(fn).T  # Read two columns from file
   xw = t[1]; yw = t[0]         # First one appears to be Latitude
   xs = []; ys = []             # Reset lists which store valid pos.
  
   # Process segments. A segment starts with nan
   for x,y in zip(xw,yw):
     if numpy.isnan(x) and len(xs):  # Useless if empty
        # Mask arrays if outside plot box
        xp, yp = annim.projection.topixel((numpy.array(xs),numpy.array(ys)))
        xp = numpy.ma.masked_where(numpy.isnan(xp) | 
                           (xp > pxlim[1]) | (xp < pxlim[0]), xp)
        yp = numpy.ma.masked_where(numpy.isnan(yp) | 
                           (yp > pylim[1]) | (yp < pylim[0]), yp)
        # Mask array could be of type numpy.bool_ instead of numpy.ndarray
        if numpy.isscalar(xp.mask):
           xp.mask = numpy.array(xp.mask, 'bool')
        if numpy.isscalar(yp.mask):
           yp.mask = numpy.array(yp.mask, 'bool')
        # Count the number of positions in these list that are inside the box
        j = 0
        for i in range(len(xp)):
           if not xp.mask[i] and not yp.mask[i]:
              j += 1
        if j > 200:   # Threshold to prevent too much detail and big pdf's
           frame.plot(xp.data, yp.data, color=col)     
        xs = []; ys = []
     else:
        # Store world coordinate that is member of current segment
        xs.append(x)
        ys.append(y)


f = maputils.FITSimage("m101cdelt.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame, cmap="YlGn")
annim.Image()
grat = annim.Graticule()
grat.setp_tick(wcsaxis=0, fmt="$%g^{\circ}$")
grat.setp_plotaxis(plotaxis='bottom', label='West - East')
grat.setp_plotaxis(plotaxis='left', label='South - North')
annim.plot()
annim.projection.allow_invalid = True

# Plot coastlines in black, borders in red
plotcoast('namerica-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('namerica-bdy.txt', annim.pxlim, annim.pylim, col='r')
plotcoast('samerica-cil.txt', annim.pxlim, annim.pylim, col='k')
plotcoast('samerica-bdy.txt', annim.pxlim, annim.pylim, col='r')

annim.interact_imagecolors()
plt.show()
                                                                                                                                                                                                                                                                        mu_imagewithblanks.py                                                                               0000644 0000764 0000144 00000001261 11270115166 015025  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

f = maputils.FITSimage("renseblank.fits")
f.set_imageaxes(1,2)

fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="mousse.lut", blankcolor='w',
                         clipmin=0.04, clipmax=0.12)
mplim.Image()
#mplim.Image()
#mplim.set_blankcolor('c')
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

                                                                                                                                                                                                                                                                                                                                               mu_interactive2.py                                                                                  0000644 0000764 0000144 00000001255 11270607155 014262  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  """Show interaction options"""
from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn.mplutil import KeyPressFilter

KeyPressFilter.allowed = ['f','g', 'l']


f = maputils.FITSimage("m101.fits")
#f.set_limits((100,500),(200,400))

fig = plt.figure(figsize=(9, 7))
frame = fig.add_subplot(1, 1, 1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="mousse.lut")
mplim.cmap.set_bad('w')
ima = mplim.Image()
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

                                                                                                                                                                                                                                                                                                                                                   mu_interactive3.py                                                                                  0000644 0000764 0000144 00000001063 11263375136 014263  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)

mycmlist = glob.glob("/home/gipsy/dat/lut/*.lut")
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame)
ima = mplim.Image(cmap="mousse.lut")
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                             mu_introduction.py                                                                                  0000644 0000764 0000144 00000001443 11277033524 014403  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  #!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

   
# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
fitsobject.set_limits(promptfie=maputils.prompt_box)
fitsobject.set_skyout(promptfie=maputils.prompt_skyout)
clipmin, clipmax = maputils.prompt_dataminmax(fitsobject)
   
# Get connected to Matplotlib
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
   
# Create an image to be used in Matplotlib
annim = fitsobject.Annotatedimage(frame, clipmin=clipmin, clipmax=clipmax)
annim.Image()
annim.Graticule()
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

plt.show()
                                                                                                                                                                                                                             mu_manyaxes.py                                                                                      0000644 0000764 0000144 00000004160 11275557271 013516  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

# 1. Read the header
fitsobj = maputils.FITSimage("manyaxes.fits")

# 2. Create a Matplotlib Figure and Axes instance
figsize=fitsobj.get_figsize(ysize=7)
fig = plt.figure(figsize=figsize)
frame = fig.add_subplot(1,1,1)

# 3. Create a graticule
fitsobj.set_imageaxes(1,4)
mplim = fitsobj.Annotatedimage(frame)
grat = mplim.Graticule(starty=1000, deltay=10)

# 4. Show the calculated world coordinates along y-axis
print "The world coordinates along the y-axis:", grat.ystarts

# 5. Show header information in attributes of the Projection object
print "CRVAL, CDELT from header:", grat.gmap.crval, grat.gmap.cdelt

# 6. Set a number of properties of the graticules and plot axes
grat.setp_tick(plotaxis="bottom", 
               fun=lambda x: x/1.0e9, fmt="%.4f",
               rotation=-30 )

grat.setp_plotaxis("bottom", label="Frequency (GHz)")
grat.setp_gratline(wcsaxis=0, position=grat.gmap.crval[0], 
                   tol=0.5*grat.gmap.cdelt[0], color='r')
grat.setp_tick(plotaxis="left", position=1000, color='m', fmt="I")
grat.setp_tick(plotaxis="left", position=1010, color='b', fmt="Q")
grat.setp_tick(plotaxis="left", position=1020, color='r', fmt="U")
grat.setp_tick(plotaxis="left", position=1030, color='g', fmt="V")
grat.setp_plotaxis("left", label="Stokes parameters")


# 7. Set a title for this frame
title = r"""Polarization as function of frequency at:
            $(\alpha_0,\delta_0) = (121^o,53^o)$"""
t = frame.set_title(title, color='#006400')
t.set_y(1.01)
t.set_linespacing(1.4)

# 8. Add labels inside plot
inlabs = grat.Insidelabels(wcsaxis=0, constval=1015, 
                           deltapx=-0.15, rotation=90, 
                           fontsize=10, color='r')

w = grat.gmap.crval[0] + 0.2*grat.gmap.cdelt[0]
cv = grat.gmap.crval[1]
inlab2 = grat.Insidelabels(wcsaxis=0, world=w, constval=cv,
                           deltapy=0.1, rotation=20, 
                           fontsize=10, color='c')

pixel = grat.gmap.topixel((w,grat.gmap.crval[1]))
frame.plot( (pixel[0],), (pixel[1],), 'o', color='red' )

# 9. Plot the objects

mplim.plot()
plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                mu_manyrulers.py                                                                                    0000644 0000764 0000144 00000002223 11277362771 014071  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

fitsobject = maputils.FITSimage(externalheader=header)

fig = plt.figure(figsize=(7,7))
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
annim = fitsobject.Annotatedimage(frame)
grat = annim.Graticule(header)
x1 = 10; y1 = 1
x2 = 10; y2 = grat.pylim[1]

ruler1 = grat.Ruler(x1, y1, x2, y2, 0.0, 1.0)
ruler1.setp_labels(color='g')
x1 = x2 = grat.pxlim[1]
ruler2 = grat.Ruler(x1, y1, x2, y2, 0.5, 2.0, 
                   fmt='%3d', mscale=-1.5, fliplabelside=True)
ruler2.setp_labels(ha='left', va='center', color='b')
ruler3 = grat.Ruler(23*15,30,22*15,15, 0.5, 1, world=True, 
                    fmt=r"$%4.0f^\prime$", 
                    fun=lambda x: x*60.0, addangle=0)
ruler3.setp_labels(color='r')
ruler4 = grat.Ruler(1,800,800,800, 0.5, 2, fmt="%4.1f", addangle=90)
ruler4.setp_labels(color='c')
annim.plot()

plt.show()
                                                                                                                                                                                                                                                                                                                                                                             mu_markers.py                                                                                       0000644 0000764 0000144 00000000664 11300035327 013320  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

f = maputils.FITSimage("m101cdelt.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame, cmap="Set1")
grat = annim.Graticule()
annim.plot()

fn = 'smallworld.txt'
xp, yp = annim.positionsfromfile(fn, 's', cols=[0,1])
frame.plot(xp, yp, ',', color='b')

annim.interact_imagecolors()
plt.show()
                                                                            mu_movie.py                                                                                         0000644 0000764 0000144 00000002175 11273362537 013011  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  #!/usr/bin/env python
from kapteyn import wcsgrat, maputils
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()
frame = fig.add_subplot(1,1,1)

#Create a container to store the annotated images
movieimages = maputils.MovieContainer()

# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('ngc6946.fits')

# Get a the range of channels in the data cube
n3 = fitsobject.hdr['NAXIS3']
ch = range(1,n3)
vmin, vmax = fitsobject.get_dataminmax()
print "Vmin, Vmax of data in cube:", vmin, vmax
cmap = None

# Start to build and store the annotated images
first = True
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
   # Set limits as in: fitsobject.set_limits(pxlim=(150,350), pylim=(200,350))
   mplim = fitsobject.Annotatedimage(frame, cmap=cmap, clipmin=vmin, clipmax=vmax)
   mplim.Image()
   mplim.plot()
   if first:
      mplim.interact_imagecolors()
      cmap = mplim.cmap
   movieimages.append(mplim, visible=first)
   first = False

movieimages.movie_events()

# Draw the graticule lines and plot WCS labels
grat = mplim.Graticule()
grat.plot(frame)

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                   mu_negativecontours.py                                                                              0000644 0000764 0000144 00000000623 11270641163 015255  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  """ Show contour lines with different lines styles """
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("RAxDEC.fits")

fig = plt.figure(figsize=(8,6))
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
cont = mplim.Contours(levels=[-500,-300, 0, 300, 500], negative="dotted")
cont.setp_label()
mplim.plot()
mplim.interact_toolbarinfo()

plt.show()
                                                                                                             mu_pixellabels.py                                                                                   0000644 0000764 0000144 00000000650 11277356771 014201  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(4,4))
frame = fig.add_subplot(1,1,1)

fitsobject = maputils.FITSimage("m101.fits")
annim = fitsobject.Annotatedimage(frame)
annim.Pixellabels(plotaxis="bottom", color="r")
annim.Pixellabels(plotaxis="right", color="b", markersize=10)
annim.Pixellabels(plotaxis="top", color="g", markersize=-10, va='top')

annim.plot()
plt.show()

                                                                                        mu_projection.py                                                                                    0000644 0000764 0000144 00000000737 11275544316 014047  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

print "Projection object from FITSimage:" 
fitsobj = maputils.FITSimage("mclean.fits")
print "crvals:", fitsobj.convproj.crval
fitsobj.set_imageaxes(1,3)
print "crvals after axes specification:", fitsobj.convproj.crval
fitsobj.set_spectrans("VOPT-???")
print "crvals after setting spectral translation:", fitsobj.convproj.crval

print "Projection object from Annotatedimage:"
annim = fitsobj.Annotatedimage()
print "crvals:", annim.projection.crval
                                 mu_savecolormap.py                                                                                  0000644 0000764 0000144 00000001153 11270016672 014351  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  """Show interaction options"""
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
#f.set_limits((100,500),(200,400))

fig = plt.figure(figsize=(9, 7))
frame = fig.add_subplot(1, 1, 1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="m101.lut")
mplim.cmap.set_bad('w')
ima = mplim.Image()
mplim.Pixellabels()
mplim.Colorbar(label="Unknown unit")
mplim.plot()

mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()


                                                                                                                                                                                                                                                                                                                                                                                                                     mu_sethistogram.py                                                                                  0000644 0000764 0000144 00000001153 11270067375 014375  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  """Show interaction options"""
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
#f.set_limits((100,500),(200,400))

fig = plt.figure(figsize=(9, 7))
frame = fig.add_subplot(1, 1, 1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

mplim = f.Annotatedimage(frame, cmap="mousse.lut")
mplim.cmap.set_bad('w')
mplim.Image()
mplim.set_histogrameq()
mplim.Pixellabels()
mplim.Colorbar()
mplim.plot()


mplim.interact_toolbarinfo()
mplim.interact_imagecolors()
mplim.interact_writepos()

plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                     mu_shapes.py                                                                                        0000644 0000764 0000144 00000001641 11273362637 013153  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  #!/usr/bin/env python
from kapteyn import wcsgrat, maputils, ellinteract
from matplotlib import pylab as plt

# Get connected to Matplotlib
fig = plt.figure()


movieimages = maputils.MovieContainer()


# Create a maputils FITS object from a FITS file on disk
fitsobject = maputils.FITSimage('rense.fits')

#ch = [10,15,20,25,30,35]
#ch = range(15,85)
ch = [20, 30]
count = 0
vmin, vmax = fitsobject.get_dataminmax()
print "Vmin, Vmax:", vmin, vmax
for i in ch:
   fitsobject.set_imageaxes(1,2, slicepos=i)
#   fitsobject.set_limits()
   frame = fig.add_subplot(1,2,count+1)
   # Create an image to be used in Matplotlib
   mplim = fitsobject.Annotatedimage(frame)
   mplim.Image()
   count += 1
   # Draw the graticule lines and plot WCS labels
   mplim.Graticule()
   mplim.plot()
   movieimages.append(mplim)


# movieimages.movie_events()


shapes = ellinteract.Shapecollection(movieimages.mplim, fig, wcs=True)

plt.show()

                                                                                               mu_simplecontours.py                                                                                0000644 0000764 0000144 00000000411 11277110627 014742  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
cont = annim.Contours()
annim.plot()

print "Levels=", cont.clevels

plt.show()

                                                                                                                                                                                                                                                       mu_simple.py                                                                                        0000644 0000764 0000144 00000000257 11274532642 013157  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
mplim = f.Annotatedimage()
im = mplim.Image()
mplim.plot()

plt.show()

                                                                                                                                                                                                                                                                                                                                                 mu_simplewithframe.py                                                                               0000644 0000764 0000144 00000000350 11274532562 015061  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
im = mplim.Image()
mplim.plot()

plt.show()

                                                                                                                                                                                                                                                                                        mu_skyout.py                                                                                        0000644 0000764 0000144 00000002530 11276340631 013215  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn.wcs import galactic, equatorial, fk4_no_e, fk5
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
                                                                                                                                                                        mu_spectraltypes.py                                                                                 0000644 0000764 0000144 00000002630 11274557221 014565  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

# Read header of FITS file
f = maputils.FITSimage('mclean.fits')

# Matplotlib 
fig = plt.figure(figsize=(7,10))
fig.subplots_adjust(left=0.12, bottom=0.05, right=0.97, 
                    top=0.97, wspace=0.20, hspace=0.90)


# Get the projection object to get allowed spectral translations
altspec = f.proj.altspec
k = len(altspec) + 1
frame = fig.add_subplot(k,1,1)

# Limit range in x to neighbourhood of CRPIX
crpix = f.proj.crpix[f.proj.specaxnum-1]
xlim = (crpix-5, crpix+5)
f.set_imageaxes(3,2)
f.set_limits(pxlim=xlim)
mplim = f.Annotatedimage(frame)
mplim.set_aspectratio(0.002)

print "Native system", f.proj.ctype[f.proj.specaxnum-1]
grat = mplim.Graticule(boxsamples=3)
grat.setp_tick(plotaxis="bottom", fmt="%.5g")
grat.setp_plotaxis(("bottom", "left"), fontsize=9)
grat.setp_plotaxis("bottom", color='r')
grat.setp_tick(wcsaxis=(0,1), fontsize='8')

mplim.plot()

print "Spectral translations"
for i, ast in enumerate(altspec):
   print i, ast
   frame = fig.add_subplot(k,1,i+2)
   mplim = f.Annotatedimage(frame)
   mplim.set_aspectratio(0.002)
   grat = mplim.Graticule(spectrans=ast[0], boxsamples=3)
   grat.setp_tick(plotaxis="bottom", fmt="%g")
   grat.setp_plotaxis("bottom", label=ast[0]+' '+ast[1], color='b', fontsize=9)
   grat.setp_plotaxis("left", fontsize=9)
   grat.setp_tick(wcsaxis=(0,1), fontsize='8')
   mplim.plot()

plt.show()
                                                                                                        mu_toworld2.py                                                                                      0000644 0000764 0000144 00000001154 11271076172 013435  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
f.set_imageaxes(1,3, slicepos=51)
annim = f.Annotatedimage()

# Which pixel coordinates correspond to CRVAL's?
crpix = annim.projection.crpix
print "CRPIX from header", crpix

# Convert these to world coordinates
x = crpix[0]; y = crpix[1]
lon, velo, lat  = annim.toworld(x, y, matchspatial=True)
print "lon, velo, lat =", lon, velo, lat
print "Should be equivalent to CRVAL:", annim.projection.crval

x, y, slicepos = annim.topixel(lon, velo, matchspatial=True)
print "Back to pixel coordinates: x, y =", x, y, slicepos 

                                                                                                                                                                                                                                                                                                                                                                                                                    mu_toworld3.py                                                                                      0000644 0000764 0000144 00000001210 11271100172 013413  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
f.set_imageaxes(1,3, slicepos=51)
f.set_spectrans("FREQ-???")
annim = f.Annotatedimage()

# Which pixel coordinates correspond to CRVAL's?
crpix = annim.projection.crpix
print "CRPIX from header", crpix

# Convert these to world coordinates
x = crpix[0]; y = crpix[1]
lon, velo, lat  = annim.toworld(x, y, matchspatial=True)
print "lon, velo, lat =", lon, velo, lat
print "Should be equivalent to CRVAL:", annim.projection.crval

x, y, slicepos = annim.topixel(lon, velo, matchspatial=True)
print "Back to pixel coordinates: x, y =", x, y, slicepos 

                                                                                                                                                                                                                                                                                                                                                                                        mu_toworld4.py                                                                                      0000644 0000764 0000144 00000000604 11271103123 013421  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
#f.set_imageaxes(1, 3, slicepos=51)
annim = f.Annotatedimage()

x = [10, 50, 300, 399]
y = [1, 44, 88, 401]

# Convert these to world coordinates
lon, velo = annim.toworld(x, y)
print "lon, velo =", lon, velo

x, y = annim.topixel(lon, velo)
print "Back to pixel coordinates: x, y =", x, y

                                                                                                                            mu_toworld5.py                                                                                      0000644 0000764 0000144 00000001076 11273313375 013444  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
f.set_imageaxes(1, 3, slicepos=51)
annim = f.Annotatedimage()

x = [10, 50, 300, 399]
y = [1, 44, 88, 100]

# Convert these to world coordinates
#lon, velo = annim.toworld(x, y)
lon, velo, lat = annim.toworld(x, y, matchspatial=True)
print "lon, velo lat=", lon, velo, lat

# We are not interested in the pixel coordinate of the slice
# because we know it is 52. Therefore we omit 'matchspatial'
x, y = annim.topixel(lon, velo)
print "Back to pixel coordinates: x, y =", x, y

                                                                                                                                                                                                                                                                                                                                                                                                                                                                  mu_toworld.py                                                                                       0000644 0000764 0000144 00000000474 11271075760 013361  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils

# The set is 3-dim. but default the first two axes are
# used to extract the image data
f = maputils.FITSimage("ngc6946.fits")

annim = f.Annotatedimage()
x = 200; y = 350
lon, lat  = annim.toworld(x,y)
print "lon, lat =", lon, lat

x, y = annim.topixel(lon, lat)
print "x, y = ", x, y
                                                                                                                                                                                                    mu_wave.py                                                                                          0000644 0000764 0000144 00000001023 11260222703 012605  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             mu_withimage.py                                                                                     0000644 0000764 0000144 00000000642 11277023226 013636  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((180,344), (180,344))

fig = plt.figure(figsize=(6,6))
frame = fig.add_axes([0,0,1,1])

annim = fitsobj.Annotatedimage(frame, cmap="spectral", clipmin=10000, clipmax=15500)
annim.Image(interpolation='spline36')
print "clip min, max:", annim.clipmin, annim.clipmax
annim.plot()

plt.show()

                                                                                              mu_xvruler.py                                                                                       0000644 0000764 0000144 00000001755 11276614032 013375  0                                                                                                    ustar   vogelaar                        users                                                                                                                                                                                                                  from kapteyn import maputils
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   