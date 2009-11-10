from kapteyn import maputils
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


