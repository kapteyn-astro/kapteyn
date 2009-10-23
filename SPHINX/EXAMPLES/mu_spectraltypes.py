from kapteyn import maputils
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
