from kapteyn import maputils
from matplotlib import pyplot as plt

# Read header of FITS file
f = maputils.FITSimage('mclean.fits')

# Matplotlib 
fig = plt.figure(figsize=(7,10))
fig.subplots_adjust(left=0.12, bottom=0.05, right=0.97, 
                    top=0.97, wspace=0.22, hspace=0.90)
fig.text(0.05,0.5,"Radial offset latitude", rotation=90, 
         fontsize=14, va='center')

# Get the projection object to get allowed spectral translations
altspec = f.proj.altspec
crpix = f.proj.crpix[f.proj.specaxnum-1]
altspec.insert(0, (None, ''))  # Add native to list
k = len(altspec) + 1
frame = fig.add_subplot(k,1,1)

# Limit range in x to neighbourhood of CRPIX
xlim = (crpix-5, crpix+5)
f.set_imageaxes(3,2)
f.set_limits(pxlim=xlim)
mplim = f.Annotatedimage(frame)
mplim.set_aspectratio(0.002)

print "Native system", f.proj.ctype[f.proj.specaxnum-1], f.proj.cunit[f.proj.specaxnum-1], 

print "Spectral translations"
for i, ast in enumerate(altspec):
   print i, ast
   frame = fig.add_subplot(k,1,i+1)
   mplim = f.Annotatedimage(frame)
   mplim.set_aspectratio(0.002)
   grat = mplim.Graticule(spectrans=ast[0], boxsamples=3)
   grat.setp_ticklabel(plotaxis="bottom", fmt="%g")
   unit = ast[1]
   ctype = ast[0]
   if ctype == None:
      ctype = "Frequency (Hz) without translation"
   grat.setp_axislabel(plotaxis="bottom", 
                       label=ctype+' '+unit, color='b', fontsize=10)
   grat.setp_axislabel("left", visible=False)
   grat.setp_ticklabel(wcsaxis=(0,1), fontsize='8')
   mplim.plot()

plt.show()
