from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj= maputils.FITSimage('ngc6946.fits')
newaspect = 1/5.0        # Needed for XV maps

fig = plt.figure(figsize=(20/2.54, 25/2.54))
fig.subplots_adjust(left=0.18)
labelx = -0.10           # Fix the  position in x for labels along y

# fig 1. Spatial map, default axes are 1 & 2
frame1 = fig.add_subplot(4,1,1)
mplim1 = fitsobj.Annotatedimage(frame1)
mplim1.Image()
graticule1 = mplim1.Graticule(deltax=15*2/60.0)

# fig 2. Velocity - Dec
frame2 = fig.add_subplot(4,1,2)
fitsobj.set_imageaxes('vel', 'dec')
mplim2 = fitsobj.Annotatedimage(frame2)
mplim2.Image()
graticule2 = mplim2.Graticule()
graticule2.setp_axislabel(plotaxis='left', xpos=labelx)

# fig 3. Velocity - Dec (Version without offsets)
frame3 = fig.add_subplot(4,1,3)
mplim3 = fitsobj.Annotatedimage(frame3)
mplim3.Image()
graticule3 = mplim3.Graticule(offsety=False)
graticule3.setp_axislabel(plotaxis='left', xpos=labelx)
graticule3.setp_ticklabel(plotaxis="left", fmt='DMs')

# fig 4. Velocity - R.A.
frame4 = fig.add_subplot(4,1,4)
fitsobj.set_imageaxes('vel','ra')
mplim4 = fitsobj.Annotatedimage(frame4)
mplim4.Image()
graticule4 = mplim4.Graticule(offsety=False)
graticule4.setp_axislabel(plotaxis='left', xpos=labelx)
graticule4.setp_ticklabel(plotaxis="left", fmt='HMs')
graticule4.Insidelabels(wcsaxis=0, constval='20h34m',
                        rotation=90, fontsize=10,
                        color='r', ha='right')
graticule4.Insidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='y')
mplim4.Minortickmarks(graticule4)

#Apply new aspect ratio for the XV maps
mplim2.set_aspectratio(newaspect)
mplim3.set_aspectratio(newaspect)
mplim4.set_aspectratio(newaspect)

maputils.showall()