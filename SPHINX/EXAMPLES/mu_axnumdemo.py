from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage('ngc6946.fits')
newaspect = 1/5.0

fig = plt.figure(figsize=(20/2.54, 25/2.54))
labelx = -0.10         # Fixed position in x for all y labels
frame = fig.add_subplot(4,1,1)

# fig 1. Spatial map
mplim1 = f.Annotatedimage(frame)
graticule1 = mplim1.Graticule()

# fig 2. Velocity - Dec
frame2 = fig.add_subplot(4,1,2)
f.set_imageaxes(3,2)
mplim2 = f.Annotatedimage(frame2)
graticule2 = mplim2.Graticule()

# fig 3. Velocity - Dec (Version without offsets)
frame3 = fig.add_subplot(4,1,3)
mplim3 = f.Annotatedimage(frame3)
graticule3 = mplim3.Graticule(offsety=False)

# fig 4. Velocity - R.A.
frame4 = fig.add_subplot(4,1,4)
f.set_imageaxes(3,1)
mplim4 = f.Annotatedimage(frame4)
graticule4 = mplim4.Graticule(offsety=False)
graticule4.setp_tick(plotaxis="left", fun=lambda x: x+360, fmt="$%.1f^\circ$")

graticule4.Insidelabels(wcsaxis=0, constval=-51, rotation=90, fontsize=10, 
                        color='r', ha='right')
graticule4.Insidelabels(wcsaxis=1, fontsize=10, fmt="%.2f", color='b')

mplim1.set_aspectratio(newaspect)
mplim2.set_aspectratio(newaspect)
mplim3.set_aspectratio(newaspect)
mplim4.set_aspectratio(newaspect)

mplim1.plot()
mplim2.plot()
mplim3.plot()
mplim4.plot()

# Align left axis title in frame of graticule
graticule1.frame.yaxis.set_label_coords(labelx, 0.5)
graticule2.frame.yaxis.set_label_coords(labelx, 0.5)
graticule3.frame.yaxis.set_label_coords(labelx, 0.5)
graticule4.frame.yaxis.set_label_coords(labelx, 0.5)


plt.show()

