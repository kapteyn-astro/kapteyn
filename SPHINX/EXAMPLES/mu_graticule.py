from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(50,440), pylim=(50,450))

fig = plt.figure(figsize=(8,5.5))
frame = fig.add_axes((0.05, 0.1, 0.8, 0.7))
fig.text(0.5, 0.96, "Combination of plot objects", 
         horizontalalignment='center',
         fontsize=14, color='r')

annim = f.Annotatedimage(frame, clipmin=3000, clipmax=15000)
cont = annim.Contours(levels=range(8000,14000,1000))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=2)
cb = annim.Colorbar(clines=False, orientation='vertical', fontsize=8)
gr = annim.Graticule()
gr.setp_ticklabel(wcsaxis=0, fmt='HMS')
ilab = gr.Insidelabels(color='b', ha='left')
ilab.setp_label(position='14h03m0s', fontsize=15) 

# Plot a second graticule for the galactic sky system
gr2 = annim.Graticule(deltax=7.5/60, deltay=5.0/60,
                      skyout="galactic", 
                      visible=True)
gr2.setp_axislabel(plotaxis=("top","right"), label="Galactic l,b",
                  color='g', visible=True)
gr2.setp_axislabel(plotaxis=("left","bottom"), visible=False)
gr2.set_tickmode(plotaxis=("top","right"), mode="Native")
gr2.set_tickmode(plotaxis=("left","bottom"), mode="NO")
gr2.setp_ticklabel(wcsaxis=(0,1), color='g')
gr2.setp_ticklabel(plotaxis='right', fmt='DMs')
gr2.setp_tickmark(plotaxis='right', markersize=8, markeredgewidth=2)
gr2.setp_gratline(wcsaxis=(0,1), color='g')
annim.Ruler(x1=120, y1=100, x2=120, y2=330, step=1/60.0)
r1 = annim.Ruler(pos1='ga 102d0m, 59d50m', pos2='ga 102d7m30s, 59d50m',  
                 world=True, step=1/60.0)
r1.setp_line(color='#ff22ff', lw=6)
r1.setp_label(color='m')
annim.Pixellabels(plotaxis='top', va='top')
pl = annim.Pixellabels(plotaxis='right')
pl.setp_marker(color='c', markersize=10)
pl.setp_label(color='m')

annim.plot()
plt.show()

