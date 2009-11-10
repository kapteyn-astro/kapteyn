from kapteyn import maputils
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
