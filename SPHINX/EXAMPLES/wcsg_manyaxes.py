from kapteyn import wcsgrat
from matplotlib import pyplot as plt
import pyfits

# 1. Read the header
hdulist = pyfits.open('manyaxes.fits')
header = hdulist[0].header

# 2. Create a graticule
grat = wcsgrat.Graticule(header, axnum=(1,4), starty=1000, deltay=10)

# 3. Show the calculated world coordinates along y-axis
print "The world coordinates along the y-axis:", grat.ystarts

# 4. Show header information in attributes of the Projection object
print "CRVAL, CDELT from header:", grat.gmap.crval, grat.gmap.cdelt

# 5. Set a number of properties of the graticules and plot axes
grat.setp_tick(plotaxis=wcsgrat.bottom, 
               fun=lambda x: x/1.0e9, fmt="%.4f",
               rotation=-30 )
grat.setp_plotaxis(wcsgrat.bottom, label="Frequency (GHz)")
grat.setp_gratline(wcsaxis=0, position=grat.gmap.crval[0], 
                   tol=0.5*grat.gmap.cdelt[0], color='r')
grat.setp_tick(plotaxis=wcsgrat.left, position=1000, color='m', fmt="I")
grat.setp_tick(plotaxis=wcsgrat.left, position=1010, color='b', fmt="Q")
grat.setp_tick(plotaxis=wcsgrat.left, position=1020, color='r', fmt="U")
grat.setp_tick(plotaxis=wcsgrat.left, position=1030, color='g', fmt="V")
grat.setp_plotaxis(wcsgrat.left, label="Stokes parameters")

# 6. Create a Matplotlib Figure and Axes instance
fig = plt.figure(figsize=(7,7))
frame = fig.add_subplot(1,1,1)

# 7. Set a title for this frame
title = r"""Polarization as function of frequency at:
            $(\alpha_0,\delta_0) = (121^o,53^o)$"""
t = frame.set_title(title, color='#006400')
t.set_y(1.01)

# 8. Add labels inside plot
inlabs = grat.insidelabels(wcsaxis=0, constval=1015, 
                           deltapx=-0.15, rotation=90, 
                           fontsize=10, color='r')

w = grat.gmap.crval[0] + 0.2*grat.gmap.cdelt[0]
cv = grat.gmap.crval[1]
inlab2 = grat.insidelabels(wcsaxis=0, world=w, constval=cv,
                           deltapy=0.1, rotation=20, 
                           fontsize=10, color='c')

pixel = grat.gmap.topixel((w,grat.gmap.crval[1]))
frame.plot( (pixel[0],), (pixel[1],), 'o', color='red' )

# 9. Tell the graticule and its derived object where it can plot itself
gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add([grat,inlabs,inlab2])

plt.show()
