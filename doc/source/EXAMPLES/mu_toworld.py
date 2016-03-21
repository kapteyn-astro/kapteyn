from kapteyn import maputils

# The set is 3-dim. but default the first two axes are
# used to extract the image data
f = maputils.FITSimage("ngc6946.fits")

annim = f.Annotatedimage()
x = 200; y = 350
lon, lat  = annim.toworld(x,y)
print "lon, lat =", lon, lat

x, y = annim.topixel(lon, lat)
print "x, y = ", x, y
