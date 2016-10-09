from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
f.set_imageaxes(1, 3, slicepos=51)
annim = f.Annotatedimage()

x = [10, 50, 300, 399]
y = [1, 44, 88, 100]

# Convert these to world coordinates
#lon, velo = annim.toworld(x, y)
lon, velo, lat = annim.toworld(x, y, matchspatial=True)
print("lon, velo lat=", lon, velo, lat)

# We are not interested in the pixel coordinate of the slice
# because we know it is 52. Therefore we omit 'matchspatial'
x, y = annim.topixel(lon, velo)
print("Back to pixel coordinates: x, y =", x, y)

