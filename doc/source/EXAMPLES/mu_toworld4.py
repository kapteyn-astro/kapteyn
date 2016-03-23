from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
#f.set_imageaxes(1, 3, slicepos=51)
annim = f.Annotatedimage()

x = [10, 50, 300, 399]
y = [1, 44, 88, 401]

# Convert these to world coordinates
lon, velo = annim.toworld(x, y)
print("lon, velo =", lon, velo)

x, y = annim.topixel(lon, velo)
print("Back to pixel coordinates: x, y =", x, y)

