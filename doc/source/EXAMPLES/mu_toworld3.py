from kapteyn import maputils

f = maputils.FITSimage("ngc6946.fits")
# Get an XV slice at DEC=51
f.set_imageaxes(1,3, slicepos=51)
f.set_spectrans("FREQ-???")
annim = f.Annotatedimage()

# Which pixel coordinates correspond to CRVAL's?
crpix = annim.projection.crpix
print("CRPIX from header", crpix)

# Convert these to world coordinates
x = crpix[0]; y = crpix[1]
lon, velo, lat  = annim.toworld(x, y, matchspatial=True)
print("lon, velo, lat =", lon, velo, lat)
print("Should be equivalent to CRVAL:", annim.projection.crval)

x, y, slicepos = annim.topixel(lon, velo, matchspatial=True)
print("Back to pixel coordinates: x, y =", x, y, slicepos) 

