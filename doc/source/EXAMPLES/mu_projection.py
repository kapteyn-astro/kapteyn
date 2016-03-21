from kapteyn import maputils

print "Projection object from FITSimage:" 
fitsobj = maputils.FITSimage("mclean.fits")
print "crvals:", fitsobj.convproj.crval
fitsobj.set_imageaxes(1,3)
print "crvals after axes specification:", fitsobj.convproj.crval
fitsobj.set_spectrans("VOPT-???")
print "crvals after setting spectral translation:", fitsobj.convproj.crval

print "Projection object from Annotatedimage:"
annim = fitsobj.Annotatedimage()
print "crvals:", annim.projection.crval
