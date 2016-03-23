from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy

# Read first image as base 
Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
print(type(Basefits), isinstance(Basefits, maputils.FITSimage))

# Get data from a second image. This is the data that 
# should be reprojected to fit the header of Basefits.
Secondfits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
#Secondfits.set_imageaxes(promptfie=maputils.prompt_imageaxes)
#Secondfits.set_limits(promptfie=maputils.prompt_box)

# Now we want to overlay the data of this Base fits object onto
# the wcs of the second fits object. This is done with the 
# reproject_to() method of
# the first FITSimage object (the data object) with the second
# FITSimage object as parameter. This results in a new fits file

#Reprojfits = Basefits.reproject_to(Secondfits.hdr, plimlo=(2,1), plimhi=(2,1))
#Reprojfits = Basefits.reproject_to(Secondfits.hdr, pxlim=(100,400),  pylim=(100,400))
Reprojfits = Basefits.reproject_to(Secondfits.hdr)
Reprojfits.writetofits("reproj.fits", clobber=True)
