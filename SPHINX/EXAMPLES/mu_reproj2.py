from kapteyn import maputils

# Read first image as base 
Basefits = maputils.FITSimage("ra_pol_freq_dec.fits")

# Get data from a FITS file. This is the data that
# should be reprojected to fit the header of Basefits.
Secondfits = maputils.FITSimage("dec_pol_freq_ra.fits")

# Now we want to re-project the data of the Base object onto
# the wcs of the second object. This is done with the
# reproject_to() method of the first FITSimage object
# (the data object) with the header of the second FITSimage
# object as parameter. This results in a new FITSimage object

Reprojfits = Basefits.reproject_to(Secondfits.hdr, 
                                   pxlim=(-10,50), pylim=(-10,50),
                                   plimlo=1, plimhi=1)

# Write the result to disk
Reprojfits.writetofits("reproj.fits", clobber=True)
