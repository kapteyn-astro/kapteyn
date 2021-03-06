from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy

# Read first image as base 
Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
Basefits.set_imageaxes(promptfie=maputils.prompt_imageaxes)
Basefits.set_limits(promptfie=maputils.prompt_box)

# Get data from a second image. This is the data that 
# should be reprojected to fit the header of Basefits.
Reprojfits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
Reprojfits.set_imageaxes(promptfie=maputils.prompt_imageaxes)
Reprojfits.set_limits(promptfie=maputils.prompt_box)

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

baseim = Basefits.Annotatedimage(frame)
baseim.Image()

# Set parameters for the interpolation routine
pars = dict(cval=numpy.nan, order=1)
overlayim = Basefits.Annotatedimage(frame, 
                                    overlay_src=Reprojfits, 
                                    overlay_dict=pars)

mi, ma = overlayim.clipmin, overlayim.clipmax
levels = eval(input("Enter contour levels between %g and %g: " % (mi, ma))) 
if not maputils.issequence(levels):
   levels = [levels]
overlayim.Contours(levels=levels, colors='r')

# Write overlay data to FITS file with same structure as 
# the base FITS file
Basefits.writetofits(boxdat=overlayim.data)
#x = overlayim.data[numpy.isfinite(overlayim.data)]

baseim.plot()
overlayim.plot()
baseim.interact_toolbarinfo()

plt.show()

