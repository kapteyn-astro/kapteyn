from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy

# Read first image as base 
Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
Basefits.set_imageaxes(promptfie=maputils.prompt_imageaxes)
Basefits.set_limits(promptfie=maputils.prompt_box)

# Get data from a second image. This sets the spatial output
Contourfits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
Contourfits.set_imageaxes(promptfie=maputils.prompt_imageaxes)
print "boxdat contour 0,0", Contourfits.boxdat[0,0]
print "Contourfits slicepos", Contourfits.slicepos

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

baseim = Basefits.Annotatedimage(frame)
baseim.Image()
baseim.Graticule()

# Set parameters for the interpolation routine
pars = dict(cval=numpy.nan, order=1)
Reprojfits = Contourfits.reproject_to(Basefits, interpol_dict=pars)
print Reprojfits.boxdat
overlayim = Basefits.Annotatedimage(frame, boxdat=Reprojfits.boxdat)
print overlayim.data

mi, ma = overlayim.clipmin, overlayim.clipmax
prompt = "Enter contour levels between %g and %g: " % (mi, ma)
levels = maputils.getnumbers(prompt)
overlayim.Contours(levels=levels, colors='r')
baseim.Contours(levels=levels, colors='g', linewidths=1)

Reprojfits.writetofits("contours.fits", clobber=True)
#print "Difference between map and overlay:"
#print Basefits.boxdat-Reprojfits.boxdat

baseim.plot()
overlayim.plot()
baseim.interact_toolbarinfo()
baseim.interact_imagecolors()

plt.show()

