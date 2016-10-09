from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy


#order = input("Enter order of spline interpolation in range 0..5: ")
order = 1
if order < 0: order = 0
if order > 5: order = 5

file1 = "m101OPT.fits"
Basefits = maputils.FITSimage(file1)
pxlim = (50,500); pylim = (50,500)
Basefits.set_limits(pxlim, pylim)

file2 = "m101HI.fits"
Overfits = maputils.FITSimage(file2)
Overfits.set_imageaxes(1,2)
# Not necessary to set limits if an overlay is required

fig = plt.figure(figsize=(6,7))
frame1 = fig.add_subplot(2,2,1)
frame2 = fig.add_subplot(2,2,2)
frame3 = fig.add_subplot(2,2,3)
frame4 = fig.add_subplot(2,2,4)
fs = 10      # Font size for titles

levels = [8000, 12000]

# Plot 1: Base
baseim = Basefits.Annotatedimage(frame1)
baseim.Image()
frame1.set_title("WCS1", fontsize=fs)

# Plot 2: Data with different wcs
overlayim = Overfits.Annotatedimage(frame2)
overlayim.Image()
overlayim.set_blankcolor('y')
frame2.set_title("WCS2", fontsize=fs)

# Plot 3: Base with contours reprojected from other source
baseim2 = Basefits.Annotatedimage(frame3)
baseim2.Image()
baseim2.Contours(levels=levels, colors='g')

# Filter the NaN's. Replace by 0.0 to be able tu use spline order > 1
#Overfits.boxdat[numpy.where(numpy.isnan(Overfits.boxdat))] = 0.0

# Set parameters for the interpolation routine
pars = dict(cval=numpy.nan, order=order)
Reprojfits = Overfits.reproject_to(Basefits, interpol_dict=pars)
overlayim2 = Basefits.Annotatedimage(frame3, boxdat=Reprojfits.boxdat)
overlayim2.Contours(levels=levels, colors='r')

frame3.set_title("Image WCS1 + \ncontours reprojected WCS2", fontsize=fs)
# Plot 4: Plot the difference between base and reprojection
x = Basefits.boxdat - overlayim2.data
print("Residual min, max, mean, std, sum:", x.flatten().min(), x.flatten().max(),\
      x.flatten().mean(), x.flatten().std(), x.flatten().sum())
diff = Basefits.Annotatedimage(frame4, boxdat=x)
diff.Image()
diff.set_histogrameq()
frame4.set_title(r"$\Delta$ = WCS1 - reprojected WCS2", fontsize=fs)

# User interaction
diff.interact_toolbarinfo()
diff.interact_imagecolors()
overlayim.interact_imagecolors()

maputils.showall()
