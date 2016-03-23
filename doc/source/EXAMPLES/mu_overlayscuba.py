from kapteyn import maputils
from matplotlib import pyplot as plt

file1 = "scuba850_AFGL2591.fits"
file2 = "13CO_3-2_integ_regrid.fits"

# Read first image as base 
Basefits = maputils.FITSimage(file1)
Secondfits = maputils.FITSimage(file2)
Reprojfits = Secondfits.reproject_to(Basefits)

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

baseim = Basefits.Annotatedimage(frame)
baseim.Image()
baseim.set_histogrameq()
baseim.Graticule()

overlayim = Basefits.Annotatedimage(frame, boxdat=Reprojfits.boxdat)
levels = list(range(20,200,20))
overlayim.Contours(levels=levels, colors='w')

baseim.plot()
overlayim.plot()
baseim.interact_toolbarinfo()
baseim.interact_imagecolors()

plt.show()

