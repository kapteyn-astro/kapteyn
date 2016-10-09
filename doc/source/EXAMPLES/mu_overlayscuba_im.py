from kapteyn import maputils
from matplotlib import pyplot as plt

file1 = "scuba850_AFGL2591.fits"
file2 = "13CO_3-2_integ_regrid.fits"

# Read first image as base 
Basefits = maputils.FITSimage(file1)
Secondfits = maputils.FITSimage(file2)

pars = dict(cval=0.0, order=1)
Reprojfits = Secondfits.reproject_to(Basefits, interpol_dict=pars)

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

baseim = Basefits.Annotatedimage(frame)
baseim.Image(alpha=1.0)
baseim.set_histogrameq()
baseim.set_blankcolor('k')
baseim.Graticule()

overlayim = Basefits.Annotatedimage(frame, cmap='OrRd', 
                                    boxdat=Reprojfits.boxdat)
levels = list(range(50,200,20))
#overlayim.Contours(levels=levels, colors='w')
overlayim.Image(alpha=0.8)
baseim.set_histogrameq()

baseim.plot()
overlayim.plot()
baseim.interact_toolbarinfo()
baseim.interact_imagecolors()

plt.show()

