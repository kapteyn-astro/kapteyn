from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('m101.fits')
fig = plt.figure()
   
# Create an image to be used in Matplotlib
image = fitsobject.createMPLimage(fig)
image.set_aspectratio()
image.add_subplot(1,1,1)
image.imshow()

# Draw the graticule lines and plot WCS labels
graticule = image.makegrat()
ruler = graticule.ruler(208,201,310,351, 0.5)
pixellabels = graticule.pixellabels()
image.addspecial([graticule, ruler, pixellabels])

plt.show()

