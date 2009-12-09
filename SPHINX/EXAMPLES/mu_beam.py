from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('m101test.fits')
annim = fitsobj.Annotatedimage()
annim.Image()
grat = annim.Graticule()
#beam = annim.Beam(210.9619, 54.261039, 0.01, 0.01, 0, hatch='*')


lat = 54
lon = 210.8025441
beam = annim.Beam(lon, lat, 0.08, 0.05, 90, fc='g', fill=True, alpha=0.6)
grat.Ruler(lon, lat, lon+0.08, lat, world=True, step=0.01, 
           lambda0=0.0, 
           fmt=r"$%g^\circ$", fun=lambda x: x, color='r')
grat.Ruler(lon, lat, lon, lat++0.08, world=True, step=0.01, 
           lambda0=0.0, 
           fmt=r"$%g^\circ$", fun=lambda x: x, color='b')
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
plt.show()

