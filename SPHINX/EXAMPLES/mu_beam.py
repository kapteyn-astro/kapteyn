from kapteyn import maputils
from matplotlib import pyplot as plt
from math import cos, radians

fitsobj = maputils.FITSimage('m101.fits')
annim = fitsobj.Annotatedimage()
annim.Image()
grat = annim.Graticule()
#beam = annim.Beam(210.9619, 54.261039, 0.01, 0.01, 0, hatch='*')

fwhm_maj = 0.08
fwhm_min = 0.06
lat = 54.347395233845
lon = 210.80254413455
beam = annim.Beam(fwhm_maj, fwhm_min, 90, xc=lon, yc=lat, 
                  fc='g', fill=True, alpha=0.6)
pos = '210.80254413455 deg, 54.347395233845 deg'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='m', fill=True, alpha=0.6)
pos = '14h03m12.6105s 54d20m50.622s'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='y', fill=True, alpha=0.6)
pos = 'ga 102.0354152 {} 59.7725125'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='g', fill=True, alpha=0.6)
pos = 'ga 102d02m07.494s {} 59.7725125'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='b', fill=True, alpha=0.6)
pos = '{ecliptic,fk4, j2000} 174.3674627 {} 59.7961737'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='r', fill=True, alpha=0.6)
pos = '{eq, fk4-no-e, B1950} 14h01m26.4501s {} 54d35m13.480s'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
pos = '{eq, fk4-no-e, B1950, F24/04/55} 14h01m26.4482s {} 54d35m13.460s'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
pos = '{ecl} 174.367764 {} 59.79623457'
beam = annim.Beam(fwhm_maj, fwhm_min, pos=pos, fc='c', fill=True, alpha=0.6)
pos = '53 58'     # Pixels
beam = annim.Beam(0.04, 0.02, pa=30, pos=pos, fc='y', fill=True, alpha=0.4)
pos = '14h03m12.6105s 58'
beam = annim.Beam(0.04, 0.02, pa=-30, pos=pos, fc='y', fill=True, alpha=0.4)

grat.Ruler(lon, lat, lon+fwhm_min/1.99/cos(radians(54.20)), lat, 
           world=True, step=0.01, lambda0=0.0, 
           fmt=r"$%.2f^\circ$", fun=lambda x: x, color='r')
grat.Ruler(lon, lat, lon, lat+fwhm_maj/1.99, world=True, step=0.01, 
           lambda0=0.0, 
           fmt=r"$%.2f^\circ$", fun=lambda x: x, color='b')
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
plt.show()
