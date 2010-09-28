from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

annim = f.Annotatedimage(frame, cmap='gist_yarg')
annim.Image()
grat = annim.Graticule()
grat.setp_gratline(color='0.75')

# Ellipse centered on crossing of two graticule lines
annim.Skypolygon("ellipse", cpos="14h03m 54d20m", major=100, minor=50,
                  pa=-30.0, units='arcsec', fill=False)

# Ellipse at given pixel coordinates
annim.Skypolygon("ellipse", cpos="10 10", major=100, minor=50,
                  pa=-30.0, units='arcsec', fc='c')  

# Circle with radius in arc minutes
annim.Skypolygon("ellipse", cpos="210.938480 deg 54.269206 deg", 
                  major=1.50, minor=1.50, units='arcmin', 
                  fc='g', alpha=0.3, lw=3, ec='r') 

# Rectangle at the projection center
annim.Skypolygon("rectangle", cpos="pc pc", major=200, minor=50,
                  pa=30.0, units='arcsec', ec='g', fc='b', alpha=0.3)

# Regular polygon with 6 angles at some position in galactic coordinates
annim.Skypolygon("npoly", cpos="ga 102d11m35.239s ga 59d50m25.734", 
                  major=150, nangles=6,
                  units='arcsec', ec='g', fc='y', alpha=0.3)

# Regular polygon 
annim.Skypolygon("npolygon", cpos="ga 102.0354152 ga 59.7725125", 
                  major=150, nangles=3,
                  units='arcsec', ec='g', fc='y', alpha=0.3)

lons = [210.969423, 210.984761, 210.969841, 210.934896, 210.894589,
        210.859949, 210.821008, 210.822413, 210.872040]
lats = [54.440575, 54.420249, 54.400778, 54.388611, 54.390166,
        54.396241, 54.416029, 54.436244, 54.454230]

annim.Skypolygon(prescription=None, lons=lons, lats=lats, fc='r', alpha=0.3)

annim.plot()
annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos(wcsfmt="%f",zfmt=None, pixfmt=None, hmsdms=False)

plt.show()

