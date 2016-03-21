from kapteyn import maputils
from matplotlib import pylab as plt

header = { 'NAXIS'  : 3,
           'BUNIT'  : 'w.u.',
           'CDELT1' : -1.200000000000E-03,
           'CDELT2' : 1.497160000000E-03, 'CDELT3' : 97647.745732, 
           'CRPIX1' : 5, 'CRPIX2' : 6, 'CRPIX3' : 32,
           'CRVAL1' : 1.787792000000E+02, 'CRVAL2' : 5.365500000000E+01,
           'CRVAL3' : 1378471216.4292786,
           'CTYPE1' : 'RA---NCP', 'CTYPE2' : 'DEC--NCP', 'CTYPE3' : 'FREQ-OHEL',
           'CUNIT1' : 'DEGREE', 'CUNIT2' : 'DEGREE', 'CUNIT3' : 'HZ',
           'DRVAL3' : 1.050000000000E+03,
           'DUNIT3' : 'KM/S',
           'FREQ0'  : 1.420405752e+9,
           'INSTRUME' : 'WSRT',
           'NAXIS1' : 100, 'NAXIS2' : 100, 'NAXIS3' : 64
}


fig = plt.figure(figsize=(7,10))
fig.suptitle("Axis label tricks (spectral+spatial offset)", fontsize=14, color='r')
fig.subplots_adjust(left=0.18, bottom=0.10, right=0.90,
                    top=0.90, wspace=0.95, hspace=0.20)
frame = fig.add_subplot(3,2,1)
f = maputils.FITSimage(externalheader=header)
f.set_imageaxes(3,2, slicepos=1)
annim = f.Annotatedimage(frame)
# Default labeling
grat = annim.Graticule()
grat.setp_tick(plotaxis="bottom", rotation=90)

frame = fig.add_subplot(3,2,2)
annim = f.Annotatedimage(frame)
# Spectral axis with start and increment
grat = annim.Graticule(startx="1.378 Ghz", deltax="2 Mhz", starty="53d42m")
grat.setp_tick(plotaxis="bottom", fontsize=7, fmt='%.3f%+9e')

frame = fig.add_subplot(3,2,3)
annim = f.Annotatedimage(frame)
# Spectral axis with start and increment
grat = annim.Graticule(spectrans="WAVE", startx="21.74 cm", 
                       deltax="0.04 cm", starty="0.5")

frame = fig.add_subplot(3,2,4)
annim = f.Annotatedimage(frame)
# Spectral axis with start and increment
grat = annim.Graticule(spectrans="VOPT", startx="9120 km/s", 
                       deltax="400 km/s", unitsy="arcsec")
grat.setp_tick(plotaxis="bottom", fontsize=7)

frame = fig.add_subplot(3,2,5)
annim = f.Annotatedimage(frame)
# Spectral axis with start and increment and unit
grat = annim.Graticule(spectrans="VOPT", startx="9000 km/s", 
                       deltax="400 km/s", unitsx="km/s")

frame = fig.add_subplot(3,2,6)
annim = f.Annotatedimage(frame)
# Spectral axis with start and increment and formatter function
grat = annim.Graticule(spectrans="VOPT", startx="9000 km/s", deltax="400 km/s")
grat.setp_tick(plotaxis="bottom", fmt='%g', fun=lambda x:x/1000.0)
grat.setp_axislabel(plotaxis="bottom", label="Optical velocity (Km/s)")

maputils.showall()