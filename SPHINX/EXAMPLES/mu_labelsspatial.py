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


fig = plt.figure(figsize=(7,7))
fig.suptitle("Axis labels for spatial maps", fontsize=14, color='r')
fig.subplots_adjust(left=0.18, bottom=0.10, right=0.90,
                    top=0.90, wspace=0.95, hspace=0.20)
frame = fig.add_subplot(2,2,1)
f = maputils.FITSimage(externalheader=header)
f.set_imageaxes(1,2)
annim = f.Annotatedimage(frame)
# Default labeling
grat = annim.Graticule()

frame = fig.add_subplot(2,2,2)
annim = f.Annotatedimage(frame)
# Plot labels with start position and increment
grat = annim.Graticule(startx='11h55m', deltax="15 hmssec", deltay="3 arcmin")

frame = fig.add_subplot(2,2,3)
annim = f.Annotatedimage(frame)
# Plot labels in string only
grat = annim.Graticule(startx='11h55m 11h54m30s')
grat.setp_tick(plotaxis="bottom", texsexa=False)

frame = fig.add_subplot(2,2,4)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(startx="178.75 deg", deltax="6 arcmin", unitsx="degree")
grat.setp_ticklabel(plotaxis="left", fmt="s")

maputils.showall()