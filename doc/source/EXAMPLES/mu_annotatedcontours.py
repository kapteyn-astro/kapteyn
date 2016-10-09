from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits(pxlim=(200,350), pylim=(200,350))

fig = plt.figure()
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
cont = mplim.Contours(levels=list(range(10000,16000,1000)))
cont.setp_contour(linewidth=1)
cont.setp_contour(levels=11000, color='g', linewidth=3)

# Second contour set only for labels
cont2 = mplim.Contours(levels=(8000,9000,10000,11000))
cont2.setp_label(11000, colors='b', fontsize=14, fmt="%.3f")
cont2.setp_label(fontsize=10, fmt="%g \lambda")

mplim.plot()

plt.show()
