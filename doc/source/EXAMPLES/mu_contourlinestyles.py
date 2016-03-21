from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
annim.Image(alpha=0.5)
cont = annim.Contours(linestyles=('solid', 'dashed', 'dashdot', 'dotted'),
                      linewidths=(2,3,4), colors=('r','g','b','m'))
annim.plot()

print "Levels=", cont.clevels

plt.show()

