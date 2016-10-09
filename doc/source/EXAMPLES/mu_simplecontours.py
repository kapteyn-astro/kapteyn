from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
cont = annim.Contours()
annim.plot()

print("Levels=", cont.clevels)

plt.show()

