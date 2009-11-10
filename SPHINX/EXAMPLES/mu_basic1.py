from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame)
annim.Image()
annim.Graticule()
annim.plot()
annim.interact_imagecolors()
plt.show()
