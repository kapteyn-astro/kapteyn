from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import tabarray
import numpy

f = maputils.FITSimage("m101cdelt.fits")
fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f.Annotatedimage(frame, cmap="Set1")
grat = annim.Graticule()
annim.plot()

fn = 'smallworld.txt'
xp, yp = annim.positionsfromfile(fn, 's', cols=[0,1])
frame.plot(xp, yp, ',', color='b')

annim.interact_imagecolors()
plt.show()
