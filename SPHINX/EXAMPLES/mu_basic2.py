from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
f.set_limits((200,400), (200,400))
fig = plt.figure()
frame = fig.add_subplot(2,1,1)
annim = f.Annotatedimage(frame)
annim.Image(interpolation="nearest")
annim.Graticule()
annim.plot()
frame = fig.add_subplot(2,1,2)
annim = f.Annotatedimage(frame)
annim.Image(interpolation="spline36")
annim.Graticule()
annim.plot()
plt.show()
