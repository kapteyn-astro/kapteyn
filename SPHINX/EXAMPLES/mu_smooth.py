from kapteyn import maputils
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def func(x):
   nx = ny = x
   annim.set_blur(True, nx, ny, new=True)

f = maputils.FITSimage("m101.fits")
fig = plt.figure(figsize=(9,7))
frame = fig.add_subplot(1,1,1)
fr2 = fig.add_axes([0.3,0.01,0.4,0.03])
valinit = 1.0
sl = Slider(fr2, "Sigma: ", 0.1, 5.0, valinit=valinit)
sl.on_changed(func)

annim = f.Annotatedimage(frame)
#annim.set_colormap("mousse.lut")
annim.Image()
annim.plot()
func(valinit)
annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

plt.show()
