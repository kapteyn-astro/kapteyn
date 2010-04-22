"""Show interaction options"""
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("m101.fits")
#f.set_limits((100,500),(200,400))

fig = plt.figure(figsize=(9, 7))
frame = fig.add_subplot(1, 1, 1)

mycmlist = ["mousse.lut", "ronekers.lut"]
maputils.cmlist.add(mycmlist)
print "Colormaps: ", maputils.cmlist.colormaps

annim = f.Annotatedimage(frame, cmap="m101.lut")
annim.cmap.set_bad('w')
ima = annim.Image()
annim.Pixellabels()
annim.Colorbar(label="Unknown unit")
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

plt.show()


