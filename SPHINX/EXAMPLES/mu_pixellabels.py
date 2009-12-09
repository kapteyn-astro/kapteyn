from kapteyn import maputils
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(4,4))
frame = fig.add_subplot(1,1,1)

fitsobject = maputils.FITSimage("m101.fits")
annim = fitsobject.Annotatedimage(frame)
annim.Pixellabels(plotaxis="bottom", color="r")
annim.Pixellabels(plotaxis="right", color="b", markersize=10)
annim.Pixellabels(plotaxis="top", color="g", markersize=-10, va='top')

annim.plot()
plt.show()

