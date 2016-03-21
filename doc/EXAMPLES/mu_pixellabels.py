from kapteyn import maputils
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(4,4))
frame = fig.add_subplot(1,1,1)

fitsobject = maputils.FITSimage("m101.fits")
annim = fitsobject.Annotatedimage(frame)
annim.Pixellabels(plotaxis="bottom", major=200, minor=10, color="r")
pl2 = annim.Pixellabels(plotaxis="right", color="b", markersize=10,
                        gridlines=True)
pl2.setp_marker(markersize=+15, color='b', markeredgewidth=2)
pl3 = annim.Pixellabels(plotaxis="top", color='g',
                        gridlines=False)
pl3.setp_marker(markersize=-10) 
pl3.setp_label(rotation=90)
pl4 = annim.Pixellabels(plotaxis="left", major=150, minor=25)
pl4.setp_label(fontsize=10)

annim.plot()
plt.show()

