""" Show contour lines with different lines styles """
from kapteyn import maputils
from matplotlib import pyplot as plt

f = maputils.FITSimage("RAxDEC.fits")

fig = plt.figure(figsize=(9,6))
frame = fig.add_subplot(1,1,1)

mplim = f.Annotatedimage(frame)
cont = mplim.Contours(negative="dotted")
mplim.plot()
mplim.interact_toolbarinfo()

plt.show()
