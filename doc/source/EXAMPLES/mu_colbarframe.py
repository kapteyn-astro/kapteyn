from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fig = plt.figure(figsize=(5,7.5))
#fig = plt.figure()
frame = fig.add_axes((0.1, 0.2, 0.8, 0.8))
cbframe = fig.add_axes((0.1, 0.1, 0.8, 0.1))

annim = fitsobj.Annotatedimage(cmap="Accent", clipmin=8000, frame=frame)
annim.Image()
units = r'$ergs/(sec.cm^2)$'
colbar = annim.Colorbar(fontsize=8, orientation='horizontal', frame=cbframe)
colbar.set_label(label=units, fontsize=24)
annim.plot()
annim.interact_imagecolors()
plt.show()
