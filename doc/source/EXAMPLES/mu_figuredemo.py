from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage('example1test.fits')

fig = plt.figure(figsize=(5.5,5))
frame = fig.add_axes([0.1, 0.1, 0.8, 0.8])
annim = fitsobj.Annotatedimage(frame)
annim.set_aspectratio(1.2)
grat = annim.Graticule()

maputils.showall()
