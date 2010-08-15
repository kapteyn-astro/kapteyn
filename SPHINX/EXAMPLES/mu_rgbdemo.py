from kapteyn import maputils
from numpy import sqrt
from matplotlib import pyplot as plt

f_red = maputils.FITSimage('m101_red.fits')
f_green = maputils.FITSimage('m101_green.fits')
f_blue = maputils.FITSimage('m101_blue.fits')

fig = plt.figure()
frame = fig.add_subplot(1,1,1)
annim = f_red.Annotatedimage(frame)
annim.RGBimage(f_red, f_green, f_blue, fun=lambda x:sqrt(x), alpha=1)

annim.interact_toolbarinfo(wcsfmt=None, zfmt="%g")
annim.interact_writepos(pixfmt=None, wcsfmt="%.12f", zfmt="%.3e", 
                        hmsdms=False)
maputils.showall()