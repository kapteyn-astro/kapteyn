from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import fft, log, asarray, float64, abs, angle

#f = maputils.FITSimage("m101.fits")
f = maputils.FITSimage("cl2.fits")

fig = plt.figure(figsize=(8,8))
frame = fig.add_subplot(2,2,1)

mplim = f.Annotatedimage(frame)
mplim.Image(cmap="gray")
mplim.plot()

fftA = fft.rfft2(f.dat, f.dat.shape)
fftre = fftA.real
fftim = fftA.imag

frame = fig.add_subplot(2,2,2)
#f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftre)+1.0))
f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftA)+1.0))
mplim2 = f.Annotatedimage(frame)
im = mplim2.Image(cmap="gray")
mplim2.plot()

frame = fig.add_subplot(2,2,3)
f = maputils.FITSimage("m101.fits", externaldata=angle(fftA))
mplim3 = f.Annotatedimage(frame)
im = mplim3.Image(cmap="gray")
mplim3.plot()

frame = fig.add_subplot(2,2,4)
D = fft.irfft2(fftA)
f = maputils.FITSimage("m101.fits", externaldata=D.real)
mplim4 = f.Annotatedimage(frame)
im = mplim4.Image(cmap="gray")
mplim4.plot()

mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()
mplim4.interact_imagecolors()

plt.show()
