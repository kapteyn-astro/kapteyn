from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import fft, log, abs, angle

f = maputils.FITSimage("m101.fits")

yshift = -0.1
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.01, bottom=0.1, right=1.0, top=0.98, 
                    wspace=0.03, hspace=0.16)
frame = fig.add_subplot(2,3,1)
frame.text(0.5, yshift, "M101", ha='center', va='center',
           transform = frame.transAxes)
mplim = f.Annotatedimage(frame)
mplim.Image(cmap="spectral")
mplim.plot()


fftA = fft.rfft2(f.dat, f.dat.shape)
frame = fig.add_subplot(2,3,2)
frame.text(0.5, yshift, "Amplitude of FFT", ha='center', va='center',
           transform = frame.transAxes)
f = maputils.FITSimage("m101.fits", externaldata=log(abs(fftA)+1.0))
mplim2 = f.Annotatedimage(frame)
im = mplim2.Image(cmap="gray")
mplim2.plot()


frame = fig.add_subplot(2,3,3)
frame.text(0.5, yshift, "Phase of FFT", ha='center', va='center',
           transform = frame.transAxes)
f = maputils.FITSimage("m101.fits", externaldata=angle(fftA))
mplim3 = f.Annotatedimage(frame)
im = mplim3.Image(cmap="gray")
mplim3.plot()


frame = fig.add_subplot(2,3,4)
frame.text(0.5, yshift, "Inverse FFT", ha='center', va='center',
           transform = frame.transAxes)
D = fft.irfft2(fftA)
f = maputils.FITSimage("m101.fits", externaldata=D.real)
mplim4 = f.Annotatedimage(frame)
im = mplim4.Image(cmap="spectral")
mplim4.plot()

frame = fig.add_subplot(2,3,5)
Diff = D.real - mplim.data
f = maputils.FITSimage("m101.fits", externaldata=Diff)
mplim5 = f.Annotatedimage(frame)
im = mplim5.Image(cmap="spectral")
mplim5.plot()

frame.text(0.5, yshift, "M101 - inv. FFT", ha='center', va='center',
           transform = frame.transAxes)
s = "Residual with min=%.1g max=%.1g" % (Diff.min(), Diff.max())
frame.text(0.5, yshift-0.08, s, ha='center', va='center',
           transform = frame.transAxes, fontsize=8)

mplim.interact_imagecolors()
mplim2.interact_imagecolors()
mplim3.interact_imagecolors()
mplim4.interact_imagecolors()
mplim5.interact_imagecolors()

plt.show()
