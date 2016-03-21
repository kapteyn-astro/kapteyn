from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy as np

Basefits = maputils.FITSimage("m101big.fits")
hdr = Basefits.hdr.copy()

hdr['CTYPE1'] = 'RA---MER'
hdr['CTYPE2'] = 'DEC--MER'
hdr['CRVAL1'] = 0.0
hdr['CRVAL2'] = 0.0
naxis1 = Basefits.hdr['NAXIS1']
naxis2 = Basefits.hdr['NAXIS2']

# Sample the border in pixels
x = np.concatenate([np.arange(1, naxis1+1), naxis1*np.ones(naxis1),
                    np.arange(1, naxis1+1), np.ones(naxis1)])
y = np.concatenate([np.ones(naxis2), np.arange(1, naxis2+1),
                    naxis2*np.ones(naxis1), np.arange(1, naxis2+1)]) 
x, y = Basefits.proj.toworld((x,y))


# Create a dummy object to calculate pixel coordinates of border in the new system. 
f = maputils.FITSimage(externalheader=hdr)
px, py = f.proj.topixel((x,y))
pxlim = [int(min(px))-1, int(max(px))+1]
pylim = [int(min(py))-1, int(max(py))+1]
# print "New limits:", pxlim, pylim

Reprojfits = Basefits.reproject_to(hdr, pxlim_dst=pxlim, pylim_dst=pylim)
#Reprojfits.writetofits("reproj.fits", clobber=True)

fig = plt.figure(figsize=(9,5))
frame1 = fig.add_subplot(1,2,1)
frame2 = fig.add_subplot(1,2,2)
im1 = Basefits.Annotatedimage(frame1)
im2 = Reprojfits.Annotatedimage(frame2)
im1.Image(); im1.Graticule()
im2.Image(); im2.Graticule()
im1.plot(); im2.plot()
fig.tight_layout()
plt.show()
