from kapteyn import maputils
from matplotlib import pyplot as plt

Basefits = maputils.FITSimage("m101big.fits")
hdr = Basefits.hdr.copy()

hdr['CTYPE1'] = 'RA---MER'
hdr['CTYPE2'] = 'DEC--MER'
hdr['CRVAL1'] = 0.0
hdr['CRVAL2'] = 0.0
naxis1 = Basefits.hdr['NAXIS1']
naxis2 = Basefits.hdr['NAXIS2']

# Get an estimate of the new corners
x = [0]*5; y = [0]*5
x[0], y[0] = Basefits.proj.toworld((1,1))
x[1], y[1] = Basefits.proj.toworld((naxis1,1))
x[2], y[2] = Basefits.proj.toworld((naxis1,naxis2))
x[3], y[3] = Basefits.proj.toworld((1,naxis2))
x[4], y[4] = Basefits.proj.toworld((naxis1/2.0,naxis2))

# Create a dummy object to calculate pixel coordinates
# in the new system. Then we can find the area in pixels
# that corresponds to the area in the sky.
f = maputils.FITSimage(externalheader=hdr)
px, py = f.proj.topixel((x,y))
pxlim = [int(min(px))-10, int(max(px))+10]
pylim = [int(min(py))-10, int(max(py))+10]

Reprojfits = Basefits.reproject_to(hdr, pxlim=pxlim, pylim=pylim)
#Reprojfits.writetofits("reproj.fits", clobber=True)

fig = plt.figure(figsize=(9,5))
frame1 = fig.add_axes([0.07,0.1,0.35, 0.8])
frame2 = fig.add_axes([0.5,0.1,0.43, 0.8])
im1 = Basefits.Annotatedimage(frame1)
im2 = Reprojfits.Annotatedimage(frame2)
im1.Image(); im1.Graticule()
im2.Image(); im2.Graticule()
im1.plot(); im2.plot()
plt.show()