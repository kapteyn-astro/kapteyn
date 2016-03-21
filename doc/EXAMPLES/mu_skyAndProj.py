from kapteyn import maputils, wcs
from matplotlib import pyplot as plt
import numpy as np

# The data you want to re-project
Basefits = maputils.FITSimage("m101.fits")
hdrIn = Basefits.hdr
projIn = Basefits.proj
crvalsI = Basefits.proj.crval 

# We want to change the sky system, so we need to know the values of CRVAL in the new system
trans = wcs.Transformation(projIn.skysys, skyout=wcs.galactic)
crvalsO = trans(crvalsI)

# Adjust the new header which was derived from the input
hdrOut = hdrIn.copy()
hdrOut['CTYPE1'] = 'GLON-SIN'
hdrOut['CTYPE2'] = 'GLAT-SIN'
hdrOut['CRVAL1'] = crvalsO[0]
hdrOut['CRVAL2'] = crvalsO[1]

# Get an estimate of the new corners by converting ALL boundary pixels of the input map
naxis1 = hdrIn['NAXIS1']
naxis2 = hdrIn['NAXIS2']
x = np.concatenate([np.arange(1, naxis1+1), naxis1*np.ones(naxis1),
                    np.arange(1, naxis1+1), np.ones(naxis1)])
y = np.concatenate([np.ones(naxis2), np.arange(1, naxis2+1),
                    naxis2*np.ones(naxis1), np.arange(1, naxis2+1)]) 
x, y = projIn.toworld((x,y))

# But we need Galactic coordinates  before converting to pixels 
# in the new system, not the original cooordinates:
x, y = trans((x,y)) 

# Create a dummy object to calculate pixel coordinates in the new system. 
f = maputils.FITSimage(externalheader=hdrOut)
px, py = f.proj.topixel((x,y))
pxlim = [int(min(px))-1, int(max(px))+1]
pylim = [int(min(py))-1, int(max(py))+1]
#print "New limits:", pxlim, pylim

# Do the re-projection
Reprojfits = Basefits.reproject_to(hdrOut, pxlim_dst=pxlim, pylim_dst=pylim)
#Reprojfits.writetofits("reproj.fits", clobber=True)
crpixs = Reprojfits.proj.crpix
crvals = Reprojfits.proj.toworld(crpixs)
#print "Check that crpix values in output are adjusted correctly:"
#print "Crvals from initial transformation and from re-projection:", crvals, crvalsO

# Some plotting to check the result
fig = plt.figure(figsize=(9,5))
frame1 = fig.add_subplot(1,2,1)
frame2 = fig.add_subplot(1,2,2)
im1 = Basefits.Annotatedimage(frame1)
im2 = Reprojfits.Annotatedimage(frame2)
im1.Image(); im1.Graticule()
im2.Image(); im2.Graticule()
im1.interact_toolbarinfo()
im2.interact_toolbarinfo()
im1.plot(); im2.plot()
fig.tight_layout()
plt.show()
