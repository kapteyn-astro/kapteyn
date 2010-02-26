from kapteyn import wcs
from math import sqrt
from numpy import arange

header_gds = { 
           'NAXIS'  : 1,
           'NAXIS1' : 127,
           'CTYPE1' : 'FREQ-RHEL',
           'CRVAL1' : 1418921567.851000,
           'CRPIX1' : 63.993952051196288,
           'CUNIT1' : 'HZ',
           'CDELT1' : -9765.625,
           'VELR'   : 304000.0,
           'RESTFRQ': 1420405752.0,
         }

f0 = header_gds['RESTFRQ']
Vr = header_gds['VELR']
fr = header_gds['CRVAL1']
df = header_gds['CDELT1']
crpix = header_gds['CRPIX1']
c = wcs.c                                   # Speed of light
p = pixrange = arange(crpix-2, crpix+3)     # Range of pixels for which we 
                                            # want world coordinates
# Calculate the barycentric equivalents
fb = f0*(1.0-Vr/c)
Vtopo = c * ((fb*fb-fr*fr)/(fb*fb+fr*fr))
dfb = df*(c-Vtopo)/sqrt(c*c-Vtopo*Vtopo)
print "Topocentric correction (km/s):", Vtopo/1000
print "Barycentric frequency and increment (Hz):", fb, dfb

# VRAD from spectral translation, assumed to give the correct velocities
proj = wcs.Projection(header_gds)
spec = proj.spectra(ctype='VRAD')
V1 = spec.toworld1d(pixrange)

# Radio velocities with GIPSY formula with barycentric 
# values (excact).
V2 = Vr - c*(p-crpix)*dfb/f0

# Radio velocities with GIPSY formula without rest frequency and 
# with barycentric values (exact).
V3 = Vr + (p-crpix)*dfb*(Vr-c)/fb

# Radio velocities with GIPSY formula using topocentric,
# values (approximation).
V4 = Vr - c*(p-crpix)*df/f0

# Check the differences
# d = -c*(p-crpix)*(df-dfb)/f0
# print (V4-V1)/1000, d/1000

print "\n%10s %14s %14s %14s %14s" % ('pix', 'WCSLIB',
'GIP+bary', 'GIP+bary-f0', 'GIP+topo')
for pixel, v1,v2,v3,v4 in zip(pixrange, V1, V2, V3, V4):
   print "%10.4f %14f %14f %14f %14f" % (pixel, v1/1000, v2/1000, 
         v3/1000, v4/1000) 