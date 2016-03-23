from kapteyn import wcs
from math import sqrt
from numpy import arange

header_gds = { 
           'NAXIS'  : 1,
           'NAXIS1' : 127,
           'CTYPE1' : 'FREQ-OHEL',
           'CRVAL1' : 1418921567.851000,
           'CRPIX1' : 63.993952051196288,
           'CUNIT1' : 'HZ',
           'CDELT1' : -9765.625,
           'VELR'   : 304000.0,
           'RESTFRQ': 1420405752.0,
         }

f0 = header_gds['RESTFRQ']
Zr = header_gds['VELR']
fr = header_gds['CRVAL1']
df = header_gds['CDELT1']
crpix = header_gds['CRPIX1']
c = wcs.c                                   # Speed of light
p = pixrange = arange(crpix-2, crpix+3)     # Range of pixels for which we 
                                            # want world coordinates
# Calculate the barycentric equivalents
fb = f0/(1.0+Zr/c)
Vtopo = c * ((fb*fb-fr*fr)/(fb*fb+fr*fr))
dfb = df*(c-Vtopo)/sqrt(c*c-Vtopo*Vtopo)
print("Topocentric correction (km/s):", Vtopo/1000)
print("Barycentric frequency and increment (Hz):", fb, dfb)

# VOPT-F2W from spectral translation, assumed to give the correct velocities
proj = wcs.Projection(header_gds)
spec = proj.spectra(ctype='VOPT-F2W')
Z1 = spec.toworld1d(pixrange)

# Non linear: Optical with GIPSY formula with barycentric 
# values (excact).
Z2 = Zr + c*f0*(1/(fb+(p-crpix)*dfb)-1/fb)

# Non Linear: Optical with GIPSY formula without rest frequency and 
# with barycentric values (exact).
Z3 = (Zr*fb - (p-crpix)*c*dfb) / (fb+(p-crpix)*dfb)

# Non Linear: Optical with GIPSY formula using topocentric,
# values (approximation).
Z4 = Zr + c*f0*(1/(fr+(p-crpix)*df)-1/fr)

# Linear: Optical with GIPSY formula with barycentric values 
# and dZ approximation for linear transformation
# Rest frequency is part of formula.
dZ = -c*f0*dfb/fb/fb
Z5 = Zr + (p-crpix) * dZ

# Linear: Optical with GIPSY formula with barycentric values 
# and dZ approximation for linear transformation
# Rest frequency is not used.
dZ = -c *dfb/fb
Z6 = Zr + (p-crpix) * dZ

print("\n%10s %14s %14s %14s %14s %14s %14s" % ('pix', 'WCSLIB',
'GIP+bary', 'GIP+bary-f0', 'GIP+topo', 'Linear+f0', 'Linear-f0'))
for pixel, z1,z2,z3,z4,z5, z6 in zip(pixrange, Z1, Z2, Z3, Z4, Z5, Z6):
   print("%10.4f %14f %14f %14f %14f %14f %14f" % (pixel, z1/1000, z2/1000, 
         z3/1000, z4/1000, z5/1000, z6/1000)) 