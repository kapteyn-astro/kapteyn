#!/usr/bin/env python
from kapteyn import wcs
from numpy import arange

header = { 'NAXIS'  : 1,
           'CTYPE1' : 'FELO-HEL',
           'CRVAL1' : 9120,
           'CRPIX1' : 32,
           'CUNIT1' : 'km/s',
           'CDELT1' : -21.882651442,
           'RESTFRQ': 1.420405752e+9
         }
crpix = header['CRPIX1']
pixrange = arange(crpix-2, crpix+3)
proj = wcs.Projection(header)
Z = proj.toworld1d(pixrange)
print "Pixel, velocity (km/s) with native header with FELO-HEL"
for p,v in zip(pixrange, Z):
   print p, v/1000.0

# Calculate the barycentric reference frequency and the frequency increment
f0 = header['RESTFRQ']
Zr = header['CRVAL1'] * 1000.0 # m/s
dZ = header['CDELT1'] * 1000.0 # m/s
c = wcs.c
fr = f0 / (1 + Zr/c)
print "\nCalculated a reference frequency: ", fr
df = -f0* dZ *c / ((c+Zr)*(c+Zr))
print "Calculated a frequency increment: ", df
Z = Zr + c*f0*(1/(fr+(pixrange-crpix)*df)-1/fr)
print "Pixel, velocity (km/s) with barycentric reference frequency and increment:"
for p,z in zip(pixrange, Z):
   print p, z/1000.0

# FELO-HEL is equivalent to VOPT-F2W
header['CTYPE1'] = 'VOPT-F2W'
proj = wcs.Projection(header)
Z = proj.toworld1d(pixrange)
print "\nPixel, velocity (km/s) with spectral translation VOPT-F2W"
for p,v in zip(pixrange, Z):
   print p, v/1000.0

# Now as a linear axis. Note that thoe output of toworld is in km/s
# and not in standard units (m/s) as for the recognized axis types
header['CTYPE1'] = 'FELO'
proj = wcs.Projection(header)
Z = proj.toworld1d(pixrange)
print "\nPixel, velocity (km/s) with CUNIT='FELO', which is unrecognized "
print "and therefore linear. This deviates from the previous output."
print "The second velocity is calculated manually."
for p,v in zip(pixrange, Z):
   print p, v, (Zr+(p-crpix)*dZ)/1000.0
