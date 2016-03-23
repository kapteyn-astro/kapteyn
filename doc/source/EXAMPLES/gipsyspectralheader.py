from kapteyn import wcs
from math import sqrt

header = { 'NAXIS'  : 1,
           'CTYPE1' : 'FREQ-OHEL',
           'CRVAL1' : 1.415418199417E+09,
           'CRPIX1' : 32,
           'CUNIT1' : 'HZ',
           'CDELT1' : -7.812500000000E+04,
           'VELR'   : 1.050000000000E+06,
           'RESTFRQ': 0.14204057520000E+10
         }

f = crval = header['CRVAL1']
df = cdelt = header['CDELT1']
crpix = header['CRPIX1']
velr = header['VELR']
f0 =  header['RESTFRQ']
c = wcs.c   # Speed of light

print("VELR is the reference velocity given in the velocity frame")
print("coded in CTYPE (e.g. HEL, LSR)")
print("The velocity is either an optical or a radio velocity. This")
print("is also coded in CTYPE (e.g. 'O', 'R')")

proj = wcs.Projection(header)
spec = proj.spectra(ctype='VOPT-F2W')
pixrange = list(range(crpix-3, crpix+3))
V = spec.toworld1d(pixrange) 
print("\n VOPT-F2W with spectral translation:")
for p, v in zip(pixrange, V):
  print("%4d %15f" % (p, v/1000))

print("\n VOPT calculated:")
fb = f0/(1.0+velr/c)
Vtopo = c * ((fb*fb-f*f)/(fb*fb+f*f))
dfb = df*(c-Vtopo)/sqrt(c*c-Vtopo*Vtopo)
for p in pixrange:
   f2 = fb + (p-crpix)*dfb
   Z = c * (f0/f2-1.0)
   print("%4d %15f" % (p, Z/1000.0))

print("\nOptical with native GIPSY formula, which is an approximation:")
fR = crval
dfR = cdelt
for p in pixrange:
   Zs = velr + c*f0*(1/(fR+(p-crpix)*dfR)-1/fR)
   print("%4d %15f" % (p, Zs/1000.0))