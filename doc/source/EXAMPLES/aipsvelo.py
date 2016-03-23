#!/usr/bin/env python
from kapteyn import wcs
from math import sqrt

V0 = -.24300000000000E+06             # Radio vel in m/s
dV = 5000.0                           # Delta in m/s
f0 = 0.14204057583700e+10
c = wcs.c                             # Speed of light 299792458.0 m/s
crpix = 32
pixels = list(range(30,35))

header = { 'NAXIS'  : 1,
           'CTYPE1' : 'VELO-HEL',
           'VELREF' : 258,
           'CRVAL1' : V0,
           'CRPIX1' : crpix,
           'CUNIT1' : 'm/s',
           'CDELT1' : dV,
           'RESTFRQ': f0
        }

print("The velocity increment is constant and equal to %f (km/s): "\
      % (dV/1000.0))

proj = wcs.Projection(header)
print("Allowed spectral translations", proj.altspec)
p2 = proj.spectra('VOPT-???')

print("\nT1. Radio velocity directly from header and optical velocity")
print("from spectral translation. VELO is a radio velocity here because")
print("VELREF > 256")

V = proj.toworld1d(pixels)
Z = p2.toworld1d(pixels)
print("Pixel Vradio in (km/s) and Voptical (km/s)")
for p,v,z in zip(pixels, V, Z):
  print("%4d %15f %15f" % (p, v/1000, z/1000))

print("\nT2. Now insert CTYPE1='VRAD' in the header and convert to VOPT-F2W")
print("with a spectral translation (Z1) and with a calculation (Z2)")
print("This should give the same results as in table T1.")
header['CTYPE1'] = 'VRAD'
proj = wcs.Projection(header)
p2 = proj.spectra('VOPT-F2W')
Z0 = proj.toworld1d(pixels)
Z1 = p2.toworld1d(pixels)
print("\nWith CTYPE='RAD' and spec.trans 'VOPT-F2W': Pixel , Vrad, Z1 (km/s), Z2 (km/s)")
for p,z0,z1 in zip(pixels, Z0, Z1):
  V = V0 + (p-crpix)*dV
  nu_r = f0* (1-V/c)
  Z2 = c*((f0-nu_r)/nu_r)
  print(p, z0/1000, z1/1000, Z2/1000)

print("\nT3. We set CTYPE1 to VELO-HEL and VELREF to 2 (Helio) and ")
print("derive optical and radio velocities from it. Compare these with")
print("the relativistic velocity in Table T4.")
header['CTYPE1'] = 'VELO-HEL'
header['VELREF'] = 2
proj = wcs.Projection(header)
print("Allowed spectral translations for VELO as optical velocity", proj.altspec)
p2 = proj.spectra('VRAD-???')
V = proj.toworld1d(pixels)
Z = p2.toworld1d(pixels)
print("Pixel Voptical in (km/s) and Vradio (km/s)")
for p,v,z in zip(pixels, V, Z):
  print("%4d %15f %15f" % (p, v/1000, z/1000))

print("\nT4. Next a list with optical velocities calculated from relativistic")
print("velocity with constant increment.")
print("If these values are different from the previous optical velocity then ")
print("obviously the velocities derived from the header are not relativistic")
print("as in pre 4.5.1 versions of WCSLIB.")
v0 = V0
for i in pixels:
  v1 = v0 + (i-crpix)*dV
  beta = v1/c
  frac = (1-beta)/(1+beta)
  f = f0 * sqrt(frac)
  Z = c* (f0-f)/f
  print("%4d %15f" % (i ,Z/1000.0))
