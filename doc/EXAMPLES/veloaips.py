#!/usr/bin/env python
from kapteyn import wcs
import math

# With this application we address the question what WCSLIB does if it
# encounters an AIPS VELO-XXX axis as in CTYPE1='VELO-HEL'.

V0 = -.24300000000000E+06             # Radio vel in m/s
dV = 5000.0                           # Delta in m/s
f0 = 0.14204057583700e+10             # Rest frequency
c  = 299792458.0                      # Speed of light in m/s
crpix = 32                            # The reference pixel
pixels = range(crpix-2,crpix+2)       # A pixel range for our calculations

header = { 'NAXIS'  : 1,
           'CTYPE1' : 'VELO-HEL',
           'CRVAL1' : V0,
           'CRPIX1' : crpix,
           'CUNIT1' : 'm/s',
           'CDELT1' : dV,
           'RESTFRQ': f0
         }

print "The velocity increment is constant and equal to %f (km/s): "\
      % (dV/1000.0)
proj = wcs.Projection(header)
V = proj.toworld1d(pixels)
print "\nWith CTYPE='VELO-HEL' we get the output: "
print "Pixel , Velocity (km/s)"
for p,v in zip(pixels, V):
  print "%4d %15f" % (p, v/1000)

proj2 = proj.spectra('VOPT-V2W')
V = proj.toworld1d(pixels)
Z = proj2.toworld1d(pixels)
print "\nWith CTYPE='VELO-HEL' and spec.trans 'VOPT-V2W':"
print "Pixel,  Velocity (km/s), Voptical (km/s)"
for p,v,z in zip(pixels, V, Z):
  print "%4d %15f %15f" % (p, v/1000, z/1000)

header['CTYPE1'] = 'VELO'     # Change explicitly to relativistic velocity
proj = wcs.Projection(header)
V = proj.toworld1d(pixels)
print "\nWith CTYPE='VELO': "
print "Pixel,  Vrelativistic (km/s)"
for p,v in zip(pixels, V):
  print "%4d %15f" % (p, v/1000)

proj2 = proj.spectra('VOPT-V2W')
V = proj.toworld1d(pixels)
Z = proj2.toworld1d(pixels)
print "\nWith CTYPE='VELO' and spectral translation 'VOPT-V2W':"
print "Pixel, Vrelativistic (km/s), Voptical (km/s)"
for p,v,z in zip(pixels, V, Z):
  print "%4d %15f %15f" % (p, v/1000, z/1000)

print ""
print """Optical velocities, calculated with the appropriate formulas, 
from relativistic velocity with constant velocity increment. This should give
the same output as the previous conversion."""
v0 = V0
for i in pixels:
  v1 = v0 + (i-crpix)*dV
  beta = v1/c
  frac = (1-beta)/(1+beta)
  f = f0 * math.sqrt(frac)
  Z = c* (f0-f)/f
  print "%4d %15f %15f" % (i, v1/1000.0, Z/1000.0)

header['CTYPE1'] = 'VRAD'
proj = wcs.Projection(header)
proj2 = proj.spectra('VOPT-F2W')
Z0 = proj.toworld1d(pixels)
Z1 = proj2.toworld1d(pixels)
print ""
print """Now replace VELO-HEL in CTYPE1 by VRAD. Calculate VOPT in two ways.
First with spectral VOPT-F2W and then with the appropriate formulas
for VRAD -> VOPT-F2W."""
print "With CTYPE='VRAD' and spec.trans 'VOPT-F2W'(Z1) and calculated (Z2):"
print "%4s %15s %15s %15s" % ("Pixel", "Vrad(km/s)", "Z1 (km/s)",  "Z2 (km/s)")
for p,z0,z1 in zip(pixels, Z0, Z1):
  V = V0 + (p-crpix)*dV
  nu_r = f0* (1-V/c)
  Z2 = c*((f0-nu_r)/nu_r)
  print "%4d %15f %15f %15f" % (p, z0/1000, z1/1000, Z2/1000)

print """Obviously the optical velocities are different compared to 
those calculated from CTYPE1='VELO' or 'VELO-HEL', This also proves
that a VELO-XXX axis form a AIPS source is not processed as a radio
velocity."""
