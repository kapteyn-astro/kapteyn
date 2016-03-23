from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage('ngc6946.fits')

print("HEADER:\n")
print(fitsobject.str_header())

print("\nAXES INFO:\n")
print(fitsobject.str_axisinfo())

print("\nEXTENDED AXES INFO:\n")
print(fitsobject.str_axisinfo(int=True))

print("\nAXES INFO for image axes only:\n")
print(fitsobject.str_axisinfo(axnum=fitsobject.axperm))

print("\nAXES INFO for non existing axis:\n")
print(fitsobject.str_axisinfo(axnum=4))

print("SPECTRAL INFO:\n")
fitsobject.set_imageaxes(axnr1=1, axnr2=3)
print(fitsobject.str_spectrans())
