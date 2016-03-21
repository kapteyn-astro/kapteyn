from kapteyn import maputils
from matplotlib import pylab as plt

fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)

print("\nAXES INFO for image axes:\n")
print fitsobject.str_axisinfo(axnum=fitsobject.axperm)

print("\nWCS INFO:\n")
print fitsobject.str_wcsinfo()
