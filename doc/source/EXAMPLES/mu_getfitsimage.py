from kapteyn import maputils

fitsobject = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
print(fitsobject.str_axisinfo())
fitsobject.set_imageaxes(promptfie=maputils.prompt_imageaxes)
fitsobject.set_limits(promptfie=maputils.prompt_box)
print(fitsobject.str_spectrans())
fitsobject.set_spectrans(promptfie=maputils.prompt_spectrans)
fitsobject.set_skyout(promptfie=maputils.prompt_skyout)

print("\nWCS INFO:")
print(fitsobject.str_wcsinfo())
