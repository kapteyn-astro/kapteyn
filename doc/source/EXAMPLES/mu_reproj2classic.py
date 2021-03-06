from kapteyn import maputils, wcs
from math import *      # To support expression evaluation with 'eval()'



Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)

classicheader, skew, hdrchanged = Basefits.header2classic()
if  hdrchanged:
   print("Original header:\n", Basefits.str_header())
   if skew != 0.0:
      print("Found two different rotation angles. Difference is %f deg." % skew)
else:
   print("Header probably already 'classic'. Nothing changed")

print("""You can copy the data and replace the header by the classic header
or you can re-project it to get rid of skew or to rotate the data
using a rotation angle (keyword CROTAn=).""")

ok = input("Do you want to remove skew or rotate image ... [Y]/N: ")
if ok in ['y', 'Y', '']:
   lat = Basefits.proj.lataxnum
   key = "CROTA%d"%lat
   crotaold = classicheader[key] # CROTA Is always available in this header
   mes = "Enter value for CROTA%d ... [%g]: " %(lat, crotaold)
   newcrota = input(mes)
   if newcrota != '':
      crota = eval(newcrota)
      classicheader[key] = crota
   print("Classic header voor reproject:")
   print(classicheader)
   print("\n Re-projecting ...")
   fnew = Basefits.reproject_to(classicheader, insertspatial=False)
else:
   # A user wants to replace the header only. Leave data untouched.
   fnew = maputils.FITSimage(externalheader=classicheader,
                             externaldata=Basefits.dat)

filename_out = "classic.fits"
# ADD (!) to 'classic.fits'
print("A copy of the selected hdu in the FITS file is APPENDED to [%s] on disk"%filename_out)
fnew.writetofits(filename_out, clobber=True, append=True)
