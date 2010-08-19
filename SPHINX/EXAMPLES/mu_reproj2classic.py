from kapteyn import maputils, wcs
from math import *      # To support expression evaluation with 'eval()'

Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
classicheader, skew, hdrchanged = Basefits.header2classic()
if hdrchanged:
   print Basefits.str_header()
   if skew != 0.0:
      print "Found two different rotation angles. Difference is %f deg." % skew
else:
   print "Header probably already 'classic'. Nothing changed"

ok = raw_input("Do you want to reproject data to classic header? ... [Y]/n: ")
if ok in ['y', 'Y', '']:
   lat = Basefits.proj.lataxnum
   key = "CROTA%d"%lat
   crotaold = classicheader[key] # CROTA Is always available in this header
   mes = "Enter value for CROTA%d ... [%g]: " %(lat, crotaold)
   newcrota = raw_input(mes)
   if newcrota != '':
      crota = eval(newcrota)
      classicheader[key] = crota
   fnew = Basefits.reproject_to(classicheader)
else:
   fnew = maputils.FITSimage(externalheader=classicheader,
                             externaldata=Basefits.dat)
# ADD (!) to 'classic.fits'
fnew.writetofits("classic.fits", clobber=True, append=True)