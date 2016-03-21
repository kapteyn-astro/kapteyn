from kapteyn import maputils
from re import split

Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
hdr = Basefits.hdr.copy()
print "Header"
print "-"*80
print Basefits.str_header()
hdr = maputils.fitsheader2dict(hdr)


crval1, crval2 = hdr['CRVAL1'], hdr['CRVAL2']
print "topixel -> crpix", Basefits.proj.topixel((crval1, crval2))

hdr['CTYPE1'] = 'RA---MER'
hdr['CTYPE2'] = 'DEC--MER'
#hdr['PV1_1'] = crval1
#hdr['PV1_2'] = crval2
hdr['CRVAL1'] = 0.0
hdr['CRVAL2'] = 0.0
#hdr['LONPOLE'] = crval1
#hdr['LATPOLE'] = +90
f = maputils.FITSimage(externalheader=hdr)
crpix1, crpix2 = f.proj.topixel((crval1, crval2))
print "CRPIX=", crpix1, crpix2

#hdr['CRPIX1'] = crpix1
#hdr['CRPIX2'] = crpix2

pxlim = (int(crpix1-600), int(crpix1+600))
pylim = (int(crpix2-600), int(crpix2+600))
print pxlim, pylim

while 1:
   keyval = raw_input("Enter key=val ..... [start reproj.]: ")
   if keyval != '':
      if 1: #try:
         key, val = split('[=]+', keyval.upper())
         hdr[key] = eval(val)
      else: #except:
         print "%s is wrong input" % keyval
   else:
      break

print "New header is:", hdr

Reprojfits = Basefits.reproject_to(hdr, pxlim=pxlim, pylim=pylim)
Reprojfits.writetofits("reproj.fits", clobber=True)
crpix1n, crpix2n = Reprojfits.hdr['CRPIX1'], Reprojfits.hdr['CRPIX2']

print "crval from new crpix", Reprojfits.proj.toworld((crpix1n, crpix2n))
print "world with old crpix", Reprojfits.proj.toworld((crpix1, crpix2))
print "crpix with old crval in new proj", Reprojfits.proj.topixel((crval1, crval2))

