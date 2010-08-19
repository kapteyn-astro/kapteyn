from kapteyn import maputils

Basefits = maputils.FITSimage(promptfie=maputils.prompt_fitsfile)
Rotfits = Basefits.reproject_to(rotation=40.0, naxis1=800, naxis2=800, 
                                crpix1=400, crpix2=400)

# If you want alignment with direction of the north, use:
# Rotfits = Basefits.reproject_to(rotation=0.0, crota2=0.0) 

# If copy on disk required: 
# Rotfits.writetofits("aligned.fits", clobber=True, append=False)

annim = Rotfits.Annotatedimage()
annim.Image()
annim.Graticule()
annim.interact_toolbarinfo()
maputils.showall()