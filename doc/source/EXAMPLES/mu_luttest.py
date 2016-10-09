from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import mplutil
import glob

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(9,7))
frame = fig.add_axes([0.1,0.2,0.85, 0.75])

extralist = mplutil.VariableColormap.luts()
print("Extra luts from Kapteyn Package", extralist)
maputils.cmlist.add(extralist)

mycmlist = glob.glob("*.lut")
print("\nFound private color maps:", mycmlist)
maputils.cmlist.add(mycmlist)

print("\nAll color maps now available: ", maputils.cmlist.colormaps)

annim = f.Annotatedimage(frame) #,cmap="mousse.lut")
annim.set_colormap("mousse.lut")
annim.Image()
annim.Pixellabels()
annim.Colorbar()
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

units = 'unknown'
if 'BUNIT' in f.hdr:
   units = f.hdr['BUNIT']
helptext  = "File: [%s]  Data units:  [%s]\n" % (f.filename, units)
helptext += annim.get_colornavigation_info()
tdict = dict(color='g', fontsize=10, va='bottom', ha='left')
fig.text(0.01,0.01, helptext, tdict)

plt.show()
