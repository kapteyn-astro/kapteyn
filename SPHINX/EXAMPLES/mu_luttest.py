from kapteyn import maputils
from matplotlib import pyplot as plt
from kapteyn import mplutil
import glob

f = maputils.FITSimage("m101.fits")

fig = plt.figure(figsize=(9,7))
frame = fig.add_axes([0.1,0.2,0.85, 0.75])

extralist = mplutil.VariableColormap.luts()
print "Extra luts from Kapteyn Package", extralist
maputils.cmlist.add(extralist)

mycmlist = glob.glob("*.lut")
print "\nFound private color maps:", mycmlist
maputils.cmlist.add(mycmlist)

print "\nAll color maps now available: ", maputils.cmlist.colormaps

annim = f.Annotatedimage(frame) #,cmap="mousse.lut")
annim.set_colormap("mousse.lut")
annim.Image()
annim.Pixellabels()
annim.Colorbar()
annim.plot()

annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

tdict = dict(color='g', fontsize=10, va='bottom', ha='left')

units = 'unknown'
if f.hdr.has_key('BUNIT'):
   units = hdr['BUNIT']

template  = "File: [%s]  Data units:  [%s]\n" % (f.filename, units)
template += "Use pgUp and pgDown keys to browse through color maps\n"
template += "Color map scaling keys: 0=reset 1=linear 2=logarithmic "
template += "3=exponential 4=square-root 5=square 9=inverse toggle\n"
template += "Histogram equalization: h\n"
template += "Save current color map to disk: m\n"
template += "Change color of bad pixels: b\n"
template += "Change slope and offset: Right mouse button"

fig.text(0.01,0.01, template, tdict)

plt.show()
