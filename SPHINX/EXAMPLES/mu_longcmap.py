from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import mgrid,exp

f = maputils.FITSimage("m101.fits")
n1, n2 = f.proj.naxis 
X,Y = mgrid[0:n1, 0:n2]
Z = exp( -(X**2)/1.0e5-(Y**2)/1.0e5 )
f2 = maputils.FITSimage(externalheader=f.hdr, externaldata=Z)

fig = plt.figure()
frame = fig.add_subplot(1,2,1)
annim = f2.Annotatedimage(frame)
annim.set_colormap("rainbow.lut")
annim.cmap.set_length(64)
annim.Image()
annim.Pixellabels()
annim.Colorbar(fontsize=7, orientation='horizontal')
annim.plot()
annim.interact_toolbarinfo()
annim.interact_imagecolors()
annim.interact_writepos()

frame2 = fig.add_subplot(1,2,2)
annim2 = f2.Annotatedimage(frame2)
annim2.set_colormap("rainbow.lut")
annim2.cmap.set_length(1021)
annim2.Image()
annim2.Pixellabels()
annim2.Colorbar(fontsize=7, orientation='horizontal')
annim2.plot()
annim2.interact_toolbarinfo()
annim2.interact_imagecolors()
annim2.interact_writepos()

plt.show()
