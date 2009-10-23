from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

f = maputils.FITSimage(externalheader=header)

fig = plt.figure(figsize=(7,7))
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
mplim = f.Annotatedimage(frame)
grat = mplim.Graticule(header)
x1 = 10; y1 = 1
x2 = 10; y2 = grat.pylim[1]

ruler1 = grat.Ruler(x1, y1, x2, y2, 0.0, 1.0)
ruler1.setp_labels(color='g')
x1 = x2 = grat.pxlim[1]
ruler2 = grat.Ruler(x1, y1, x2, y2, 0.5, 2.0, 
                   fmt='%3d', mscale=-1.5, fliplabelside=True)
ruler2.setp_labels(ha='left', va='center', color='b')
ruler3 = grat.Ruler(23*15,30,22*15,15, 0.5, 1, world=True, 
                    fmt=r"$%4.0f^\prime$", 
                    fun=lambda x: x*60.0, addangle=0)
ruler3.setp_labels(color='r')
ruler4 = grat.Ruler(1,800,800,800, 0.5, 2, fmt="%4.1f", addangle=90)
ruler4.setp_labels(color='c')
mplim.plot()

plt.show()
