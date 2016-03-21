from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

fitsobject = maputils.FITSimage(externalheader=header)

fig = plt.figure(figsize=(7,7))
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
annim = fitsobject.Annotatedimage(frame)
grat = annim.Graticule(header)
x1 = 10; y1 = 1
x2 = 10; y2 = grat.pylim[1]

ruler1 = annim.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=0.0, step=1.0)
ruler1.setp_labels(color='g')
x1 = x2 = grat.pxlim[1]
ruler2 = annim.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=0.5, step=2.0, 
                     fmt='%3d', mscale=-1.5, fliplabelside=True)
ruler2.setp_labels(ha='left', va='center', color='b', clip_on=False)

ruler3 = annim.Ruler(x1=23*15, y1=30, x2=22*15, y2=15, lambda0=0.0, 
                     step=1, world=True, 
                     fmt=r"$%4.0f^\prime$", 
                     fun=lambda x: x*60.0, addangle=0)
ruler3.setp_labels(color='r')

ruler4 = annim.Ruler(pos1="23h0m 15d0m", pos2="22h0m 30d0m", lambda0=0.0, 
                     step=1,
                     fmt=r"$%4.0f^\prime$", 
                     fun=lambda x: x*60.0, addangle=0)
ruler4.setp_line(color='g')
ruler4.setp_labels(color='m')

ruler5 = annim.Ruler(x1=1, y1=800, x2=800, y2=800, lambda0=0.5, step=2,
                     fmt="%4.1f", addangle=90)
ruler5.setp_labels(color='c')
annim.plot()

plt.show()
