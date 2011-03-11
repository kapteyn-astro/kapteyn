from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS' : 2, 'NAXIS1': 800, 'NAXIS2': 800,
          'CTYPE1':'RA---TAN',
          'CRVAL1': 0.0, 'CRPIX1' : 1, 'CUNIT1' : 'deg', 'CDELT1' : -0.05,
          'CTYPE2':'DEC--TAN',
          'CRVAL2': 0.0, 'CRPIX2' : 1, 'CUNIT2' : 'deg', 'CDELT2' : 0.05,
         }

fitsobject = maputils.FITSimage(externalheader=header)

fig = plt.figure()
frame = fig.add_axes([0.1,0.1, 0.82,0.82])
annim = fitsobject.Annotatedimage(frame)
grat = annim.Graticule(header)
x1 = 10; y1 = 1
x2 = 10; y2 = annim.pylim[1]

ruler1 = annim.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=0.0, step=1.0)
ruler1.setp_label(color='g')
x1 = x2 = annim.pxlim[1]
ruler2 = annim.Ruler(x1=x1, y1=y1, x2=x2, y2=y2, lambda0=0.5, step=2.0, 
                     fmt='%3d', mscale=-1.5, fliplabelside=True)
ruler2.setp_label(ha='left', va='center', color='b', clip_on=False)

ruler3 = annim.Ruler(x1=23*15, y1=30, x2=22*15, y2=15, lambda0=0.0, 
                     step=2, world=True, 
                     units='deg', addangle=90)
ruler3.setp_label(color='r')

ruler4 = annim.Ruler(pos1="23h0m 15d0m", pos2="22h0m 30d0m", lambda0=0.0, 
                     step=1,
                     fmt="%4.0f^\prime", 
                     fun=lambda x: x*60.0, addangle=0)
ruler4.setp_line(color='g')
ruler4.setp_label(color='m')

ruler5 = annim.Ruler(x1=1, y1=800, x2=800, y2=800, lambda0=0.5, step=2,
                     fmt="%4.1f", addangle=90)
ruler5.setp_label(color='c')

ruler6 = annim.Ruler(pos1="23h0m 15d0m", rulersize=15, step=2,
                     units='deg', lambda0=0, fliplabelside=True)
ruler6.setp_label(color='b')
ruler6.set_title("Size in deg", fontsize=10)

ruler7 = annim.Ruler(pos1="23h0m 30d0m", rulersize=5, rulerangle=90, step=1,
                     units='deg', lambda0=0)
ruler7.setp_label(color='#ffbb33')

ruler8 = annim.Ruler(pos1="23h0m 15d0m", rulersize=5, rulerangle=10, step=1.25,
                     units='deg', lambda0=0, fmt="%02g", fun=lambda x: x*8)
ruler8.setp_label(color='#3322ff')
ruler8.set_title("Size in kpc", fontsize=10)

# Increase size and lambda a bit to get all the labels 
# from 5 to 0 and to 5 again plotted
# Show LaTeX in ruler label
ruler9 = annim.Ruler(pos1="23h0m 10d0m", rulersize=10.1, rulerangle=90, step=1,
                     units='deg', lambda0=0.51)
ruler9.setp_label(color='#33ff22')
ruler9.set_title("$\lambda = 0.5$", fontsize=10)

annim.plot()
annim.interact_toolbarinfo()
annim.interact_writepos()
plt.show()
