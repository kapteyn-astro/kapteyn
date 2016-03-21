from kapteyn import maputils
from matplotlib import pylab as plt


def plotgrat(n, ax1, ax2, offsetx=None, offsety=None, unitsx=None):
   f.set_imageaxes(ax1,ax2)
   frame = fig.add_subplot(4,2,n)
   annim = f.Annotatedimage(frame)
   grat = annim.Graticule(offsetx=offsetx, offsety=offsety, unitsx=unitsx)
   grat.setp_axislabel((0,1,2), fontsize=10)
   grat.setp_ticklabel(fontsize=7)

   xmax = annim.pxlim[1]+0.5; ymax = annim.pylim[1]+0.5
   ruler = annim.Ruler(x1=xmax, y1=0.5, x2=xmax, y2=ymax, 
                       lambda0=0.5, step=10.0,
                       units='arcmin',
                       fliplabelside=True)

   ruler.setp_line(lw=2, color='r')
   ruler.setp_label(clip_on=True, color='r', fontsize=9)

   ruler2 = annim.Ruler(x1=0.5, y1=0.5, x2=xmax, y2=ymax, 
                        lambda0=0.5, step=10.0/60.0, 
                        fun=lambda x: x*60.0, fmt="%4.0f^\prime", 
                        mscale=6, fliplabelside=True)
   ruler2.setp_line(lw=2, color='b')
   ruler2.setp_label(color='b', fontsize=9)
   grat.setp_axislabel("right", label="Offset (Arcmin.)", 
                       visible=True, backgroundcolor='y')


# Main ...
fig = plt.figure(figsize=(7,8))
fig.subplots_adjust(left=0.17, bottom=0.10, right=0.92, 
                    top=0.93, wspace=0.24, hspace=0.34)


header = { 'NAXIS':3,'NAXIS1':100, 'NAXIS2':100 , 'NAXIS3':101 ,
#'CDELT1':  -7.165998823000E-03,
'CDELT1': -11.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1':  -5.128208479590E+01, 'CTYPE1': 'RA---NCP' , 'CUNIT1': 'DEGREE',
'CDELT2':   7.165998823000E-03, 'CRPIX2': 5.100000000000E+01,
'CRVAL2':   6.015388802060E+01, 'CTYPE2': 'DEC--NCP', 'CUNIT2': 'DEGREE',
'CDELT3':   4.199999809000E+00, 'CRPIX3': -2.000000000000E+01,
'CRVAL3':  -2.430000000000E+02, 'CTYPE3': 'VELO-HEL', 'CUNIT3': 'km/s',
'EPOCH ':   2.000000000000E+03,
'FREQ0 ':   1.420405758370E+09
}

f = maputils.FITSimage(externalheader=header)
plotgrat(1,3,1)
plotgrat(2,1,3)
plotgrat(3,3,2)
plotgrat(4,2,3, unitsx="arcsec")
plotgrat(5,1,2, offsetx=True)
plotgrat(6,2,1)
plotgrat(7,3,1, offsetx=True, unitsx='km/s')
plotgrat(8,1,3, offsety=True)

maputils.showall()
