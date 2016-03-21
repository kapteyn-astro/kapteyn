from kapteyn import maputils
from matplotlib import pylab as plt


header = {'NAXIS': 2 ,'NAXIS1':100 , 'NAXIS2': 100 ,
'CDELT1': -7.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1': -5.128208479590E+01, 'CTYPE1': 'RA---NCP', 'CUNIT1': 'DEGREE ',
'CDELT2': 7.165998823000E-03 , 'CRPIX2': 5.100000000000E+01,
'CRVAL2': 6.015388802060E+01 , 'CTYPE2': 'DEC--NCP ', 'CUNIT2': 'DEGREE'
}


fig = plt.figure()
#frame = fig.add_axes([0.15,0.15,0.8,0.8])
frame = fig.add_subplot(1,1,1)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat.setp_gratline(visible=False)
grat.setp_ticklabel(plotaxis="bottom", position="20h34m", fmt="%6f")
grat.setp_tickmark(plotaxis="bottom", position="20h34m", 
                   color='b', markeredgewidth=4, markersize=20)
fig.text(0.5, 0.5, "Empty", fontstyle='italic', fontsize=18, ha='center',
         color='r')
annim.plot()
plt.show()
