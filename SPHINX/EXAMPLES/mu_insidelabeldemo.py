from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS': 2 ,'NAXIS1':100 , 'NAXIS2': 100 ,
'CDELT1': -7.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1': -5.128208479590E+01, 'CTYPE1': 'RA---NCP', 'CUNIT1': 'DEGREE ',
'CDELT2': 7.165998823000E-03 , 'CRPIX2': 5.100000000000E+01,
'CRVAL2': 6.015388802060E+01 , 'CTYPE2': 'DEC--NCP ', 'CUNIT2': 'DEGREE'
}
 
fig = plt.figure()
frame = fig.add_axes([0.15,0.15,0.8,0.8])
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat2 = annim.Graticule(skyout='Galactic')
grat2.setp_gratline(color='g')
grat2.setp_ticklabel(visible=False)
inswcs0 = grat2.Insidelabels(wcsaxis=0, deltapx=5, deltapy=5)
inswcs1 = grat2.Insidelabels(wcsaxis=1, constval='95d45m')
inswcs0.setp_label(color='r')
inswcs0.setp_label(position="96d0m", color='b', tex=False, fontstyle='italic')
inswcs1.setp_label(position="12d0m", fontsize=14, color='m')
annim.plot()
annim.interact_toolbarinfo()
plt.show()
