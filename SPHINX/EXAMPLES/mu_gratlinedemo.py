from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS': 2 ,'NAXIS1':100 , 'NAXIS2': 100 ,
'CDELT1':  -7.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1':  -5.128208479590E+01, 'CTYPE1': 'RA---NCP', 'CUNIT1': 'DEGREE ',
'CDELT2':   7.165998823000E-03 , 'CRPIX2': 5.100000000000E+01, 
'CRVAL2': 6.015388802060E+01 , 'CTYPE2': 'DEC--NCP ', 'CUNIT2': 'DEGREE'
}

fig = plt.figure(figsize=(6,5.2))
frame = fig.add_subplot(1,1,1)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat.setp_gratline(lw=2)
grat.setp_gratline(wcsaxis=0, color='r')
grat.setp_gratline(wcsaxis=1, color='g')
grat.setp_gratline(wcsaxis=1, position=60.25, linestyle=':') 
grat.setp_gratline(wcsaxis=0, position="20d34m0s", linestyle=':') 
# If invisible, use: grat.setp_gratline(visible=False)

annim.plot()
plt.show()

