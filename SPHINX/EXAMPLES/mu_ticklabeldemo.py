from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS': 2 ,'NAXIS1':100 , 'NAXIS2': 100 ,
'CDELT1': -7.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1': -5.128208479590E+01, 'CTYPE1': 'RA---NCP', 'CUNIT1': 'DEGREE ',
'CDELT2':  7.165998823000E-03, 'CRPIX2': 5.100000000000E+01,
'CRVAL2':  6.015388802060E+01, 'CTYPE2': 'DEC--NCP ', 'CUNIT2': 'DEGREE'
}

fig = plt.figure()
frame = fig.add_axes([0.20,0.15,0.75,0.8])
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat2 = annim.Graticule(skyout='Galactic')
grat.setp_ticklabel(plotaxis="bottom", position="20h34m", fmt="%g",
                    color='r', rotation=30)
grat.setp_ticklabel(plotaxis='left', color='b', rotation=20,
                    fontsize=14, fontweight='bold', style='italic')
grat.setp_ticklabel(plotaxis='left', color='m', position="60d0m0s", 
                    fmt="DMS", tex=False) 
grat.setp_axislabel(plotaxis='left', xpos=-0.25, ypos=0.5)
# Rotation is inherited from previous setting 
grat2.setp_gratline(color='g')
grat2.setp_ticklabel(visible=False)
grat2.setp_axislabel(visible=False)

annim.plot()
plt.show()
