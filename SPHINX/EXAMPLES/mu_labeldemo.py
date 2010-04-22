from kapteyn import maputils
from matplotlib import pylab as plt

header = {
'NAXIS':                     2 ,
'NAXIS1':                  100 ,
'NAXIS2':                  100 ,
'CDELT1':  -7.165998823000E-03 ,
'CRPIX1':   5.100000000000E+01 ,
'CRVAL1':  -5.128208479590E+01 ,
'CTYPE1': 'RA---NCP        ' ,
'CUNIT1': 'DEGREE          ' ,
'CDELT2':   7.165998823000E-03 ,
'CRPIX2':   5.100000000000E+01 ,
'CRVAL2':   6.015388802060E+01 ,
'CTYPE2': 'DEC--NCP        ' ,
'CUNIT2': 'DEGREE          ' ,
}

fig = plt.figure(figsize=(6,5.2))
frame = fig.add_axes([0.15,0.15,0.8,0.8])
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule()
grat.setp_axislabel(fontstyle='italic')      # Apply to all
grat.setp_axislabel("top", visible=True, xpos=0.0, ypos=1.0, rotation=180)
grat.setp_axislabel("left", 
                    backgroundcolor='y', 
                    color='b', 
                    style='oblique',
                    weight='bold', 
                    ypos=0.3)
grat.setp_axislabel("bottom",                # Label in LaTeX
                    label=r"$\mathrm{Right\ Ascension\ (2000)}$", 
                    fontsize=14)
annim.plot()
plt.show()
