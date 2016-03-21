from kapteyn import maputils
from matplotlib import pylab as plt

header = {'NAXIS': 2 ,'NAXIS1':100 , 'NAXIS2': 100 ,
'CDELT1': -7.165998823000E-03, 'CRPIX1': 5.100000000000E+01 ,
'CRVAL1': -5.128208479590E+01, 'CTYPE1': 'RA---NCP', 'CUNIT1': 'DEGREE ',
'CDELT2': 7.165998823000E-03 , 'CRPIX2': 5.100000000000E+01,
'CRVAL2': 6.015388802060E+01 , 'CTYPE2': 'DEC--NCP ', 'CUNIT2': 'DEGREE',
'CROTA2': 80
}

fig = plt.figure(figsize=(7,7))
fig.suptitle("Messy plot. Rotation is 80 deg.", fontsize=14, color='r')
fig.subplots_adjust(left=0.18, bottom=0.10, right=0.90, 
                    top=0.90, wspace=0.95, hspace=0.20)
frame = fig.add_subplot(2,2,1)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
xpos = -0.42
ypos = 1.2
grat = annim.Graticule()
grat.setp_axislabel(plotaxis=0, xpos=xpos)
frame.set_title("Default", y=ypos)

frame2 = fig.add_subplot(2,2,2)
annim2 = f.Annotatedimage(frame2)
grat2 = annim2.Graticule()
grat2.setp_axislabel(plotaxis=0, xpos=xpos)
grat2.set_tickmode(mode="sw")
frame2.set_title("Switched ticks", y=ypos)

frame3 = fig.add_subplot(2,2,3)
annim3 = f.Annotatedimage(frame3)
grat3 = annim3.Graticule()
grat3.setp_axislabel(plotaxis=0, xpos=xpos)
grat3.set_tickmode(mode="na")
frame3.set_title("Only native ticks", y=ypos)

frame4 = fig.add_subplot(2,2,4)
annim4 = f.Annotatedimage(frame4)
grat4 = annim4.Graticule()
grat4.setp_axislabel(plotaxis=0, xpos=xpos)
grat4.set_tickmode(plotaxis=['bottom','left'], mode="Switch")
grat4.setp_ticklabel(plotaxis=['top','right'], visible=False)
frame4.set_title("Switched and cleaned", y=ypos)

maputils.showall()
