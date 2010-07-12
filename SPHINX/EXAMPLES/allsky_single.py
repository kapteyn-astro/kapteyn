from kapteyn import maputils
from numpy import arange
from matplotlib import pyplot as plt

dec0 = 89.9999999999   # Avoid plotting on the wrong side
header = {'NAXIS'  :  2, 
          'NAXIS1' :  100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' :  0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 
          'CDELT1' : -5.0, 'CTYPE2' : 'DEC--TAN',
          'CRVAL2' :  dec0, 'CRPIX2' : 40, 
          'CUNIT2' : 'deg', 'CDELT2' : 5.0,
         }
X = arange(0,360.0,15.0)
Y = [20, 30,45, 60, 75, 90]

fig = plt.figure(figsize=(7,6))
frame = fig.add_axes((0.1,0.1,0.8,0.8))
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(wylim=(20.0,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
lon_world = range(0,360,30)
lat_world = [20, 30, 60, dec0]
grat.setp_lineswcs1(20, color='g', linestyle='--')

# Plot labels inside the plot
lon_constval = None
lat_constval = 20
lon_kwargs = {'color':'r', 'fontsize':15}
lat_kwargs = {'color':'b', 'fontsize':10}
grat.Insidelabels(wcsaxis=0, 
                  world=lon_world, constval=lat_constval, 
                  fmt="Dms",
                  **lon_kwargs)
grat.Insidelabels(wcsaxis=1, 
                  world=lat_world, constval=lon_constval, 
                  fmt="Dms",
                  **lat_kwargs)
   
annim.plot()
# Set title for Matplotlib
titlepos = 1.02
title = r"Gnomonic projection (TAN) diverges at $\theta=0^\circ$. (Cal. fig.8)"
t = frame.set_title(title, color='g')
t.set_y(titlepos)

plt.show()
