from kapteyn import wcsgrat
import numpy
from matplotlib import pyplot as plt

dec0 = 89.9999999999   # Avoid plotting on the wrong side
header = {'NAXIS'  : 2, 
          'NAXIS1' : 100, 'NAXIS2': 80,
          'CTYPE1' : 'RA---TAN',
          'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 
          'CDELT1' : -5.0, 'CTYPE2' : 'DEC--TAN',
          'CRVAL2' : dec0, 'CRPIX2' : 40, 
          'CUNIT2' : 'deg', 'CDELT2' : 5.0,
         }
X = numpy.arange(0,360.0,15.0)
Y = [20, 30,45, 60, 75, 90]
grat = wcsgrat.Graticule(header, axnum= (1,2), 
                         wylim=(20.0,90.0), wxlim=(0,360),
                         startx=X, starty=Y)
lon_world = range(0,360,30)
lat_world = [20, 30, 60, dec0]
grat.setp_lineswcs1(20, color='g', linestyle='--')

# Plot labels inside the plot
lon_constval = None
lat_constval = 20
lon_kwargs = {'color':'r', 'fontsize':12}
lat_kwargs = {'color':'b', 'fontsize':10}
inlabs0 = grat.insidelabels(wcsaxis=0, 
                     world=lon_world, constval=lat_constval, 
                     **lon_kwargs)
inlabs1 = grat.insidelabels(wcsaxis=1, 
                     world=lat_world, constval=lon_constval, 
                     **lat_kwargs)
   
# Select figure size ans create Matplotlib Axes object
# figsize = grat.figsize
fig = plt.figure(figsize=(8,8))
frame = fig.add_axes(grat.axesrect, aspect=grat.aspectratio, 
                     adjustable='box', autoscale_on=False, clip_on=True)

gratplot = wcsgrat.Plotversion('matplotlib', fig, frame)
gratplot.add( [grat,inlabs0,inlabs1] )

# Set title for Matplotlib
titlepos = 1.02
title = r"Gnomonic projection (TAN) diverges at $\theta=0$. (Cal. fig.8)"
t = frame.set_title(title, color='g')
t.set_y(titlepos)

plt.show()
