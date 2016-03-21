import numpy
from kapteyn import maputils
from matplotlib.pyplot import show, figure
import csv   # Read some poitions from file in Comma Separated Values format

# Some initializations
blankcol = "#334455"                  # Represent undefined values by this color
epsilon = 0.0000000001
figsize = (9,7)                       # Figure size in inches
plotbox = (0.1,0.05,0.8,0.8)
fig = figure(figsize=figsize)
frame = fig.add_axes(plotbox)

Basefits = maputils.FITSimage("allsky_raw.fits")  # Here is your downloaded FITS file in rectangular coordinates
Basefits.hdr['CTYPE1'] = 'GLON-CAR'               # For transformations we need to give it a projection type
Basefits.hdr['CTYPE2'] = 'GLAT-CAR'               # CAR is rectangular

# Use some header values to define reprojection parameters
cdelt1 = Basefits.hdr['CDELT1']
cdelt2 = Basefits.hdr['CDELT2']
naxis1 = Basefits.hdr['NAXIS1']
naxis2 = Basefits.hdr['NAXIS2']

# Header works only with a patched wcslib 4.3
# Note that changing CRVAL1 to 180 degerees, shifts the plot 180 deg.
header = {'NAXIS'  : 2, 'NAXIS1': naxis1, 'NAXIS2': naxis2,
          'CTYPE1' : 'GLON-AIT',
          'CRVAL1' : 0, 'CRPIX1' : naxis1//2, 'CUNIT1' : 'deg', 'CDELT1' : cdelt1,
          'CTYPE2' : 'GLAT-AIT',
          'CRVAL2' : 30.0, 'CRPIX2' : naxis2//2, 'CUNIT2' : 'deg', 'CDELT2' : cdelt2,
          'LONPOLE' :60.0,
          'PV1_1'  : 0.0, 'PV1_2' : 90.0,  # IMPORTANT. This is a setting from Cal.section 7.1, p 1103
         }
Reprojfits = Basefits.reproject_to(header)
annim_rep = Reprojfits.Annotatedimage(frame)
annim_rep.set_colormap("heat.lut")               # Set color map before creating Image object
annim_rep.set_blankcolor(blankcol)               # Background are NaN's (blanks). Set color here
annim_rep.Image(vmin=30000, vmax=150000)         # Just a selection of two clip levels
annim_rep.plot()

# Draw the graticule, but do not cover near -90 to prevent ambiguity
X = numpy.arange(0,390.0,15.0); 
Y = numpy.arange(-75,90,15.0)
f = maputils.FITSimage(externalheader=header)
annim = f.Annotatedimage(frame)
grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                       startx=X, starty=Y)
grat.setp_lineswcs0(0, color='w', lw=2)
grat.setp_lineswcs1(0, color='w', lw=2)

# Draw border with standard graticule, just to make the borders look smooth
header['CRVAL1'] = 0.0
header['CRVAL2'] = 0.0
del header['PV1_1']
del header['PV1_2']
header['LONPOLE'] = 0.0
header['LATPOLE'] = 0.0
border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                         startx=(180-epsilon, -180+epsilon), skipy=True)
border.setp_lineswcs0(color='w', lw=2)   # Show borders in arbitrary color (e.g. background color)
border.setp_lineswcs1(color='w', lw=2)

# Plot the 'inside' graticules
lon_constval = 0.0
lat_constval = 0.0
lon_fmt = 'Dms'; lat_fmt = 'Dms'  # Only Degrees must be plotted
addangle0 = addangle1=0.0
deltapx0 = deltapx1 = 1.0
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'r', 'va':'center', 'ha':'center'}
lon_world = range(0,360,30)
lat_world = [-60, -30, 30, 60]

ilabs1 = grat.Insidelabels(wcsaxis=0,
                     world=lon_world, constval=lat_constval,
                     deltapx=1.0, deltapy=1.0,
                     addangle=addangle0, fmt=lon_fmt, **labkwargs0)
ilabs2 = grat.Insidelabels(wcsaxis=1,
                     world=lat_world, constval=lon_constval,
                     deltapx=1.0, deltapy=1.0,
                     addangle=addangle1, fmt=lat_fmt, **labkwargs1)

# Read marker positions (in 0h0m0s 0d0m0s format) from file
reader = csv.reader(open("positions.txt"), delimiter=' ',  skipinitialspace=True)
for line in reader:
    if line:
       hms, dms = line
       postxt = "{eq fk4-no-e} "+hms+" {} "+dms   # Define the sky system of the source
       print postxt
       annim.Marker(pos=postxt, marker='*', color='yellow', ms=20)


# Plot a title
titlepos = 1.02
title = r"""All sky map in Hammer Aitoff projection (AIT) oblique with:
$(\alpha_p,\delta_p) = (0^\circ,30^\circ)$, $\phi_p = 75^\circ$ also:
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$."""
t = frame.set_title(title, color='g', fontsize=13, linespacing=1.5)
t.set_y(titlepos)

annim.plot()
annim.interact_toolbarinfo()
annim_rep.interact_imagecolors()
show()