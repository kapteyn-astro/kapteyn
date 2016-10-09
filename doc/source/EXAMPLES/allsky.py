from kapteyn import maputils, tabarray
import numpy
import sys
from matplotlib import pyplot as plt


__version__ = '1.91'

epsilon = 0.0000000001

def radians(a):
   return a*numpy.pi/180.0
def degrees(a):
   return a*180.0/numpy.pi
def cylrange():
   X = numpy.arange(0,400.0,30.0); 
   # Replace last two (dummy) values by two values around 180 degrees   
   X[-1] = 180.0 - epsilon
   X[-2] = 180.0 + epsilon
   return X

def polrange(): 
   X = numpy.arange(0,380.0,15); 
   # Replace last two (dummy) values by two values around 180 degrees   
   X[-1] = 180.0 - epsilon
   X[-2] = 180.0 + epsilon
   return X


def getperimeter(grat):
   # Calculate perimeter of QUAD projection 
   xlo, y = grat.gmap.topixel((-45.0-epsilon, 0.0))
   xhi, y = grat.gmap.topixel((315+epsilon, 0.0))
   x, ylo = grat.gmap.topixel((180, -45))
   x, yhi = grat.gmap.topixel((180, 45))
   x1, y = grat.gmap.topixel((45-epsilon, 0.0))
   x, ylolo = grat.gmap.topixel((0, -135+epsilon))
   x, yhihi = grat.gmap.topixel((0, 135-epsilon))
   perimeter = [(xlo,ylo), (x1,ylo), (x1,ylolo), (xhi,ylolo), (xhi,yhihi),
               (x1,yhihi), (x1,yhi), (xlo,yhi), (xlo,ylo)]
   return perimeter


def plotcoast(fn, frame, grat, col='k', lim=100, decim=5, plotsym=None, sign=1.0):
   coasts = tabarray.tabarray(fn, comchar='s')  # Read two columns from file
   for segment in coasts.segments:
      coastseg = coasts[segment].T
      xw = sign * coastseg[1]; yw = coastseg[0] # First one appears to be Latitude
      xs = xw; ys = yw                          # Reset lists which store valid pos.
      if 1:
         # Mask arrays if outside plot box
         xp, yp = grat.gmap.topixel((numpy.array(xs),numpy.array(ys)))
         # Be sure you understand 
         # the operator precedence: (a > 2) | (a < 5) is the proper syntax 
         # because a > 2 | a < 5 will result in an error due to the fact 
         # that 2 | a is evaluated first.
         xp = numpy.ma.masked_where(numpy.isnan(xp) |
                              (xp > grat.pxlim[1]) | (xp < grat.pxlim[0]), xp)
         yp = numpy.ma.masked_where(numpy.isnan(yp) |
                              (yp > grat.pylim[1]) | (yp < grat.pylim[0]), yp)
         # Mask array could be of type numpy.bool_ instead of numpy.ndarray
         if numpy.isscalar(xp.mask):
            xp.mask = numpy.array(xp.mask, 'bool')
         if numpy.isscalar(yp.mask):
            yp.mask = numpy.array(yp.mask, 'bool')
         # Count the number of positions in this list that are inside the box
         xdc = []; ydc = []
         for i in range(len(xp)):
            if not xp.mask[i] and not yp.mask[i]:
               if not i%decim:
                  xdc.append(xp.data[i])
                  ydc.append(yp.data[i])
         if len(xdc) >= lim:
            if plotsym == None:
               frame.plot(xdc, ydc, color=col)
            else:
               frame.plot(xdc, ydc, '.', markersize=1, color=col)


def plotfig(fignum, smallversion=False):
   # Set defaults
   pixel = None
   markerpos = None
   border = None
   title = ''
   titlepos = 1.02
   dec0 = 89.9999999999
   lat_constval = None
   lon_constval = None
   perimeter = None
   lon_world = list(range(0,360,30))
   lat_world = [-dec0, -60, -30, 30, 60, dec0]
   deltapx = deltapy = 0.0
   annotatekwargs0 = {'color':'r'}
   annotatekwargs1 = {'color':'b'}
   plotdata = False
   fsize = 11
   figsize = (7,6)
   datasign = -1
   addangle0 = addangle1 = 0.0
   drawgrid = False
   oblique = None
   framebackgroundcolor = None
   grat = None                            # Just initialize
   ilabs1 = ilabs2 = None
   
   
   fig = plt.figure(figsize=figsize)
   frame = fig.add_axes((0.1,0.05,0.8,0.85))
    
   if fignum == 1:
      # Fig 2 in celestial article (Calabretta et al) shows a  positive cdelt1 
      title = r"""Plate Carree projection (CAR), non oblique with:
$(\alpha_0,\delta_0,\phi_p) = (120^\circ,0^\circ,0^\circ)$. (Cal. fig.2)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CAR',
                'CRVAL1' : 120.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : 5.0,
                'CTYPE2' : 'DEC--CAR',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
                'LONPOLE' : 0.0,
               }
      X = numpy.arange(0,380.0,30.0);
      Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(header, axnum=(1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      #pixel = grat.gmap.topixel((120.0,60))
      markerpos = "120 deg 60 deg"
      header['CRVAL1'] = 0.0
      border = annim.Graticule(header, axnum=(1,2), wylim=(-90,90.0), wxlim=(-180,180),
                               startx=(180-epsilon,-180+epsilon, 0), starty=(-90,0,90))
      lat_world = lon_world = None
   elif fignum == 2:
      title = r"""Plate Carree projection (CAR), oblique with:
$(\alpha_0,\delta_0,\phi_p) = (120^\circ,0^\circ,0^\circ)$
and obviously cdelt1 $>$ 0. (Cal. fig. 2)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CAR',
                'CRVAL1' : 120.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : 5.0,
                'CTYPE2' : 'DEC--CAR',
                'CRVAL2' : 60.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
                'LONPOLE' : 0.0,
               }
      X = numpy.arange(0,360.0,30.0);
      Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      markerpos = "120 deg 60 deg"
      # Get the non-oblique version for the border
      header['CRVAL1'] = 0.0
      header['CRVAL2'] = 0.0
      border = annim.Graticule(header, axnum= (1,2), boxsamples=10000, wylim=(-90,90.0), wxlim=(-180,180),
                               startx=(180-epsilon,-180+epsilon), starty=(-90,90))
      lat_world = lon_world = None
   elif fignum == 3:
      mu = 2.0; gamma = 30.0
      title = r"""Slant zenithal (azimuthal) perspective projection (AZP) with:
$\gamma=30$ and $\mu=2$ (Cal. fig.6)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---AZP',
                'CRVAL1' :0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--AZP',
                'CRVAL2' : dec0, 'CRPIX2' : 30, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : mu, 'PV2_2'  : gamma,
               }
      lowval = (180.0/numpy.pi)*numpy.arcsin(-1.0/mu) + 0.00001  # Calabretta eq.32
      X = numpy.arange(0,360,15.0)
      Y = numpy.arange(-30,90,15.0); 
      Y[0] = lowval                    # Add lowest possible Y to array
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum=(1,2), wylim=(lowval,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs0(0, lw=2)
      grat.setp_lineswcs1(0, lw=2)
      grat.setp_lineswcs1(lowval, lw=2, color='g')
      lat_world = [0, 30, 60, 90]
   elif fignum == 4:
      mu = 2.0; phi = 180.0; theta = 60
      title = r"""Slant zenithal perspective (SZP) with:
($\mu,\phi,\theta)=(2,180,60)$ with special algorithm for border (Cal. fig.7)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---SZP',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--SZP',
                'CRVAL2' : dec0, 'CRPIX2' : 20, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : mu, 'PV2_2'  : phi, 'PV2_3' : theta,
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-90,90,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum=(1,2), wylim=(-90.0,90.0), wxlim=(-180,180),
                             startx=X, starty=Y)

      # PROBLEM: markerpos = "180 deg -30 deg"

      grat.setp_lineswcs0(0, lw=2)
      grat.setp_lineswcs1(0, lw=2)
      # grat.setp_tick(plotaxis=wcsgrat.top, rotation=30, ha='left')
      titlepos = 1.01
      # Special care for the boundary
      # The algorithm seems to work but is not very accurate
      xp = -mu * numpy.cos(theta*numpy.pi/180.0)* numpy.sin(phi*numpy.pi/180.0)
      yp =  mu * numpy.cos(theta*numpy.pi/180.0)* numpy.cos(phi*numpy.pi/180.0)
      zp =  mu * numpy.sin(theta*numpy.pi/180.0) + 1.0
      a = numpy.linspace(0.0,360.0,500)
      arad = a*numpy.pi/180.0
      rho = zp - 1.0
      sigma = xp*numpy.sin(arad) - yp*numpy.cos(arad)
      s = numpy.sqrt(rho*rho+sigma*sigma)
      omega = numpy.arcsin(1/s)
      psi = numpy.arctan2(sigma,rho)
      thetaxrad = psi - omega
      thetax = thetaxrad * 180.0/numpy.pi + 5
      g = grat.addgratline(a, thetax, pixels=False)
      grat.setp_linespecial(g, lw=2, color='c')
      # Select two starting points for a scan in pixel to find borders
      g2 = grat.scanborder(68.26,13,3,3)
      g3 = grat.scanborder(30,66.3,3,3)
      grat.setp_linespecial(g2, color='r', lw=1)
      grat.setp_linespecial(g3, color='r', lw=1)
   elif fignum == 5:
      title = r"Gnomonic projection (TAN) diverges at $\theta=0$. (Cal. fig.8)"
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---TAN',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--TAN',
                'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
               }
      X = numpy.arange(0,360.0,15.0)
      #Y = numpy.arange(0,90,15.0)
      Y = [20, 30,45, 60, 75, 90]
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(20.0,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_constval = 20
      lat_world = [20, 30, 60, dec0]
      grat.setp_lineswcs1(20, color='g', linestyle='--')
   elif fignum == 6:
      title = r"Stereographic projection (STG) diverges at $\theta=-90$. (Cal. fig.9)"
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
               'CTYPE1' : 'RA---STG',
               'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -12.0,
               'CTYPE2' : 'DEC--STG',
               'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 12.0,
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-60,90,10.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-60,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_constval = -62
      lat_world = list(range(-50, 10, 10))
   elif fignum == 7:
      title = r"Slant orthograpic projection (SIN) with: $\xi=\frac{-1}{\sqrt{6}}$ and $\eta=\frac{1}{\sqrt{6}}$ (Cal. fig.10b)"
      xi =  -1/numpy.sqrt(6); eta = 1/numpy.sqrt(6)
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---SIN',
                'CRVAL1' :0.0, 'CRPIX1' : 40, 'CUNIT1' : 'deg', 'CDELT1' : -2,
                'CTYPE2' : 'DEC--SIN',
                'CRVAL2' : dec0, 'CRPIX2' : 30, 'CUNIT2' : 'deg', 'CDELT2' : 2,
                'PV2_1'  : xi, 'PV2_2'  : eta
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-90,90,10.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      # Special care for the boundary (algorithm from Calabretta et al)
      a = numpy.linspace(0,360,500)
      arad = a*numpy.pi/180.0
      thetaxrad = -numpy.arctan(xi*numpy.sin(arad)-eta*numpy.cos(arad))
      thetax = thetaxrad * 180.0/numpy.pi + 0.000001  # Little shift to avoid NaN's at border
      g = grat.addgratline(a, thetax, pixels=False)
      grat.setp_linespecial(g, color='g', lw=1)
      lat_constval = 50
      lon_constval = 180
      lat_world = [0,30,60,dec0]
   elif fignum == 8:
      title = r"Zenithal equidistant projection (ARC). (Cal. fig.11)"
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ARC',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--ARC',
                'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-90,90,30.0)
      Y[0]= -89.999999   # Graticule for -90 exactly is not plotted
      #lat_world = range(-80,80,20)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
   elif fignum == 9:
      title = r"""Zenithal polynomial projection (ZPN) with PV2_n parameters 0 to 7. 
(Cal. fig.12)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZPN',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--ZPN',
                'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0, 
                'PV2_0' : 0.05, 'PV2_1' : 0.975, 'PV2_2' : -0.807, 'PV2_3' : 0.337, 'PV2_4' : -0.065,
                'PV2_5' : 0.01, 'PV2_6' : 0.003,' PV2_7' : -0.001 
               }
      X = numpy.arange(0,360.0,30.0)
      #Y = numpy.arange(-70,90,30.0)   # Diverges (this depends on selected parameters)
      Y = [-70, -60, -45, -30, 0, 15, 30, 45, 60, 90]
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-70,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      # Configure annotations
      lat_constval = -72
      lat_world = [-60, -30, 0, 60, dec0]
      addangle0 = 90.0
      annotatekwargs1.update({'ha':'left'})
   elif fignum == 10:
      title = r"Zenith equal area projection (ZEA). (Cal. fig.13)"
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZEA',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.0,
                'CTYPE2' : 'DEC--ZEA',
                'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.0
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-90,90,30.0)
      Y[0]= -dec0+0.00000001
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs1(position=0, color='g', lw=2)   # Set attributes for graticule line at lat = 0
      lat_world = [-dec0, -30, 30, 60]
   elif fignum == 11:
      title = r"Airy projection (AIR) with $\theta_b = 45^\circ$. (Cal. fig.14)"
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---AIR',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--AIR',
                'CRVAL2' : dec0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0
               }
      X = numpy.arange(0,360.0,30.0)
      Y = numpy.arange(-30,90,10.0)
      # Diverges at dec = -90, start at dec = -30
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-30, -20, -10, 10, 40, 70]
   # CYLINDRICALS
   elif fignum == 12:
      title =  r"Gall's stereographic projection (CYP) with $\mu = 1$ and $\theta_x = 45^\circ$. (Cal. fig.16)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CYP',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
                'CTYPE2' : 'DEC--CYP',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5,
                'PV2_1'  : 1, 'PV2_2' : numpy.sqrt(2.0)/2.0
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-90, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
      annotatekwargs0.update({'va':'bottom', 'ha':'right'})
   elif fignum == 13:
      title = r"Lambert's equal area projection (CEA) with $\lambda = 1$. (Cal. fig.17)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CEA',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--CEA',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
                'PV2_1'  : 1
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-60,-30, 30, 60]
      lon_world = list(range(0,360,30))
      lon_world.append(180.00000001)
   elif fignum == 14:
      title = "Plate Carree projection (CAR). (Cal. fig.18)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CAR',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--CAR',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)  # i.e. include +90 also
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-90, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180.00000001)
   elif fignum == 15:
      title =  "Mercator's projection (MER). (Cal. fig.19)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---MER',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--MER',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
               }
      
      X = cylrange()
      Y = numpy.arange(-80,90,10.0)  # Diverges at +-90 so exclude these values
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(header, axnum= (1,2), wylim=(-80,80.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-90, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
      grat.setp_lineswcs1((-80,80), linestyle='--', color='g')
   elif fignum == 16:
      title = "Sanson-Flamsteed projection (SFL). (Cal. fig.20)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---SFL',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--SFL',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-dec0, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
   elif fignum == 17:
      title = "Parabolic projection (PAR). (Cal. fig.21)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---PAR',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--PAR',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-dec0, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
   elif fignum == 18:
      title=  "Mollweide's projection (MOL). (Cal. fig.22)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---MOL',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--MOL',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)  # Diverges at +-90 so exclude these values
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-60,-30, 30, 60]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
   elif fignum == 19:
      title = "Hammer Aitoff projection (AIT). (Cal. fig.23)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---AIT',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--AIT',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = cylrange()
      Y = numpy.arange(-90,100,30.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lat_world = [-dec0, -60,-30, 30, 60, dec0]
      lon_world = list(range(0,360,30))
      lon_world.append(180+epsilon)
   # CONIC PROJECTIONS
   elif fignum == 20:
      theta_a = 45
      t1 = 20.0; t2 = 70.0
      eta = abs(t1-t2)/2.0
      title = r"""Conic perspective projection (COP) with:
$\theta_a=45^\circ$, $\theta_1=20^\circ$ and $\theta_2=70^\circ$. (Cal. fig.24)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---COP',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.5,
                'CTYPE2' : 'DEC--COP',
                'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.5,
                'PV2_1'  : theta_a, 'PV2_2' : eta
               }
      X = numpy.arange(0,370.0,30.0);  X[-1] = 180+epsilon
      Y = numpy.arange(-30,90,15.0)  # Diverges at theta_a +- 90.0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs1(-30, linestyle='--', color='g')
      lon_world.append(180+epsilon)
   elif fignum == 21:
      theta_a = -45
      t1 = -20.0; t2 = -70.0
      eta = abs(t1-t2)/2.0
      title = r"""Conic equal area projection (COE) with:
$\theta_a=-45$, $\theta_1=-20$ and $\theta_2=-70$. (Cal. fig.25)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---COE',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--COE',
                'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : theta_a, 'PV2_2' : eta
               }
      X = cylrange()
      Y = numpy.arange(-90,91,30.0); Y[-1] = dec0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lon_world.append(180+epsilon)
      lat_constval = 5
      lat_world = [-60,-30,0,30,60]
      addangle0 = -90.0
   elif fignum == 22:
      theta_a = 45
      t1 = 20.0; t2 = 70.0
      eta = abs(t1-t2)/2.0
      title = r"""Conic equidistant projection (COD) with:
$\theta_a=45$, $\theta_1=20$ and $\theta_2=70$. (Cal. fig.26)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---COD',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--COD',
                'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
                'PV2_1'  : theta_a, 'PV2_2' : eta
               }
      X = cylrange()
      Y = numpy.arange(-90,91,15)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lon_world.append(180.0+epsilon)
   elif fignum == 23:
      theta_a = 45
      t1 = 20.0; t2 = 70.0
      eta = abs(t1-t2)/2.0
      title = r"""Conic orthomorfic projection (COO) with:
$\theta_a=45$, $\theta_1=20$ and $\theta2=70$. (Cal. fig.27)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---COO',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--COO',
                'CRVAL2' : theta_a, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : theta_a, 'PV2_2' : eta
               }
      X = cylrange()
      Y = numpy.arange(-30,90,30.0)  # Diverges at theta_a= -90.0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs1(-30, linestyle='--', color='g')
      lon_world.append(180.0+epsilon)
   # POLYCONIC AND PSEUDOCONIC
   elif fignum == 24:
      theta1 = 45
      title = r"Bonne's equal area projection (BON) with $\theta_1=45$. (Cal. fig.28)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---BON',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--BON',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : theta1
               }
      X = polrange()
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lon_world.append(180+epsilon)
   elif fignum == 25:
      title = r"Polyconic projection (PCO). (Cal. fig.29)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---PCO',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--PCO',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0
               }
      X = polrange()
      Y = numpy.arange(-90,100,15.0)
      # !!!!!! Let the world coordinates for constant latitude run from 180,180
      # instead of 0,360. Then one prevents the connection between the two points
      # 179.9999 and 180.0001 which is a jump, but smaller than the definition of
      # a rejected jump in the wcsgrat module.
      # Also we need to increase the value of 'gridsamples' to
      # increase the relative size of a jump.
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2),
                             wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y, gridsamples=2000)
      lon_world.append(180+epsilon)
   # QUAD CUBE PROJECTIONS
   elif fignum == 26:
      title = r"Tangential spherical cube projection (TSC). (Cal. fig.30)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---TSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--TSC',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0
               }
      X = numpy.arange(0,370.0,15.0)
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y)
      
      # Make a polygon for the border
      perimeter = getperimeter(grat)
   elif fignum == 27:
      title = r"COBE quadrilateralized spherical cube projection (CSC). (Cal. fig.31)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--CSC',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0
               }
      X = numpy.arange(0,370.0,15.0)
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y)
      perimeter = getperimeter(grat)
   elif fignum == 28:
      title = r"Quadrilateralized spherical cube projection (QSC). (Cal. fig.32)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---QSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--QSC',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = numpy.arange(-180,180,15)
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      perimeter = getperimeter(grat)
      deltapx = 1
      plotdata = True
   elif fignum == 280:
      title = r"Quadrilateralized spherical cube projection (QSC). (Cal. fig.32)"
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---QSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--QSC',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
               }
      X = numpy.arange(-180,180,15)
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      lon_world = list(range(0,360,30))
      lat_world = [-90, -60, -30, 30, 60, dec0]
      perimeter = getperimeter(grat)
      deltapx = 1
      plotdata = True
   elif fignum == 270:
      title = r"Quadrilateralized spherical cube projection (QSC). (Cal. fig.32)"
      header = {'NAXIS'  : 3, 'NAXIS1': 100, 'NAXIS2': 80, 'NAXIS3' : 6,
                'CTYPE1' : 'RA---QSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -7.0,
                'CTYPE2' : 'DEC--QSC',
                'CRVAL2' : 0.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 7.0,
                'CTYPE3' : 'CUBEFACE',
                'CRVAL3' : 0, 'CRPIX3' : 2,'CDELT3' : 90, 'CUNIT3' : 'deg',
               }
      X = numpy.arange(0,370.0,15.0)
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y)
      lon_world = list(range(-180,180,30))
      lat_world = [-90, -60, -30, 30, 60, dec0]
      perimeter = getperimeter(grat)
   elif fignum == 29:
      title = r"""Zenith equal area projection (ZEA) oblique with:
$\alpha_p=0$, $\delta_p=30$ and $\phi_p=180$. (Cal. fig.33a)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZEA',
                'CRVAL1' :0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
                'CTYPE2' : 'DEC--ZEA',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5,
               }
      X = numpy.arange(0,360,15.0)
      Y = numpy.arange(-90,90,15.0)
      Y[0]= -dec0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
   elif fignum == 30:
      title = r"""Zenith equal area projection (ZEA) oblique with:
$\alpha_p=45$, $\delta_p=30$ and $\phi_p=180$. (Cal. fig.33b)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZEA',
                'CRVAL1' :45.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
                'CTYPE2' : 'DEC--ZEA',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5
               }
      X = numpy.arange(0,360.0,15.0)
      Y = numpy.arange(-90,90,15.0)
      Y[0]= -dec0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs0((0,180), color='g', lw=2)
   elif fignum == 31:
      title = r"""Zenith equal area projection (ZEA) oblique with:
$\alpha_p=0$, $\theta_p=30$ and $\phi_p = 150$.  (Cal. fig.33c)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZEA',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
                'CTYPE2' : 'DEC--ZEA',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5,
                'PV1_3' : 150.0    # Works only with patched wcslib 4.3
               }
      X = numpy.arange(0,360.0,15.0)
      Y = numpy.arange(-90,90,15.0)
      Y[0]= -dec0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs0((0,180), color='g', lw=2)
   elif fignum == 32:
      title = r"""Zenith equal area projection (ZEA) oblique with:
$\alpha_p=0$, $\theta_p=30$ and $\phi_p = 75$ (Cal. fig.33d)"""
      header = {'NAXIS' : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---ZEA',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -3.5,
                'CTYPE2' : 'DEC--ZEA',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 3.5,
                'PV1_3' : 75.0
               }
      X = numpy.arange(0,360.0,15.0)
      Y = numpy.arange(-90,90,15.0)
      Y[0]= -dec0
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs0((0,180), color='g', lw=2)
   elif fignum == 33:
      theta_a = 45.0
      t1 = 20.0; t2 = 70.0
      eta = abs(t1-t2)/2.0
      title = r"""Conic equidistant projection (COD) oblique with $\theta_a=45$, $\theta_1=20$
and $\theta_2=70$, $\alpha_p = 0$, $\delta_p = 30$, $\phi_p = 75$ also: 
$(\phi_0,\theta_0) = (0,90^\circ)$. (Cal. fig.33d)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---COD',
                'CRVAL1' : 0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -5.0,
                'CTYPE2' : 'DEC--COD',
                'CRVAL2' : 30, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 5.0,
                'PV2_1'  : theta_a, 
                'PV2_2' :  eta,
                'PV1_1'  : 0.0, 'PV1_2' : 90.0,  # IMPORTANT. This is a setting from section 7.1, p 1103
                'LONPOLE' :75.0
               }
      X = numpy.arange(0,370.0,15.0);  X[-1] = 180.000001
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      # Draw border with standard graticule
      header['CRVAL1'] = 0.0
      header['CRVAL2'] = theta_a
      header['LONPOLE'] = 0.0
      del header['PV1_1']
      del header['PV1_2']
      # Non oblique version as border
      border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                                 startx=(180-epsilon, -180+epsilon), starty=(-90,90))
      border.setp_lineswcs0(color='g')   # Show borders in different color
      border.setp_lineswcs1(color='g')
   elif fignum == 34:
      title = r"""Hammer Aitoff projection (AIT) oblique with:
$(\alpha_p,\delta_p) = (0^\circ,30^\circ)$, $\phi_p = 75^\circ$ also: 
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$. (Cal. fig.34d)"""
      # Header works only with a patched wcslib 4.3
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---AIT',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--AIT',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0, 
                'LONPOLE' :75.0,
                'PV1_1'  : 0.0, 'PV1_2' : 90.0,  # IMPORTANT. This is a setting from Cal.section 7.1, p 1103
               }
      X = numpy.arange(0,390.0,15.0); 
      Y = numpy.arange(-90,100,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      # Draw border with standard graticule
      header['CRVAL1'] = 0.0
      header['CRVAL2'] = 0.0
      del header['PV1_1']
      del header['PV1_2']
      header['LONPOLE'] = 0.0
      header['LATPOLE'] = 0.0
      border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                               startx=(180-epsilon, -180+epsilon), skipy=True)
      border.setp_lineswcs0(color='g')   # Show borders in different color
      border.setp_lineswcs1(color='g')
   elif fignum == 35:
      title = r"""COBE quadrilateralized spherical cube projection (CSC) oblique with:
$(\alpha_p,\delta_p) = (0^\circ,30^\circ)$, $\phi_p = 75^\circ$ also: 
$(\phi_0,\theta_0) = (0^\circ,90^\circ)$. (Cal. fig.34d)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---CSC',
                'CRVAL1' : 0.0, 'CRPIX1' : 85, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--CSC',
                'CRVAL2' : 30.0, 'CRPIX2' : 40, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'LONPOLE': 75.0,
                'PV1_1'  : 0.0, 'PV1_2' : 90.0,
               }
      X = numpy.arange(0,370.0,30.0)
      Y = numpy.arange(-90,100,30.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y)
      # Take border from non-oblique version
      header['CRVAL2'] = 0.0
      del header['PV1_1']
      del header['PV1_2']
      del header['LONPOLE']
      border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                               skipx=True, skipy=True)
      perimeter = getperimeter(border)
   elif fignum == 36: 
      title = 'Earth in zenithal perspective (AZP). (Cal. fig.36)'
      # The ctype's are TLON, TLAT. These are recognized by WCSlib as longitude and latitude.
      # Any other prefix is also valid.
      header = {'NAXIS'  : 2, 'NAXIS1': 2048, 'NAXIS2': 2048,
                'PC1_1' : 0.9422, 'PC1_2' : -0.3350,
                'PC2_1' : 0.3350, 'PC2_2' : 0.9422,
                'CTYPE1' : 'TLON-AZP',
                'CRVAL1' : 31.15, 'CRPIX1' : 681.67, 'CUNIT1' : 'deg', 'CDELT1' : 0.008542,
                'CTYPE2' : 'TLAT-AZP',
                'CRVAL2' : 30.03, 'CRPIX2' : 60.12, 'CUNIT2' : 'deg', 'CDELT2' : 0.008542,
                'PV2_1' : -1.350, 'PV2_2' : 25.8458,
                'LONPOLE' : 143.3748,
               }
      X = numpy.arange(-30,60.0,10.0)
      Y = numpy.arange(-40,65,10.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-30,90.0), wxlim=(-20,60),
                             startx=X, starty=Y, gridsamples=4000)
      grat.setp_lineswcs1(color='#B30000')
      grat.setp_lineswcs0(color='#B30000')
      grat.setp_lineswcs0(0, color='r', lw=2)
      grat.setp_plotaxis('bottom', mode='all_ticks', label='Latitude / Longitude')
      grat.setp_plotaxis('left', mode='switched_ticks', label='Latitude')
      grat.setp_plotaxis('right', mode='native_ticks')
      grat.setp_tick(wcsaxis=0, color='g')
      grat.setp_tick(wcsaxis=1, color='m')
      grat.setp_tick(wcsaxis=1, plotaxis=('bottom','right'), color='m', rotation=-30, ha='left')
      #grat.setp_tick(plotaxis=wcsgrat.right, backgroundcolor='yellow')
      grat.setp_tick(plotaxis='left', position=-10, visible=False)
      g = grat.scanborder(560, 1962, 2)
      grat.setp_linespecial(g, color='b', lw=2)
      lat_world = lon_world = []
      drawgrid = True
      plotdata = True
      datasign = +1
      # Proof that WCSlib thinks TLON, TLAT are valid longitudes & latitudes
      print("TLON and TLAT are recognized as:", grat.gmap.types)
   elif fignum == 37: 
      title = 'WCS polyconic (PGSBOX fig.1)'
      rot = 30.0 *numpy.pi/180.0
      header = {'NAXIS'  : 2, 'NAXIS1': 512, 'NAXIS2': 512,
                'CTYPE1' : 'RA---PCO',
                'PC1_1' : numpy.cos(rot), 'PC1_2' : numpy.sin(rot),
                'PC2_1' : -numpy.sin(rot), 'PC2_2' : numpy.cos(rot),
                'CRVAL1' : 332.0, 'CRPIX1' : 192, 'CUNIT1' : 'deg', 'CDELT1' : -1.0/5.0,
                'CTYPE2' : 'DEC--PCO',
                'CRVAL2' : 40.0, 'CRPIX2' : 640, 'CUNIT2' : 'deg', 'CDELT2' : 1.0/5.0,
                'LONPOLE' : -30.0,
               }
      X = numpy.arange(-180,180.0,15.0);
      Y = numpy.arange(-90,120,15.0) 
      # Here we demonstrate how to avoid a jump at the right corner boundary 
      # of the plot by increasing the value of 'gridsamples'.
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                             startx=X, starty=Y, gridsamples=4000)
      grat.setp_tick(position=(-15.0,-45.0, -60.0,-75.0), visible=False)
      deltapx = 3
      header['CRVAL1'] = 0.0
      header['CRVAL2'] = 0.0
      header['LONPOLE'] = 999
      border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                                 startx=(180-epsilon, -180+epsilon), starty=(-90,90))
      border.setp_gratline((0,1), color='g', lw=2)
      border.setp_plotaxis((0,1,2,3), mode='no_ticks', visible=False)
   elif fignum == 38:
      theta_a = 60.0; eta = 15.0 
      title = r"WCS conic equal area projection with $\theta_a=60$ and $\eta=15$ (Cal. PGSBOX fig.2)"
      header = {'NAXIS'  : 2, 'NAXIS1': 512, 'NAXIS2': 512,
                'CTYPE1' : 'RA---COE',
                'CRVAL1' : 90.0, 'CRPIX1' : 256, 'CUNIT1' : 'deg', 'CDELT1' : -1.0/3.0,
                'CTYPE2' : 'DEC--COE',
                'CRVAL2' : 30.0, 'CRPIX2' : 256, 'CUNIT2' : 'deg', 'CDELT2' : 1.0/3.0,
                'LONPOLE' : 150.0,
                'PV2_1'  : theta_a, 'PV2_2' : eta
               }
      X = numpy.arange(0,390.0,30.0);
      Y = numpy.arange(-90,120,30.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      grat.setp_lineswcs0(color='r')
      grat.setp_lineswcs1(color='b')
      grat.setp_tick(plotaxis=1, position=(150.0,210.0), visible=False)
      deltapx = 10
      # Draw border with standard graticule
      header['CRVAL1'] = 0.0;
      header['CRVAL2'] = 60.0
      header['LONPOLE'] = 999
      header['LATPOLE'] = 999
      border = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(-180,180),
                               startx=(180-epsilon, -180+epsilon), starty=(-90,90))
      border.setp_gratline((0,1), color='g', lw=2)
      border.setp_plotaxis((0,1,2,3), mode='no_ticks', visible=False)
      framebackgroundcolor = 'k'  # i.e. black
      annotatekwargs0.update({'color':'w'})
      annotatekwargs1.update({'color':'w'})
   elif fignum == 39:
      theta1 = 35
      title = r"""Bonne's equal area projection (BON) with conformal latitude $\theta_1=35$ and
$\alpha_p=0^\circ$, $\theta_p=+45^\circ$ and N.C.P. at $(45^\circ,0^\circ)$. (Cal. PGSBOX example)"""
      header = {'NAXIS'  : 2, 'NAXIS1': 100, 'NAXIS2': 80,
                'CTYPE1' : 'RA---BON',
                'CRVAL1' : 0.0, 'CRPIX1' : 50, 'CUNIT1' : 'deg', 'CDELT1' : -4.0,
                'CTYPE2' : 'DEC--BON',
                'CRVAL2' : 0.0, 'CRPIX2' : 35, 'CUNIT2' : 'deg', 'CDELT2' : 4.0,
                'PV2_1'  : theta1
               }
      X = polrange()
      Y = numpy.arange(-90.0,100.0,15.0)
      f = maputils.FITSimage(externalheader=header)
      annim = f.Annotatedimage(frame)
      grat = annim.Graticule(axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                             startx=X, starty=Y)
      annotatekwargs0.update({'visible':False})
      annotatekwargs1.update({'visible':False})
      grat.setp_lineswcs0(color='#339333')  # Dark green
      grat.setp_lineswcs1(color='#339333')
      header['LONPOLE'] = 45.0  # Or PV1_3
      header['CRVAL1'] = 0.0
      header['CRVAL2'] = 45.0
      oblique = annim.Graticule(header, axnum= (1,2), wylim=(-90,90.0), wxlim=(0,360),
                                startx=X, starty=Y)
      oblique.setp_lineswcs0(0.0, color='y')
      oblique.setp_lineswcs1(0.0, color='y')
      oblique.setp_lineswcs0(list(range(15,360,45)), color='b')
      oblique.setp_lineswcs1([15,-15,60, -60], color='b')
      oblique.setp_lineswcs0(list(range(30,360,45)), color='r')
      oblique.setp_lineswcs1([30,-30,75, -75], color='r')
      oblique.setp_lineswcs0(list(range(45,360,45)), color='w')
      oblique.setp_lineswcs1((-45,45), color='w')
      framebackgroundcolor = 'k'
      if not smallversion:
         txt ="""Green:  Native, non-oblique graticule.  Yellow: Equator and prime meridian
Others: Colour coded oblique graticule"""
         plt.figtext(0.1, 0.008, txt, fontsize=10)
   
   #------------------------------------ Settings ----------------------------------------
   
   # Apply some extra settings
   if framebackgroundcolor != None:
      frame.set_axis_bgcolor(framebackgroundcolor)

   # Plot coastlines if required
   if plotdata:
      if fignum == 36:
         plotcoast('WDB/world.txt', frame, grat, col='k', lim=100)
      else:
         plotcoast('WDB/world.txt', frame, grat, col='r', lim=50, decim=20, plotsym=',', sign=-1)

   # Plot alternative borders
   if perimeter != None:
      p = plt.Polygon(perimeter, facecolor='#d6eaef', lw=2)
      frame.add_patch(p)
      Xp, Yp = list(zip(*perimeter))
      frame.plot(Xp, Yp, color='r')

   # Plot labels inside graticule if required
   annotatekwargs0.update({'fontsize':fsize})
   annotatekwargs1.update({'fontsize':fsize})
   ilabs1 = grat.Insidelabels(wcsaxis=0, 
                        world=lon_world, constval=lat_constval, deltapx=deltapx, deltapy=deltapy, 
                        addangle=addangle0, fmt="$%g$", **annotatekwargs0)
   ilabs2 = grat.Insidelabels(wcsaxis=1, 
                        world=lat_world, constval=lon_constval, deltapx=deltapx, deltapy=deltapy, 
                        addangle=addangle1, fmt="$%g$", **annotatekwargs1)

   # Plot just 1 pixel c.q. marker
   if markerpos != None:
      annim.Marker(pos=markerpos, marker='o', color='red' )

   if drawgrid:
      pixellabels = annim.Pixellabels(plotaxis=(2,3))

   # Plot the title
   if smallversion:
      t = frame.set_title(title, color='g', fontsize=10)
   else:
      t = frame.set_title(title, color='g', fontsize=13, linespacing=1.5)
   t.set_y(titlepos)
   #gratplot.plot()
   annim.plot()
   annim.interact_toolbarinfo()

   if smallversion:
      fn = "allsky-fig%d_small.png"%fignum
   else:
      fn = "allsky-fig%d.png"%fignum
   plt.show()



if __name__ == "__main__":
   # Process command line arguments. First is number of the all sky plot,
   # second argument can be any character and sets the figure format to small.
   #
   # Usage:
   # python plotwcs.py [<figure number> <s>]
   #    e.g.:
   #   python plotwcs.py          # You will be prompted for a figure number
   #   python plotwcs.py 23       # Figure 23
   #   python plotwcs.py 23 s     # Figure 23 small version
   #
   smallversion = False
   if len(sys.argv) > 1:
      fignum = int(sys.argv[1])
   if len(sys.argv) > 2:
      figsize = (5.0,5.0)
      smallversion = True
      fsize = 8
   
   if fignum == None:
      print("enter number of figures", file=sys.stderr)
      #fignum = eval(input("Enter number of figure: "))
   
   plotfig(fignum, smallversion)
