from kapteyn import maputils
import numpy
from service import *

fignum = 4
fig = plt.figure(figsize=figsize)
frame = fig.add_axes(plotbox)
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
grat = annim.Graticule(axnum=(1,2), 
                       wylim=(-90.0,90.0), wxlim=(-180,180),
                       startx=X, starty=Y)

grat.setp_lineswcs0(0, lw=2)
grat.setp_lineswcs1(0, lw=2)
# Special care for the boundary
# The algorithm seems to work but is not very accurate
xp = -mu * numpy.cos(theta*numpy.pi/180.0)* numpy.sin(phi*numpy.pi/180.0)
yp =  mu * numpy.cos(theta*numpy.pi/180.0)* numpy.cos(phi*numpy.pi/180.0)
zp =  mu * numpy.sin(theta*numpy.pi/180.0) + 1.0
a = numpy.linspace(0.0,360.0,500)
arad = a*numpy.pi/180.0
rho = zp - 1.0
sigma = xp*numpy.sin(arad) - yp*numpy.cos(arad)
sq = numpy.sqrt(rho*rho+sigma*sigma)
omega = numpy.arcsin(1/sq)
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
lon_world = range(0,360,30)
lat_world = [-60, -30, 30, 60]
labkwargs0 = {'color':'r', 'va':'center', 'ha':'center'}
labkwargs1 = {'color':'b', 'va':'bottom', 'ha':'right'}
doplot(frame, fignum, annim, grat, title, 
       lon_world=lon_world, lat_world=lat_world, 
       labkwargs0=labkwargs0, labkwargs1=labkwargs1,
       markerpos=markerpos)
