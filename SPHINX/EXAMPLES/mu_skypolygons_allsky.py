from kapteyn import maputils
from matplotlib import pyplot as plt
import numpy


def shapes(proj, fig, plnr, crval2=0.0, **pv):
   naxis1 = 800; naxis2 = 800
   header = {'NAXIS': 2,  
             'NAXIS1': naxis1, 'NAXIS2': naxis2, 
             'CRPIX1': naxis1/2.0, 'CRPIX2': naxis2/2.0,
             'CRVAL1': 0.0,   'CRVAL2': crval2, 
             'CDELT1': -0.5,  'CDELT2': 0.5, 
             'CUNIT1': 'deg', 'CUNIT2': 'deg',
             'CTYPE1': 'RA---%s'%proj, 'CTYPE2': 'DEC--%s'%proj}
   if len(pv):
      header.update(pv)
   
   print header
   X = numpy.arange(0,390.0,30.0); X[-1] = 180+0.00000001
   Y = numpy.arange(-90,91,30.0)
   f = maputils.FITSimage(externalheader=header)
   frame = fig.add_subplot(2,2,plnr)
   annim = f.Annotatedimage(frame)
   grat = annim.Graticule(axnum=(1,2),
                        wylim=(-90.0,90.0), wxlim=(-180,180),
                        startx=X, starty=Y)
   grat.setp_gratline(color='0.75')
   if plnr in [1,2]:
     grat.setp_axislabel(plotaxis='bottom', visible=False)
   print "Projection %d is %s" % (plnr, proj)
   # Ellipse centered on crossing of two graticule lines
   try:
      annim.Skypolygon("ellipse", cpos="5h00m 20d0m", major=50, minor=30,
                        pa=-30.0, fill=False)
   except:
      print "Failed to plot ellipse"
   # Ellipse at given pixel coordinates
   try:
      cpos = "%f %f"%(naxis1/2.0+20, naxis2/2.0+10)
      annim.Skypolygon("ellipse", cpos=cpos, major=20, minor=10,
                        pa=-30.0, fc='m')  
   except:
      print "Failed to plot ellipse"
   # Circle with radius in arc minutes
   try:
      annim.Skypolygon("ellipse", cpos="0 deg 60 deg", 
                     major=30, minor=30,
                     fc='g', alpha=0.3, lw=3, ec='r') 
   except:
      print "Failed to plot circle"
   # Rectangle at the projection center
   try:
      annim.Skypolygon("rectangle", cpos="pc pc", major=50, minor=20,
                     pa=30.0, ec='g', fc='b', alpha=0.3)
   except:
      print "Failed to plot rectangle"
   # Square centered at 315 deg -45 deg and with size equal
   # to distance on sphere between 300,-30 and 330,-30 deg (=25.9)
   try:
      annim.Skypolygon("rectangle", cpos="315 deg -45 deg", major=25.9, minor=25.9,
                        pa=0.0, ec='g', fc='#ff33dd', alpha=0.8)
   except:
      print "Failed to plot square"
   # Regular polygon with 6 angles at some position in galactic coordinates
   try:
      annim.Skypolygon("npoly", cpos="ga 102d11m35.239s ga 59d50m25.734", 
                        major=20, nangles=6,
                        ec='g', fc='y', alpha=0.3)
   except:
      print "Failed to plot regular polygon"
   # Regular polygon as a triangle
   try:
      annim.Skypolygon("npolygon", cpos="ga 0 ga 90", 
                        major=70, nangles=3,
                        ec='g', fc='c', alpha=0.7)
   except:
      print "Failed to plot triangle"
   # Set of (absolute) coordinates, no prescription
   lons = [270, 240, 240, 270]
   lats = [-60, -60, -30, -30]
   try:
      annim.Skypolygon(prescription=None, lons=lons, lats=lats, fc='r', alpha=0.9)
   except:
      print "Failed to plot set of coordinates as polygon"

   grat.Insidelabels(wcsaxis=0,
                     world=range(0,360,30), constval=0, fmt='Hms', 
                     color='b', fontsize=7)
   grat.Insidelabels(wcsaxis=1,
                     world=[-60, -30, 30, 60], constval=0, fmt='Dms', 
                     color='b', fontsize=7)
   annim.interact_toolbarinfo()
   annim.interact_writepos(wcsfmt="%f",zfmt=None, pixfmt=None, hmsdms=False)
   frame.set_title(proj, y=0.8)
   annim.plot()

fig = plt.figure()
fig.subplots_adjust(left=0.03, bottom=0.05, right=0.97,
                    top=0.97, wspace=0.02, hspace=0.02)
shapes("AIT", fig, 1)
shapes("CAR", fig, 2)
shapes("BON", fig, 3, PV2_1=45)
shapes("PCO", fig, 4)
plt.show()