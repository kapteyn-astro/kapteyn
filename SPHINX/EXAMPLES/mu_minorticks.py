from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fig = plt.figure()
fig.subplots_adjust(left=0.18, bottom=0.10, right=0.90,
                    top=0.90, wspace=0.95, hspace=0.20)
for i in range(4):   
   f = fig.add_subplot(2,2,i+1)
   mplim = fitsobj.Annotatedimage(f)
   if i == 0:
      majorgrat = mplim.Graticule()
      majorgrat.setp_gratline(visible=False)
   elif i == 1:
      majorgrat = mplim.Graticule(offsetx=True, unitsx='ARCMIN')
      majorgrat.setp_gratline(visible=False)
   elif i == 2:
      majorgrat = mplim.Graticule(skyout='galactic', unitsx='ARCMIN')
      majorgrat.setp_gratline(color='b')
   else:
      majorgrat = mplim.Graticule(skyout='galactic', 
                         offsetx=True, unitsx='ARCMIN')
      majorgrat.setp_gratline(color='b')

   majorgrat.setp_tickmark(markersize=10)
   majorgrat.setp_ticklabel(fontsize=6)
   majorgrat.setp_plotaxis(plotaxis=[0,1], fontsize=10)
   minorgrat = mplim.Minortickmarks(majorgrat, 3, 5, 
             color="#aa44dd", markersize=3, markeredgewidth=2)

maputils.showall()
plt.show()
