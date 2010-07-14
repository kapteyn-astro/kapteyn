from kapteyn import maputils
from matplotlib import pyplot as plt

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
pos = "300 300"
print "Is " + pos + " inside?", annim.inside(pos=pos) 
# Mode has no effect on parameter pos
print "Is " + pos + " inside?", annim.inside(pos=pos, mode='p') 
print "Is " + pos + " inside?", annim.inside(pos=pos, mode='w') 

print "Is 300, 300 inside?", annim.inside(x=300, y=300, mode='p') 
print "Is 300, 300 inside?", annim.inside(x=300, y=300, mode='w') 

print "Is 300, 300, 20,200 inside?", annim.inside(x=[300,20], y=[300,200], 
                                                  mode='p') 
crval1 = annim.projection.crval[0]
crval2 = annim.projection.crval[1]

print "Is %f %f" % (crval1, crval2) + " inside?", annim.inside(x=crval1, 
                                                 y=crval2, mode='w')

pos = '{} ' + str(crval1) + ' {} ' + str(crval2)
print "Is " + pos + " inside?", annim.inside(pos=pos)

# Raise an exception
print "Is 300, 300 inside?", annim.inside(x=300, y=300) 