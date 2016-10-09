from kapteyn import maputils
from matplotlib import pyplot as plt
from numpy import array as narray

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((200,400), (200,400))

annim = fitsobj.Annotatedimage()
pos = "300 300"
print("Is " + pos + " inside?", annim.inside(pos=pos)) 
# Mode has no effect on parameter pos
print("Is " + pos + " inside?", annim.inside(pos=pos, mode='p')) 
print("Is " + pos + " inside?", annim.inside(pos=pos, mode='w')) 

print("Is 300, 300 inside?", annim.inside(x=300, y=300, mode='p')) 
print("Is 300, 300 inside?", annim.inside(x=300, y=300, mode='w')) 

print("Is 300, 300, 20,200 inside?", annim.inside(x=[300,20], y=[300,200], 
                                                  mode='p')) 
x = narray([100,200,300,20])
y = narray([100,200,270,0])

print("Two numpy arrays inside?", annim.inside(x=x, y=y, mode='p')) 

crval1 = annim.projection.crval[0]
crval2 = annim.projection.crval[1]

print("Is %f %f" % (crval1, crval2) + " inside?", annim.inside(x=crval1, 
                                                 y=crval2, mode='w'))

pos = '{} ' + str(crval1) + ' {} ' + str(crval2)
print("Is " + pos + " inside?", annim.inside(pos=pos))

# Raise an exception
try:
   print("Is 300, 300 inside?", annim.inside(x=300, y=300)) 
except Exception as message:
   print("Fails:", message)

try:
   print("x=300 y=None", annim.inside(x=300, y=None, mode='p')) 
except Exception as message:
   print("Fails:", message)

try:
   print("x=300 y=300 pos='10,20'", annim.inside(x=300, y=300, pos='10 20', 
                                    mode='p')) 
except Exception as message:
   print("Fails:", message)

try:
   print("pos=(10,10)", annim.inside(pos=(10,10))) 
except Exception as message:
   print("Fails:", message)
