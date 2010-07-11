from kapteyn import maputils

fitsobj = maputils.FITSimage("m101.fits")
fitsobj.set_limits((180,344), (100,200))

print "Limits in x:", fitsobj.pxlim 
print "Limits in y:", fitsobj.pylim 

annim = fitsobj.Annotatedimage()

# We have a position in world coordinates from which we know
# it is inside the boundaries of pxlim and pylim
pos="{} 210.870170 {} 54.269001" 
print "pos, inside:", pos, annim.inside(pos=pos)
pos="ga 101.973853, ga 59.816461" 
print "pos, inside:", pos, annim.inside(pos=pos)


# Demonstrate the use of plain coordinates
x = range(180,400,40)
y = range(100,330,40)
print "x,y, inside?", zip(x,y), annim.inside(x=x, y=y, world=False)
print "world 210.870170  54.269001 inside?",\
      annim.inside(x=210.870170, y=54.269001, world=True)

while 1:
   pos = raw_input("Enter position ..... [abort]: ")
   if pos == '':
      break
   print "%s inside? %s" % (pos, annim.inside(pos=pos))
