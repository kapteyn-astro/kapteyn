#!/bin/env python
from pylab import *
from numpy import *
from matplotlib.patches import Ellipse, Wedge
rc('text', usetex=True)

deg2rad = pi/180.0
rad2deg = 180.0/pi

def Cos(x):
   return cos(pi*x/180.0)

def Sin(x):
   return sin(pi*x/180.0)

def findendpoint(m, b, Vlen, x1, y1):
   A = m*m+1
   B = -2*x1 - 2*y1*m + 2*m*b
   C = -(Vlen*Vlen - x1*x1 - y1*y1 + 2*y1*b - b*b)
   x2 = (-B - sqrt(B*B-4*A*C))/(2*A)
   y2 = m*x2 + b
   x3 = (-B + sqrt(B*B-4*A*C))/(2*A)
   y3 = m*x3 + b
   return x2, y2, x3, y3

axis('equal')
#axis([-6,6,-6,6])
axis('off')
ax = gca()
ex = 0.0; ey = 0.0

# Set properties ellipse and draw it
major = 10.0; minor= 9.0
e1 = Ellipse((0,0), major, minor, 0, fill=False)
ax.add_patch(e1)

# Plot focus positions
# Position of focus = c^2 = a^2 - b^2
a = 0.5*major; b = 0.5*minor
c = sqrt(a*a-b*b)
x = [c]; y= [0]
plot(x,y, 'ro')
text(c, -0.1*minor, '(c,0)', horizontalalignment='center')
text(c, 0.02*minor, 'S', horizontalalignment='center')
plot([-c],y, 'ro')
text(-c, -0.1*minor, '(-c,0)', horizontalalignment='center')

# Draw minor and major axis
X1 = -a
Y1 = 0
X2 = a
Y2 = 0
plot([X1,X2], [Y1,Y2], 'g--')
Y1 = -b
X1 = 0
Y2 = b
X2 = 0
plot([X1,X2], [Y1,Y2], 'g--')


# Select position of object (e.g. earth)
theta = 40.0
x1 = a * Cos(theta)
y1 = b * Sin(theta)
#aw = 0.04
#a1 = Arrow(c, 0.0, x1-c,y1, width=0.2)
#ax.add_patch(a1)
plot([x1],[y1], 'bo')
text(x1, y1-0.03*minor, 'P', verticalalignment='top')
angle = arctan2(y1, x1-c)*rad2deg
radius = 0.07*major
w1 = Wedge((c,0), radius, 0, angle, fill=False)
ax.add_patch(w1)
angle /= 2.0
xa = 1.2*radius * Cos(angle)
ya = 1.2*radius * Sin(angle)
text(c+xa, ya, r'$\alpha$') 



# Draw the line connecting the focal points with object on ellipse
m1 = y1/(x1+c)
m2 = y1/(x1-c)
X1 = -c; Y1 = 0
lam = 1.0
X2 = -c + lam*(x1+c)
Y2 = 0 + lam*y1
plot([X1,X2], [Y1,Y2], 'b--')
X1 = c; Y1 = 0
lam = 1.0
X2 = c + lam*(x1-c)
Y2 = 0 + lam*y1
plot([X1,X2], [Y1,Y2], 'b--')

# Find the bisector that is tangent to the ellipse, not crossing
a1 = y1/(x1+c); a2 = -1; k1 =  y1*c/(x1+c)
b1 = y1/(x1-c); b2 = -1; k2 = -y1*c/(x1-c)
h=sqrt(b1*b1+b2*b2)
k=sqrt(a1*a1+a2*a2)
A1 = h*a1-k*b1; A2 = h*a2 - k*b2; A3 = h*k1 - k*k2
m = -A1/A2; off = -A3/A2
X1 = x1-3
Y1 = m*X1 + off
X2 = x1+2
Y2 = m*X2 + off
plot([X1,X2], [Y1,Y2], 'r--')



# Draw velocity vector along tangent at x1,y1
Vlen = 3.0
b = off
x2, y2, x3,y3 = findendpoint(m, b, Vlen, x1, y1)
arr1 = Arrow( x1, y1, x2-x1,y2-y1, width=0.2, facecolor='red' )
ax.add_patch(arr1)
text(x2, y2, r'\bf{V}', horizontalalignment='left', verticalalignment='bottom')
xdotted1 = x2; ydotted1 = y2

# Draw vector for radial velocity i.e. perpendicular to radius vector
# i.e. r.dtheta/dt
phi1 = arctan2( y1,x1-c)
phi2 = arctan2(y2-y1, x2-x1)
phi3 = phi1 + pi/2 - phi2
m2 = tan(phi1+pi/2)
b2 = y1 - m2*x1
PL = Vlen*cos(phi3)
x2, y2, x3, y3 = findendpoint(m2, b2, PL, x1, y1)
arr2 = Arrow( x1, y1, x2-x1,y2-y1, width=0.2, edgecolor='g' )
ax.add_patch(arr2)
text(x2, y2-0.03*minor, 'L', verticalalignment='top')
xdotted2 = x2; ydotted2 = y2

# Draw velocity component dr/dt as an extension to the radius vector
m3 = tan(phi1)
b3 = y1 - m3*x1
PR = Vlen*sin(phi3)
x2, y2, x3, y3 = findendpoint(m3, b3, PR, x1, y1)
arr3 = Arrow(x1, y1, x3-x1,y3-y1, width=0.2, edgecolor='g')
ax.add_patch(arr3)
text(x3, y3, 'R', horizontalalignment='left')
xdotted3 = x3; ydotted3 = y3
plot([xdotted2, xdotted1,xdotted3], [ydotted2, ydotted1,ydotted3], ':')

# Draw velocity component perpendicular to major axis
PQ = PR / sin(phi1) 
arr4 = Arrow( x1, y1, 0, PQ, width=0.2, facecolor='y' )
ax.add_patch(arr4)
text(x1, y1+PQ, r'$Q$', verticalalignment='bottom')

# Draw velocity component perpendicular to radius vector
PL2 = PL - PR/tan(phi1)
x2, y2, x3, y3 = findendpoint(m2, b2, PL2, x1, y1)
arr5 = Arrow(x1, y1, x2-x1,y2-y1, width=0.2, facecolor='y')
ax.add_patch(arr5)
text(x2, y2-0.03*minor, r'$L_1$', verticalalignment='top')

# The other bisector
"""
A1 = h*a1+k*b1; A2 = h*a2 + k*b2; A3 = h*k1 + k*k2
m = -A1/A2; off = -A3/A2
X1 = x1 - 3
Y1 = m*X1 + off
X2 = x1 + 3
Y2 = m*X2 + off
plot( [X1,X2], [Y1,Y2])
"""

savefig("etermfig.png")
show()
