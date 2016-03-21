#!/usr/bin/env python
import numpy
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import fmin


def chi2(pars, x, ydata):
   a = pars[0]
   b = pars[1]
   d = b*x + a - ydata
   d *= d
   return d.sum()

def chi2X(pars, x, ydata):
   f = pars[0]
   g = pars[1]
   d = g*ydata + f - x
   d *= d
   return d.sum()

def chi2pp(pars, x, ydata):
   a = pars[0]
   b = pars[1]
   p2 = b*b + 1.0
   p1 = ydata - a - b*x
   d2 = p1 * p1
   return d2.sum() / p2

def chi2robust(pars, x, ydata):
   a = pars[0]
   b = pars[1]
   d = abs(b*x + a - ydata)
   return d.sum()


# Create 'real' data. i.e. use known parameters for straight line and add noise
sc1, sc2 = input('Enter scaling factors random noise as s1,s2: ')
no = input('Number of real data points: ')
A,B = input('Enter offset a and slope b as a,b: ')

x = numpy.linspace(0.0, 10.0, no)
nsey = sc1 * numpy.random.randn(len(x))
ydata = A + B*x + nsey
nsex = sc2 * numpy.random.randn(len(x))
x += nsex

s = raw_input("Add outlier x, y or press return to skip: ")
if s:
  ox, oy = eval(s)
  x[-1] = ox
  ydata[-1] = oy

# The line is written as Y = a + bX. We want to plot the chi2 landscape
# for a range of values of a and b.

nx = 100
ny = 100
aa = numpy.linspace(A-5,A+5,nx)
bb = numpy.linspace(B-3,B+3,ny)
Z = numpy.zeros( (ny,nx) )
Z2 = numpy.zeros( (ny,nx) )
Z3 = numpy.zeros( (ny,nx) )

# Get the Chi^2 landscape. Un-optimized code
i = -1
for a in aa:
   i += 1
   j = -1
   for b in bb:
      j += 1
      Z[j,i] = chi2( [a,b], x, ydata ) / (no-2)  # Get reduced chi2
      Z2[j,i] = chi2pp( [a,b], x, ydata ) / (no-2)
      Z3[j,i] = chi2robust( [a,b], x, ydata ) / (no-2)


print "Minimum in reduced chi^2 landscape for standard merit function:",Z.min(), Z.max()
print "Minimum in reduced chi^2 landscape for perpendicular merit function:",Z2.min(), Z2.flatten().max()
print "Minimum in reduced chi^2 landscape for robust merit function:",Z3.min(), Z3.max()

XY = numpy.meshgrid( aa, bb )

XY0 = [0,0]
(m1,fmi,p2,p3,p4) = fmin( chi2, XY0, args=(x,ydata), full_output=1, retall=0 )
print "\nFit deviations in Y:", m1

XY0 = [0,0]
m2 = fmin( chi2pp, XY0, args=(x,ydata) )
print "\nFit orthogonal deviations:", m2

XY0 = [0,0]
m3 = fmin( chi2X, XY0, args=(x,ydata) )
bm4 = 1/m3[1]; am4 = -m3[0]/m3[1]
print "\nFit deviations in X: ", am4, bm4

XY0 = [0,0]
m_rob = fmin(chi2robust, XY0, args=(x,ydata) )
print "\nFit absolute deviations in Y:", m_rob

# Plotting
fig1 = figure(1)
frame = fig1.add_subplot(111, projection='3d', azim=-41, elev=21)
frame.plot_surface( XY[0], XY[1], Z3, color='g', alpha=0.8)
frame.set_xlabel('a (offset)')
frame.set_ylabel('b (slope)')
frame.set_zlabel('$Z=\\chi^2_{\\nu}$')
fig1.savefig('lsqfit.chi2landscape.png')

V = [1.0, 0.1, 0.5, 1.5, 2.0, 5, 10, 15, 20]
fig2 = figure(2)
frame2 = fig2.add_subplot(2,2,2)
cs = frame2.contour(XY[0], XY[1], Z, V)
zc = cs.collections[0]
zc.set_color('red')
zc.set_linewidth(2)
frame2.clabel(cs, V, inline=False, fmt='%1.1f', fontsize=10, color='k')
frame2.plot((A,),(B,), 'bo')
frame2.plot((m1[0],),(m1[1],), 'ro')
frame2.plot((m2[0],),(m2[1],), 'go')
frame2.plot((am4,),(bm4,), 'mo')
frame2.plot( (m_rob[0],),(m_rob[1],), 'co')
frame2.set_xlabel( 'a (offset)')
frame2.set_ylabel( 'b (slope)')
frame2.set_title('Standard Chi2 contours')
frame2.set_title('$\\chi^2_{\\nu}\, \\rm{deviates\, in\, Y\, only}$')

frame3 = fig2.add_subplot(2,2,3)
cs = frame3.contour(XY[0], XY[1], Z2, V)
zc = cs.collections[0]
zc.set_color('red')
zc.set_linewidth(2)
frame3.clabel(cs, V, inline=False, fmt='%1.1f', fontsize=10)
frame3.plot((A,),(B,), 'bo')
frame3.plot((m1[0],),(m1[1],), 'ro')
frame3.plot((m2[0],),(m2[1],), 'go')
frame3.plot((am4,),(bm4,), 'mo')
frame3.plot((m_rob[0],),(m_rob[1],), 'co')
frame3.set_xlabel('a (offset)')
frame3.set_ylabel('b (slope)')
frame3.set_title('$\\chi^2_{\\nu}\, \\rm{Orthogonal\, deviates}$')

frame4 = fig2.add_subplot(2,2,4)
cs = frame4.contour(XY[0], XY[1], Z3, V)
zc = cs.collections[0]
zc.set_color('red')
zc.set_linewidth(2)
frame4.clabel(cs, V, inline=False, fmt='%1.1f', fontsize=10)
frame4.plot((A,),(B,), 'bo')
frame4.plot((m1[0],),(m1[1],), 'ro')
frame4.plot((m2[0],),(m2[1],), 'go')
frame4.plot((am4,),(bm4,), 'mo')
frame4.plot((m_rob[0],),(m_rob[1],), 'co')
frame4.set_xlabel('a (offset)')
frame4.set_ylabel('b (slope)')
frame4.set_title('$\\chi^2_{\\nu}\, \\rm{Absolute\, deviates}$')

frame1 = fig2.add_subplot(2,2,1)
frame1.set_aspect('equal', 'datalim')
frame1.plot(x, ydata, 'ob', label='Observed data')
frame1.plot(x, A + B*x, 'b', label='True model')
frame1.plot(x, m1[1]*x+m1[0], 'r', label='Deviations in Y only')
frame1.plot(x, m2[1]*x+m2[0], 'g', label='Perpendicular distance')
frame1.plot(x, bm4*x+am4, 'm', label='Deviations in X only' )
frame1.plot(x, m_rob[1]*x+m_rob[0], 'c', label='Absolute deviations')
leg = frame1.legend(loc='upper left')
for txt in leg.get_texts():  # Make font size smaller
   txt.set_fontsize(9)
frame1.set_xlabel('X')
frame1.set_ylabel('Y')
t = '$\\rm{Real\, data\, and\, fitted\, parameters:}\, Y = %.2f + %.2f*X$' % (A,B)
frame1.set_title(t)
fig2.savefig('merit_functions.png')


# Goodness of fit
from scipy.special import gammainc
N = no
M = 2
degfreedom = N - M
Q = gammainc( 0.5*degfreedom, 0.5*fmi )
print "deg of freedom: ", degfreedom
print "Chi2 min: ", fmi
print "Goodness of fit: gammainc(%f,%f) = %f" % (0.5*degfreedom, 0.5*fmi, Q)

show()
