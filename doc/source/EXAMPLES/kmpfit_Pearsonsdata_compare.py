#!/usr/bin/env python
#------------------------------------------------------------
# Purpose: Program to straight line parameters
#          to data with errors in both coordinates.
#          Use the (famous) data set from Pearson with
#          weights from York
# Vog, 12 Dec, 2011
#
# The data for x and y are from Pearson
# Pearson, K. 1901. On lines and planes of closest fit to systems 
# of points in space. Philosophical Magazine 2:559-572
# Copy of this article can be found at:
# stat.smmu.edu.cn/history/pearson1901.pdf
#
# Pearson's best fit through (3.82,3.70) ->
# a=5.7857  b=-0.546
# York added weights in 
# York, D. Can. J. Phys. 1968, 44, 1079-1086
# The Williamson Approach is also implemented.
# The steps are described in Ogren, Paul J., Norton, J. Russel
# Applying a simple Least-Squares Algorithm to Data
# with Uncertainties in Both Variables,
# J. of Chem. Education, Vol 69, Number 4, April 1992
# Best fit parameters for this method are:
# a=5.47991022403  b=-0.48053340745
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from kapteyn import kmpfit


def model(p, x):
   a, b = p
   return a + b*x

def residuals(p, data):
   # Residuals function for data with errors in both coordinates
   a, b = p
   x, y, ex, ey = data
   w = ey*ey + ex*ex*b*b
   wi = numpy.sqrt(numpy.where(w==0.0, 0.0, 1.0/(w)))
   d = wi*(y-model(p,x))
   return d

def residuals2(p, data):
   # Residuals function for data with errors in y only
   a, b = p
   x, y, ey = data
   wi = numpy.where(ey==0.0, 0.0, 1.0/ey)
   d = wi*(y-model(p,x))
   return d


# Pearsons data with York's weights 
x = numpy.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
y = numpy.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
wx = numpy.array([1000.0,1000,500,800,200,80,60,20,1.8,1.0])
wy = numpy.array([1,1.8,4,8,20,20,70,70,100,500])
errx = 1/numpy.sqrt(wx)  # We need the errors in the residuals functions
erry = 1/numpy.sqrt(wy)
N = len(x)


# Prepare fit routine
beta0 = [5.0, 1.0]         # Initial estimates
fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
fitobj.fit(params0=beta0)
print("\n\n======== Results kmpfit: weights for both coordinates =========")
print("Fitted parameters:      ", fitobj.params)
print("Covariance errors:      ", fitobj.xerror)
print("Standard errors         ", fitobj.stderr)
print("Chi^2 min:              ", fitobj.chi2_min)
print("Reduced Chi^2:          ", fitobj.rchi2_min)
print("Iterations:             ", fitobj.niter)

# Prepare fit routine
fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, erry))
fitobj2.fit(params0=beta0)
print("\n\n======== Results kmpfit errors in Y only =========")
print("Fitted parameters:      ", fitobj2.params)
print("Covariance errors:      ", fitobj2.xerror)
print("Standard errors         ", fitobj2.stderr)
print("Chi^2 min:              ", fitobj2.chi2_min)
print("Reduced Chi^2:          ", fitobj2.rchi2_min)


# Unweighted (unit weighting)
fitobj3 = kmpfit.Fitter(residuals=residuals2, data=(x, y, numpy.ones(N)))
fitobj3.fit(params0=beta0)
print("\n\n======== Results kmpfit unit weighting =========")
print("Fitted parameters:      ", fitobj3.params)
print("Covariance errors:      ", fitobj3.xerror)
print("Standard errors         ", fitobj3.stderr)
print("Chi^2 min:              ", fitobj3.chi2_min)
print("Reduced Chi^2:          ", fitobj3.rchi2_min)


from scipy.odr import Data, Model, ODR, RealData, odr_stop
# Compare result with ODR
linear = Model(model)
mydata = RealData(x, y, sx=errx, sy=erry)
myodr = ODR(mydata, linear, beta0=beta0, maxit=5000, sstol=1e-14)
myoutput = myodr.run()
print("\n\n======== Results ODR =========")
print("Fitted parameters:      ", myoutput.beta)
print("Covariance errors:      ", numpy.sqrt(myoutput.cov_beta.diagonal()))
print("Standard errors:        ", myoutput.sd_beta)
print("Minimum chi^2:          ", myoutput.sum_square)
print("Minimum (reduced)chi^2: ", myoutput.res_var)
beta = myoutput.beta


a,b = beta0
a_y = a; b_y = b
ui = errx**2
vi = erry**2

n = 0 
cont = True
while cont:
   # Step 2: Use this slope to find weighting for each point
   wi = (vi+b*b*ui)**-1

   # Step 3: Calcu;ate weighted avarages
   w_sum = wi.sum()
   x_av = (wi*x).sum() / w_sum
   x_diff = x - x_av
   y_av = (wi*y).sum() / w_sum
   y_diff = y - y_av

   # Step 4: Calculate the 'improvement' vector zi
   zi = wi*(vi*x_diff + b*ui*y_diff)
   b_will = (wi*zi*y_diff).sum()/ (wi*zi*x_diff).sum()
   cont = abs(b-b_will) > 1e-12 and n < 100
   n += 1
   b = b_will

print("average x weighted, unweighted:", x_av, x.mean())
# Step 5: Repeat steps 2-4 until convergence

# Step 6: Calculate 'a' using the averages of a and y
a_will = y_av - b_will*x_av 

# Step 7: The variances
wi = (vi+b_will*b_will*ui)**-1
w_sum = wi.sum()
z_av = (wi*zi).sum() / w_sum
zi2 = zi - z_av
Q =1.0/(wi*(x_diff*y_diff/b_will + 4*zi2*(zi-x_diff))).sum()
sigb2 = Q*Q * (wi*wi*(x_diff**2*vi+y_diff**2*ui)).sum()
siga2 = 1.0/w_sum + 2*(x_av+2*z_av)*z_av*Q + (x_av+2*z_av)**2*sigb2
siga = numpy.sqrt(siga2)
sigb = numpy.sqrt(sigb2)
chi2 = (residuals(fitobj.params,(x,y,errx,erry))**2).sum()
print("\n\n======== Results Williamson =========")
print("Fitted parameters:      ", [a_will, b_will])
print("Covariance errors:      ", [siga, sigb])
print("Minimum chi^2:          ", chi2)


print("\n\nReference                    a                b")
print("-----------------------------------------------------------")
print("Literature results:")
print("Pearson unweighted           5.7857           -0.546")
print("Williamson                   5.47991022403    -0.48053340745")
print("Reed                         5.47991022723    -0.48053340810")
print("Lybanon                      5.47991025       -0.480533415")
print() 
print("Practical results:")
print("kmpfit unweighted            %13.11f    %13.11f"%(fitobj3.params[0],fitobj3.params[1]))
print("kmpfit weights in Y only     %13.11f    %13.11f"%(fitobj2.params[0],fitobj2.params[1]))
print("kmpfit effective variance    %13.11f    %13.11f"%(fitobj.params[0],fitobj.params[1]))
print("ODR                          %13.11f    %13.11f"%(beta[0],beta[1]))
print("Williamson                   %13.11f    %13.11f"%(a_will,b_will))
print() 



# Some plotting
rc('font', size=9)
rc('legend', fontsize=8)
fig = figure(1)
d = (x.max() - x.min())/10
X = numpy.linspace(x.min()-d, x.max()+d, 50)
frame = fig.add_subplot(1,1,1, aspect=1, adjustable='datalim')
frame.errorbar(x, y, xerr=errx, yerr=erry,  fmt='bo')
frame.plot(X, model(fitobj.params,X), 'c', ls='--', lw=2, label="kmpfit effective variance")
frame.plot(X, model(fitobj2.params,X), 'g', label="kmpfit errors in y only")
frame.plot(X, model(fitobj3.params,X), 'r', label="kmpfit unweighted")
frame.plot(X, model((5.7857,-0.546),X), 'b', label="Pearson's values")
frame.plot(X, model((5.463,-0.477),X), 'm', label="York's values")
frame.set_xlabel("X")
frame.set_ylabel("Y")
frame.set_title("Pearson's data with York's weights")
frame.grid(True)
leg = frame.legend(loc=1)
show()