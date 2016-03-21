from scipy.stats import ksone
from math import sqrt

alphas=(0.1, 0.05, 0.025, 0.01, 0.005)
header= "\n%5s"%""
header1 = "%5s"%"n"
for alpha in alphas:
   s = "a1=%.3f"%alpha
   header += " %12s"%s
   s = "a2=%.3f"%(2*alpha)
   header1 += " %12s"%s
print header
print header1
print "="*len(header)

for n in range(3,21):
   s = "%5d"%n
   rv = ksone(n)
   for alpha in alphas:
      Dcrit = rv.ppf(1-alpha)   # Lower tail probability
      s += " %12.5f"%(Dcrit)
   print s