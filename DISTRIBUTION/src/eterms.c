/*
 *  Author: J.P. Terlouw
 */
#include "eterms.h"
#include <math.h>

void eterms(double *xyz, int n, int direct, double A0, double A1, double A2)
{
   int i;
   double x, y, z, w;

#if 1
/*  Exact methods of E-term removal and adding.
 *  For documentation refer to the equivalent functions removeEterms and
 *  addEterms in the Python module celestial.py.
 */
   if (direct<0) {
      for (i=0; i<n; i++) {
         xyz[i*3]   -= A0;
         xyz[i*3+1] -= A1;
         xyz[i*3+2] -= A2;
      }
   } else if (direct>0) {
      double d, p, lambda1;
      for (i=0; i<n; i++) {
         x = xyz[i*3];
         y = xyz[i*3+1];
         z = xyz[i*3+2];
         d = 1.0/sqrt(x*x + y*y + z*z);
         w = 2.0 * (A0*x + A1*y + A2*z);
         p = A0*A0 + A1*A1 + A2*A2 - 1.0;
         lambda1 = (-w + sqrt(w*w-4.0*p))/2.0;
         xyz[i*3]   = lambda1*d*x + A0;
         xyz[i*3+1] = lambda1*d*y + A1;
         xyz[i*3+2] = lambda1*d*z + A2;
      }
   }
#else
/* For reference: conventional method of E-term removal and adding.
 */
   if (direct<0) {
      for (i=0; i<n; i++) {
         x = xyz[i*3];
         y = xyz[i*3+1];
         z = xyz[i*3+2];
         w = x*A0 + y*A1 + z*A2 + 1.0;
         xyz[i*3]   = (w*x - A0); 
         xyz[i*3+1] = (w*y - A1);
         xyz[i*3+2] = (w*z - A2);
      }
   } else if (direct>0) {
      for (i=0; i<n; i++) {
         x = xyz[i*3];
         y = xyz[i*3+1];
         z = xyz[i*3+2];
         w = 1.0/sqrt(x*x + y*y + z*z);
         xyz[i*3]   = (w*x + A0);
         xyz[i*3+1] = (w*y + A1);
         xyz[i*3+2] = (w*z + A2);
      }
   }
#endif
}
