/*
 *  Author: J. P. Terlouw
 */
#include <stdlib.h>
#include <math.h>
#include "xyz.h"

#define LATLIM 89.9999999999

static double d2r = 3.1415926535897931/180.0;

void to_xyz(double *world, double *xyz, int n, int ndims,
            int lonindex, int latindex)
{
   double lon, lat, sinlon, coslon, coslat;
   int i;

   for (i=0; i<n; i++) {
      lon = d2r*world[ndims*i+lonindex];
      lat = d2r*world[ndims*i+latindex];
      sinlon = sin(lon);
      coslon = cos(lon);
      coslat = cos(lat);
      xyz[3*i]   = coslon*coslat;
      xyz[3*i+1] = sinlon*coslat;
      xyz[3*i+2] = sin(lat);
   }
}

void from_xyz(double *world, double *xyz, int n, int ndims,
              int lonindex, int latindex)
{
   double lon, lat, x, y, z, r2d;
   int i;

   r2d = 1.0/d2r;

   for (i=0; i<n; i++) {
      x = xyz[3*i];
      y = xyz[3*i+1];
      z = xyz[3*i+2];
      lat = r2d*atan2(z, sqrt(x*x+y*y));
      lon = fabs(lat)>LATLIM ? 0.0 : r2d*atan2(y,x);
      lon = lon<0.0 ? lon+360.0 : lon;
      world[ndims*i+lonindex] = lon>=360.0 ? lon-360.0 : lon;
      world[ndims*i+latindex] = lat;
   }
}

void flag_invalid(double *world, int n, int ndims, int *stat, double flag)
{
   int i, j;
      
   for (i=0; i<n; i++) {
      if (stat[i]) {
         register int offset=ndims*i;
         for (j=0; j<ndims; j++) {
            world[offset+j] = flag;
         }
      }
   }
}
