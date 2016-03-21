void to_xyz(double *world, double *xyz, int n, int ndims,
            int lonindex, int latindex);
void from_xyz(double *world, double *xyz, int n, int ndims,
              int lonindex, int latindex);
void flag_invalid(double *world, int n, int ndims, int *stat, double flag);
