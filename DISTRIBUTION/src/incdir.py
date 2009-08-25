from distutils.sysconfig import get_python_inc
import numpy
pi = get_python_inc()
ni = numpy.get_include()
print "-I%s -I%s -I%s/numpy" % (pi, ni, ni)
