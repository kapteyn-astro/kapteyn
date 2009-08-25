#-----------------------------------------------------------------------------
#
#  Makefile used for development, NOT for installation.
#
#-----------------------------------------------------------------------------
SHELL = /bin/sh

default:: wcs.so

wcs.so : wcs.o xyz.o eterms.o
	cc -shared -g wcs.o xyz.o eterms.o -lwcs -o wcs.so

wcs.o : wcs.c
	cc -c -g -fPIC wcs.c `python incdir.py`

wcs.c : wcs.pyx c_wcs.pxd
	cython wcs.pyx
