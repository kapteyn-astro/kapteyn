#-----------------------------------------------------------------------------
#
#  Makefile used for development, NOT for installation.
#
#-----------------------------------------------------------------------------
SHELL = /bin/sh

default:: ascarray.so

ascarray.so : ascarray.o
	cc -shared -g ascarray.o -o ascarray.so

ascarray.o : ascarray.c
	cc -c -g -fPIC ascarray.c `python incdir.py`

ascarray.c : ascarray.pyx
	cython ascarray.pyx
