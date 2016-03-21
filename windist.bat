::  Batch file to build a binary Windows installer.
::
::  It should be invoked in the directory where it resides and it will
::  put the result in the subdirectory 'dist'. In order to run successfully,
::  it needs a C compiler and the header file 'string.h' should not
::  declare the depracated POSIX function wcsset().
::
C:\Python26\python.exe setup.py bdist_wininst --bitmap wininstlogo.bmp
pause

