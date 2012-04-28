#!/usr/bin/env python
#
# Generate BibTeX entry from template file.
#
import time
from kapteyn import __version__ as version

fi = open('kapteynbib.in')

entry = fi.read()
fi.close()

now   = time.asctime().split()
year  = now[-1]
month = now[1].lower()

entry = entry % (version, year, month)

fo = open('kapteyn.bib', 'w')
fo.write(entry)
fo.close()
