for i in range(1,40):
 filename = "allskyfig%d.py" % i
 fp = open(filename,"w")
 fp.write("import allsky\n")
 fp.write("allsky.plotfig(%d)\n" % i)
 fp.close()
