#!/usr/bin/python

import os

class bcolors:
    FAILURE = '\033[91m'
    NORMAL = '\033[0m'

def loadFunctionsAndDatatypes(flist, dtlist, fname):
    f = open(fname, 'r')
    for line in f:
        mline = line[:-1]
        if mline.count("\\fn"):
            if mline.count("fft") == 0:
                func = mline.replace("\\fn ", "")
                flist.append(list([func, fname]))
        if mline.count("\\struct ") or mline.count("\\def ") or mline.count("\\typedef ") or mline.count("\\enum "):
            datatype = mline.replace("\\struct ", "").replace("\\def ", "").replace("\\typedef ", "").replace("\\enum ","")
            dtlist.append(list([datatype, fname]))
    f.close()

functions = []
datatypes = []

for docfile in os.listdir('chapters/api'):
    if docfile.count(".doxy"):
        loadFunctionsAndDatatypes(functions, datatypes, "chapters/api/"+docfile)

for function in functions:
    x = os.system("fgrep -l \"" + function[0] + "\" ../../include/*.h ../../mpi/include/*.h ../../starpufft/*h ../../sc_hypervisor/include/*.h > /dev/null")
    if x != 0:
        print "Function <" + bcolors.FAILURE + function[0] + bcolors.NORMAL + "> documented in <" + function[1] + "> does not exist in StarPU's API"

for datatype in datatypes:
    x = os.system("fgrep -l \"" + datatype[0] + "\" ../../include/*.h ../../mpi/include/*.h ../../starpufft/*h ../../sc_hypervisor/include/*.h > /dev/null")
    if x != 0:
        print "Datatype <" + bcolors.FAILURE + datatype[0] + bcolors.NORMAL + "> documented in <" + datatype[1] + "> does not exist in StarPU's API"

