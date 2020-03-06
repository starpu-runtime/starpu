#!/usr/bin/python
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

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

