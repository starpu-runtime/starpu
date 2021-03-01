#!/usr/bin/env python3
# coding=utf-8
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import sys

class bcolors:
    FAILURE = '\033[91m'
    NORMAL = '\033[0m'

def list_files(directory):
    return list(map(lambda a : directory+a, list(filter(lambda a:a.count(".h") and not a.count("starpu_deprecated_api.h"),os.listdir(directory)))))

def loadFunctionsAndDatatypes(flist, dtlist, file_name):
    f = open(file_name, 'r')
    for line in f:
        mline = line[:-1]
        if mline.count("\\fn"):
            if mline.count("fft") == 0:
                func = mline.replace("\\fn ", "")
                l = func.split("(")[0].split()
                func_name = l[len(l)-1].replace("*", "")
                flist.append(list([func, func_name, file_name]))
        if mline.count("\\struct ") or mline.count("\\def ") or mline.count("\\typedef ") or mline.count("\\enum "):
            datatype = mline.replace("\\struct ", "").replace("\\def ", "").replace("\\typedef ", "").replace("\\enum ","")
            l = datatype.split("(")
            if len(l) > 1:
                datatype_name = l[0]
            else:
                datatype_name = datatype
            dtlist.append(list([datatype, datatype_name, file_name]))
    f.close()

functions = []
datatypes = []

dirname=os.path.dirname(sys.argv[0])
docfile_dir=dirname+"/../chapters/api/"

for docfile in os.listdir(docfile_dir):
    if docfile.count(".doxy"):
        loadFunctionsAndDatatypes(functions, datatypes, docfile_dir+docfile)

list_incfiles = [dirname + "/../../../include/starpu_config.h.in"]
for d in [dirname+"/../../../include/", dirname + "/../../../mpi/include/", dirname + "/../../../starpufft/include/", dirname + "/../../../sc_hypervisor/include/"]:
    list_incfiles.extend(list_files(d))
incfiles=" ".join(list_incfiles)

for function in functions:
    x = os.system("sed 's/ *STARPU_ATTRIBUTE_UNUSED *//g' " + incfiles + "| sed 's/ STARPU_WARN_UNUSED_RESULT//g' | fgrep \"" + function[0] + "\" > /dev/null")
    if x != 0:
        print("Function <" + bcolors.FAILURE + function[0] + bcolors.NORMAL + "> documented in <" + function[2] + "> does not exist in StarPU's API")
        os.system("grep " + function[1] + " " + dirname+"/../../../include/starpu_deprecated_api.h")

for datatype in datatypes:
    x = os.system("fgrep -l \"" + datatype[0] + "\" " + incfiles + " > /dev/null")
    if x != 0:
        print("Datatype <" + bcolors.FAILURE + datatype[0] + bcolors.NORMAL + "> documented in <" + datatype[2] + "> does not exist in StarPU's API")
        os.system("grep " + datatype[1] + " " + dirname+"/../../../include/starpu_deprecated_api.h")
