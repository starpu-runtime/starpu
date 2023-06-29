#!/usr/bin/env python3
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import os, sys

pathname = os.path.abspath(os.path.dirname(__file__)) + "/../../../"
pathname = "./"

if len(sys.argv) > 1:
    pathname += sys.argv[1]

def addFiles(pathname,files):
    onlyfiles = [os.path.join(pathname, f) for f in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, f))]
    for d in [os.path.join(pathname, d) for d in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, d))]:
        addFiles(d, files)
    files.extend(onlyfiles)

files = []
addFiles(pathname, files)

doxyFiles=[f for f in files if f.endswith(".doxy")]
cFiles=[f for f in files if f.endswith(".c") and not f.startswith("./build") and f.find("min-dgels")==-1 and f.find("experimental")==-1 and f.find("julia")==-1 and f.find("SobolQRNG")==-1 and f.find("socl/src/CL")==-1 and f.find("doc/")==-1]
hFiles=[f for f in files if f.endswith(".h") and not f.startswith("./build") and f.find("min-dgels")==-1 and f.find("experimental")==-1 and f.find("julia")==-1 and f.find("SobolQRNG")==-1 and f.find("socl/src/CL")==-1 and f.find("doc/")==-1]

for l in [cFiles]:#, hFiles, doxyFiles]:
    for f in l:
        f_p = 1
        c=0
        for line in open(f, "r").readlines():
            c+=1
            #            if line.startswith(" ") and not line.startswith(" *") and line.find("@")==-1 and line.find(";")!=-1:
            #                print(f)
            #                print(str(c)+line)
            #                break
            #            if line[len(line)-2] == " ":
            #                print(f)
            #                #print(str(c)+line)
            #                break
            if line.find(" (") != -1 and line.find("Copyright") == -1 and line.find("Free") == -1 and line.find("= (") == -1 and line.find("if (") == -1 and line.find("for (") == -1 and line.find(", (") == -1 and line.find("while (") == -1 and line.find("return (") == -1 and line.find("switch (") == -1:
                if (f_p):
                    print(f)
                    f_p = 0
                print(c,line)

#print(cFiles)
# with emacs select region and call M-x tabify
