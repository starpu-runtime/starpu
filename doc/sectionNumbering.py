#!/usr/bin/python3
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import operator
import sys

files = {}

with open(sys.argv[1]+"/doxygen-config.cfg", "r", encoding="utf-8") as fin:
    for line in fin.readlines():
        if ".doxy" in line:
            for x in line.split(" "):
                if ".doxy" in x:
                    with open(x, "r", encoding="utf-8") as fin:
                        for line in fin.readlines():
                            if "\page" in line:
                                line = line.replace("/*! \page ", "").strip()
                                files[x] = line[0:line.index(" ")]+".html"

htmlfiles = ["index.html"]
htmlfiles.extend(files.values())

htmldir=sys.argv[2]+"/"

chapter=0
for x in htmlfiles:
    chapter+=1
    section=0
    with open(htmldir+x, "r", encoding="utf-8") as fin:
        with open(htmldir+x+".count.html", "w", encoding="utf-8") as fout:
            for line in fin.readlines():
                if "<div class=\"title\">" in line:
                    line = line.replace("<div class=\"title\">", "<div class=\"title\">"+str(chapter)+". ")
                if "<h1>" in line:
                    section += 1
                    line = line.replace("<h1>", "<h1>" + str(chapter) + "." + str(section))
                    subsection = 0
                if "<h2>" in line:
                    subsection += 1
                    line = line.replace("<h2>", "<h2>" + str(chapter) + "." + str(section) + "." + str(subsection))
                    subsubsection = 0
                if "<h3>" in line:
                    subsubsection += 1
                    line = line.replace("<h3>", "<h3>" + str(chapter) + "." + str(section) + "." + str(subsection) + "." + str(subsubsection))
                fout.write(line)
    os.rename(htmldir+x+".count.html", htmldir+x)
