#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2026   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

import sys
import imageio
import numpy as np

PROGNAME=sys.argv[0]

def usage():
    print("Turn an execution trace into a video of the accesses to a given matrix")
    print("This assumes tasks.rec and data.rec are in the current directory")
    print("")
    print("Usage: %s <matrix_name>" % PROGNAME)
    print("")
    print("Report bugs to <@PACKAGE_BUGREPORT@>")
    sys.exit(1)

if len(sys.argv) == 2:
    if sys.argv[1] == '-v' or sys.argv[1] == '--version':
        print("%s (@PACKAGE_NAME@) @PACKAGE_VERSION@" % PROGNAME)
        sys.exit(0)
    elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
        usage()
    else:
        matrix_name = sys.argv[1]
else:
    usage()

# First turn data.rec into a dictionary of the matrice tiles

matrix_handles = {}
coords = None
maxc0 = 0
maxc1 = 0

with open("data.rec", 'r') as data:
    for line in data:
        line = line.strip()
        if line == "":
            if name == matrix_name:
                matrix_handles[handle] = coords
            coords = None
        elif line.startswith("Handle: "):
            handle = int(line[8:], 16)
        elif line.startswith("Name: "):
            name = line[6:]
        elif line.startswith("Coordinates: "):
            if name == matrix_name:
                blank = line.index(' ', 13)
                c0 = int(line[13:blank])
                c1 = int(line[blank:])
                maxc0 = max(c0, maxc0)
                maxc1 = max(c1, maxc1)
                coords = (c0, c1)

# Now read tasks

handles = None
modes = None
start_time = None
end_time = None
workerid = None
priority = None
tasks = []
START = 0
END = 1
with open("tasks.rec", 'r') as data:
    for line in data:
        line = line.strip()
        if line == "":
            if handles is not None and \
               modes is not None and \
               start_time is not None and \
               end_time is not None:
                # Generate events for start and stop of the task
                tasks.append((start_time, START, handles, modes, workerid, priority))
                tasks.append((end_time, END, handles, modes, workerid, priority))
            handles = None
            modes = None
            start_time = None
            end_time = None
            workerid = None
            priority = None
        elif line.startswith("Handles: "):
            values = line[9:].split(' ')
            handles = [ int(s, 16) for s in values ]
        elif line.startswith("Modes: "):
            modes = line[7:].split(' ')
        elif line.startswith("StartTime: "):
            start_time = float(line[11:])
        elif line.startswith("EndTime: "):
            end_time = float(line[9:])
        elif line.startswith("WorkerId: "):
            workerid = int(line[10:])
        elif line.startswith("Priority: "):
            priority = int(line[10:])

# Order events by date
tasks.sort(key = lambda x: x[0])

# FIXME: make this a parameter
SCALE=8
# and create the video, remembering what is read/written
width = (maxc0+1)*SCALE
height = (maxc1+1)*SCALE
read_accesses = [ [0] * height for _ in range(width) ]
write_accesses = [ [False] * height for _ in range(width) ]
frames = []
n = 0
frame = np.zeros((width, height, 3), dtype=np.uint16)
for (time, what, handles, modes, workerid, priority) in tasks:
    n+=1
    print(f"\r{n}/{len(tasks)}", end='')
    for i in range(len(handles)):
        c0, c1 = matrix_handles[handles[i]]
        mode = modes[i]
        if what == START:
            if mode == 'R':
                read_accesses[c0][c1] += 1
            elif mode == 'RW':
                assert(write_accesses[c0][c1]) == False
                write_accesses[c0][c1] = True
            else:
                assert(False)
        elif what == END:
            if mode == 'R':
                assert(read_accesses[c0][c1]) > 0
                read_accesses[c0][c1] -= 1
            elif mode == 'RW':
                assert(write_accesses[c0][c1]) == True
                assert(read_accesses[c0][c1]) == 0
                write_accesses[c0][c1] = False
            else:
                assert(False)
    # Decay factor
    frame *= 128
    frame //= 129
    read_values = [
        [128,0,0],
        [0,128,0],
        [0,0,128],
    ]
    write_values = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
    ]
    for c1 in range(height):
        for c0 in range(width):
            #if read_accesses[c0][c1] > 0:
            #    frame[c0*SCALE:(c0+1)*SCALE, \
            #          c1*SCALE:(c1+1)*SCALE] \
            #          = read_values[workerid] # Reading
            if write_accesses[c0][c1]:
                frame[c0*SCALE:(c0+1)*SCALE, \
                      c1*SCALE:(c1+1)*SCALE] \
                      = write_values[workerid] # Writing
    frames.append(frame.astype(np.uint8))
print("")

for c0 in range(width):
    for c1 in range(height):
        assert(read_accesses[c0][c1] == 0)
        assert(write_accesses[c0][c1] == False)

imageio.mimsave("data_"+matrix_name+".mp4", frames, fps=100)
