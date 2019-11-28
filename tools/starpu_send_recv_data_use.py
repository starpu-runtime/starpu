#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019                                      INRIA
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

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

PROGNAME = sys.argv[0]


def convert_rec_file(filename):
    lines = []
    item = dict()

    with open(filename, "r") as f:
        for l in f.readlines():
            if l == "\n":
                lines.append(item)
                item = dict()
            else:
                ls = l.split(":")
                key = ls[0].lower()
                value = ls[1].strip()

                if key in item:
                    print("Warning: duplicated key '" + key + "'")
                else:
                    if re.match('^\d+$', value) != None:
                        item[key] = int(value)
                    elif re.match("^\d+\.\d+$", value) != None:
                        item[key] = float(value)
                    else:
                        item[key] = value

    return lines


def usage():
    print("Offline tool to draw graph about elapsed between sent or received data and their use by tasks")
    print("")
    print("Usage: %s <folder containing comms.rec and tasks.rec files>" % PROGNAME)


if len(sys.argv) != 2:
    print("Provide folder where *.rec files are located !")
    sys.exit(1)

working_directory = sys.argv[1]

comms = convert_rec_file(os.path.join(working_directory, "comms.rec"))
tasks = [t for t in convert_rec_file(os.path.join(working_directory, "tasks.rec")) if "control" not in t and "starttime" in t]

def plot_graph(comm_type, match, filename, title, xlabel):
    delays = []
    workers = dict()
    nb = 0
    durations = []
    min_time = 0.
    max_time = 0.

    for c in comms:
        if c["type"] == comm_type:
            t_matched = None
            for t in tasks:
                if match(t, c):
                    t_matched = t
                    break

            if t_matched is None:
                if comm_type == "recv":
                    print("No match found")
            else:
                worker = str(t_matched['mpirank']) + "-" + str(t_matched['workerid'])
                if worker not in workers:
                    workers[worker] = []

                eps = t["starttime"] - c["time"]
                assert(eps > 0)
                durations.append(eps)
                workers[worker].append((c["time"], eps))

                if min_time == 0 or c["time"] < min_time:
                    min_time = c["time"]
                if max_time == 0 or c["time"] > max_time:
                    max_time = c["time"]

                nb += 1


    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(2, 2, figure=fig)
    axs = [fig.add_subplot(gs[0, :-1]), fig.add_subplot(gs[1, :-1]), fig.add_subplot(gs[0:, -1])]
    i = 0
    for y, x in workers.items():
        # print(y, x)
        axs[0].broken_barh(x, [i*10, 8], facecolors=(0.1, 0.2, 0.5, 0.2)) 
        i += 1

    i = 0
    for y, x in workers.items(): 
        for xx in x:
            axs[1].broken_barh([xx], [i, 1])
            i += 1


    axs[0].set_yticks([i*10+4 for i in range(len(workers))])
    axs[0].set_yticklabels(list(workers))
    axs[0].set(xlabel="Time (ms) - Duration: " + str(max_time - min_time) + "ms", ylabel="Worker [mpi]-[*pu]", title=title)


    axs[2].hist(durations, bins=np.logspace(np.log10(1), np.log10(max(durations)), 50), rwidth=0.8)
    axs[2].set_xscale("log")
    axs[2].set(xlabel=xlabel, ylabel="Number of occurences", title="Histogramm")

    fig.set_size_inches(15, 9)

    plt.savefig(os.path.join(working_directory, filename), dpi=100)
    plt.show()

plot_graph("recv", lambda t, c: (t["mpirank"] == c["dst"] and t["starttime"] >= c["time"] and c["handle"] in t["handles"]), "recv_use.png", "Elapsed time between recv and use (ms)", "Time between data reception and its use by a task")
plot_graph("send", lambda t, c: (t["mpirank"] == c["src"] and t["starttime"] >= c["time"] and c["handle"] in t["handles"]), "send_use.png", "Elapsed time between send and use (ms)", "Time between data sending and its use by a task")
