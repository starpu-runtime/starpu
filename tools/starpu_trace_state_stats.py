#!/usr/bin/env python3
# coding=utf-8
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

##
# This script parses the generated trace.rec file and reports statistics about
# the number of different events/tasks and their durations. The report is
# similar to the starpu_paje_state_stats.in script, except that this one
# doesn't need R and pj_dump (from the pajeng repository), and it is also much
# faster.
##

import getopt
import os
import sys

class Event():
    def __init__(self, type, name, category, start_time):
        self._type = type
        self._name = name
        self._category = category
        self._start_time = start_time

class EventStats():
    def __init__(self, name, duration_time, category, count = 1):
        self._name = name
        self._duration_time = duration_time
        self._category = category
        self._count = count

    def aggregate(self, duration_time):
        self._duration_time += duration_time
        self._count += 1

    def show(self):
        if not self._name == None and not self._category == None:
            print("\"" + self._name + "\"," + str(self._count) + ",\"" + self._category + "\"," + str(round(self._duration_time, 6)))

class Worker():
    def __init__(self, id):
        self._id        = id
        self._events    = []
        self._stats     = []
        self._stack     = []
        self._current_state = None

    def get_event_stats(self, name):
        for stat in self._stats:
            if stat._name == name:
                return stat
        return None

    def add_event(self, type, name, category, start_time):
        self._events.append(Event(type, name, category, start_time))

    def add_event_to_stats(self, curr_event):
        if curr_event._type == "PushState":
            self._stack.append(curr_event)
            return # Will look later to find a PopState event.
        elif curr_event._type == "PopState":
            if len(self._stack) == 0:
                print("warning: PopState without a PushState, probably a trace with start/stop profiling")
                self._current_state = None
                return
            next_event = curr_event
            curr_event = self._stack.pop()
        elif curr_event._type == "SetState":
            if self._current_state == None:
                # First SetState event found
                self._current_state = curr_event
                return
            saved_state = curr_event
            next_event = curr_event
            curr_event = self._current_state
            self._current_state = saved_state
        else:
            sys.exit("ERROR: Invalid event type!")

        # Compute duration with the next event.
        a = curr_event._start_time
        b = next_event._start_time

        # Add the event to the list of stats.
        for i in range(len(self._stats)):
            if self._stats[i]._name == curr_event._name:
                self._stats[i].aggregate(b - a)
                return
        self._stats.append(EventStats(curr_event._name, b - a,
                                      curr_event._category))

    def calc_stats(self, start_profiling_times, stop_profiling_times):
        num_events = len(self._events)
        use_start_stop = len(start_profiling_times) != 0
        for i in range(0, num_events):
            event = self._events[i]
            if i > 0 and self._events[i-1]._name == "Deinitializing":
                # Drop all events after the Deinitializing event is found
                # because they do not make sense.
                break

            if not use_start_stop:
                self.add_event_to_stats(event)
                continue

            # Check if the event is inbetween start/stop profiling events
            for t in range(len(start_profiling_times)):
                if (event._start_time > start_profiling_times[t] and
                    event._start_time < stop_profiling_times[t]):
                    self.add_event_to_stats(event)
                    break

        if not use_start_stop:
            return

        # Special case for SetState events which need a next one for computing
        # the duration.
        curr_event = self._events[-1]
        if curr_event._type == "SetState":
            for i in range(len(start_profiling_times)):
                if (curr_event._start_time > start_profiling_times[i] and
                    curr_event._start_time < stop_profiling_times[i]):
                    curr_event = Event(curr_event._type, curr_event._name,
                                       curr_event._category,
                                       stop_profiling_times[i])
            self.add_event_to_stats(curr_event)

def read_blocks(input_file):
    empty_lines = 0
    first_line = 1
    blocks = []
    for line in open(input_file):
        if first_line:
            blocks.append([])
            blocks[-1].append(line)
            first_line = 0

        # Check for empty lines
        if not line or line[0] == '\n':
            # If 1st one: new block
            if empty_lines == 0:
                blocks.append([])
            empty_lines += 1
        else:
            # Non empty line: add line in current(last) block
            empty_lines = 0
            blocks[-1].append(line)
    return blocks

def read_field(field, index):
    return field[index+1:-1]

def insert_worker_event(workers, prog_events, block):
    worker_id = -1
    name = None
    start_time = 0.0
    category = None

    for line in block:
        key   = line[:2]
        value = read_field(line, 2)
        if key == "E:": # EventType
            event_type = value
        elif key == "C:": # Category
            category = value
        elif key == "W:": # WorkerId
            worker_id = int(value)
        elif key == "N:": # Name
            name = value
        elif key == "S:": # StartTime
            start_time = float(value)

    # Program events don't belong to workers, they are globals.
    if category == "Program":
        prog_events.append(Event(event_type, name, category, start_time))
        return

    for worker in workers:
        if worker._id == worker_id:
            worker.add_event(event_type, name, category, start_time)
            return
    worker = Worker(worker_id)
    worker.add_event(event_type, name, category, start_time)
    workers.append(worker)

def calc_times(stats):
    tr = 0.0 # Runtime
    tt = 0.0 # Task
    ti = 0.0 # Idle
    ts = 0.0 # Scheduling
    for stat in stats:
        if stat._category == None:
            continue
        if stat._category == "Runtime":
            if stat._name == "Scheduling":
                # Scheduling time is part of runtime but we want to have
                # it separately.
                ts += stat._duration_time
            else:
                tr += stat._duration_time
        elif stat._category == "Task":
            tt += stat._duration_time
        elif stat._category == "Other":
            ti += stat._duration_time
        else:
            print("WARNING: Unknown category '" + stat._category + "'!")
    return ti, tr, tt, ts

def save_times(ti, tr, tt, ts):
    f = open("times.csv", "w+")
    f.write("\"Time\",\"Duration\"\n")
    f.write("\"Runtime\"," + str(tr) + "\n")
    f.write("\"Task\"," + str(tt) + "\n")
    f.write("\"Idle\"," + str(ti) + "\n")
    f.write("\"Scheduling\"," + str(ts) + "\n")
    f.close()

def calc_et(tt_1, tt_p):
    """ Compute the task efficiency (et). This measures the exploitation of
    data locality. """
    return tt_1 / tt_p

def calc_es(tt_p, ts_p):
    """ Compute the scheduling efficiency (es). This measures time spent in
    the runtime scheduler. """
    return tt_p / (tt_p + ts_p)

def calc_er(tt_p, tr_p, ts_p):
    """ Compute the runtime efficiency (er). This measures how the runtime
    overhead affects performance."""
    return (tt_p + ts_p) / (tt_p + tr_p + ts_p)

def calc_ep(tt_p, tr_p, ti_p, ts_p):
    """ Compute the pipeline efficiency (et). This measures how much
    concurrency is available and how well it's exploited. """
    return (tt_p + tr_p + ts_p) / (tt_p + tr_p + ti_p + ts_p)

def calc_e(et, er, ep, es):
    """ Compute the parallel efficiency. """
    return et * er * ep * es

def save_efficiencies(e, ep, er, et, es):
    f = open("efficiencies.csv", "w+")
    f.write("\"Efficiency\",\"Value\"\n")
    f.write("\"Parallel\"," + str(e) + "\n")
    f.write("\"Task\"," + str(et) + "\n")
    f.write("\"Runtime\"," + str(er) + "\n")
    f.write("\"Scheduling\"," + str(es) + "\n")
    f.write("\"Pipeline\"," + str(ep) + "\n")
    f.close()

def usage():
    print("USAGE:")
    print("starpu_trace_state_stats.py [ -te -s=<time> ] <trace.rec>")
    print("")
    print("OPTIONS:")
    print(" -t or --time            Compute and dump times to times.csv")
    print("")
    print(" -e or --efficiency      Compute and dump efficiencies to efficiencies.csv")
    print("")
    print(" -s or --seq_task_time   Used to compute task efficiency between sequential and parallel times")
    print("                         (if not set, task efficiency will be 1.0)")
    print("")
    print("EXAMPLES:")
    print("# Compute event statistics and report them to stdout:")
    print("python starpu_trace_state_stats.py trace.rec")
    print("")
    print("# Compute event stats, times and efficiencies:")
    print("python starpu_trace_state_stats.py -te trace.rec")
    print("")
    print("# Compute correct task efficiency with the sequential task time:")
    print("python starpu_trace_state_stats.py -s=60093.950614 trace.rec")

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hets:",
                                   ["help", "time", "efficiency", "seq_task_time="])
    except getopt.GetoptError as err:
        usage()
        sys.exit(1)

    dump_time = False
    dump_efficiency = False
    tt_1 = 0.0

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-t", "--time"):
            dump_time = True
        elif o in ("-e", "--efficiency"):
            dump_efficiency = True
        elif o in ("-s", "--seq_task_time"):
            tt_1 = float(a)

    if len(args) < 1:
        usage()
        sys.exit()
    recfile = args[0]

    if not os.path.isfile(recfile):
        sys.exit("File does not exist!")

    # Declare a list for all workers.
    workers = []

    # Declare a list for program events
    prog_events = []

    # Read the recutils file format per blocks.
    blocks = read_blocks(recfile)
    for block in blocks:
        if not len(block) == 0:
            first_line = block[0]
            if first_line[:2] == "E:":
                insert_worker_event(workers, prog_events, block)

    # Find allowed range times between start/stop profiling events.
    start_profiling_times = []
    stop_profiling_times = []
    for prog_event in prog_events:
        if prog_event._name == "start_profiling":
            start_profiling_times.append(prog_event._start_time)
        if prog_event._name == "stop_profiling":
            stop_profiling_times.append(prog_event._start_time)

    if len(start_profiling_times) != len(stop_profiling_times):
        sys.exit("Mismatch number of start/stop profiling events!")

    # Compute worker statistics.
    stats = []
    for worker in workers:
        worker.calc_stats(start_profiling_times, stop_profiling_times)
        for stat in worker._stats:
            found = False
            for s in stats:
                if stat._name == s._name:
                    found = True
                    break
            if not found == True:
                stats.append(EventStats(stat._name, 0.0, stat._category, 0))

    # Compute global statistics for all workers.
    for i in range(0, len(workers)):
        for stat in stats:
            s = workers[i].get_event_stats(stat._name)
            if not s == None:
                # A task might not be executed on all workers.
                stat._duration_time += s._duration_time
                stat._count += s._count

    # Output statistics.
    print("\"Name\",\"Count\",\"Type\",\"Duration\"")
    for stat in stats:
        stat.show()

    # Compute runtime, task, idle, scheduling times and dump them to times.csv
    ti_p = tr_p = tt_p = ts_p = 0.0
    if dump_time == True:
        ti_p, tr_p, tt_p, ts_p = calc_times(stats)
        save_times(ti_p, tr_p, tt_p, ts_p)

    # Compute runtime, task, idle efficiencies and dump them to
    # efficiencies.csv.
    if dump_efficiency == True or not tt_1 == 0.0:
        if dump_time == False:
            ti_p, tr_p, tt_p, ts_p = calc_times(stats)
        if tt_1 == 0.0:
            sys.stderr.write("WARNING: Task efficiency will be 1.0 because -s is not set!\n")
            tt_1 = tt_p

        # Compute efficiencies.
        et = round(calc_et(tt_1, tt_p), 6)
        es = round(calc_es(tt_p, ts_p), 6)
        er = round(calc_er(tt_p, tr_p, ts_p), 6)
        ep = round(calc_ep(tt_p, tr_p, ti_p, ts_p), 6)
        e  = round(calc_e(et, er, ep, es), 6)
        save_efficiencies(e, ep, er, et, es)
if __name__ == "__main__":
    main()
