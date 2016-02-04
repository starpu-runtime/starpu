#!/usr/bin/python

##
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016 INRIA
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
##

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
    def __init__(self, type, name, start_time):
        self._type = type
        self._name = name
        self._start_time = start_time

class EventStats():
    def __init__(self, name, duration_time, count = 1):
        self._name = name
        self._duration_time = duration_time
        self._count = count

    def aggregate(self, duration_time):
        self._duration_time += duration_time
        self._count += 1

    def show(self):
        if not self._name == None:
            print "\"" + self._name + "\"," + str(self._count) + "," + str(round(self._duration_time, 6))

class Worker():
    def __init__(self, id):
        self._id        = id
        self._events    = []
        self._stats     = []
        self._stack     = []

    def get_event_stats(self, name):
        for stat in self._stats:
            if stat._name == name:
                return stat
        return None

    def add_event(self, type, name, start_time):
        self._events.append(Event(type, name, start_time))

    def calc_stats(self):
        num_events = len(self._events) - 1
        for i in xrange(0, num_events):
            curr_event = self._events[i]
            next_event = self._events[i+1]

            if next_event._type == "PushState":
                self._stack.append(next_event)
                for j in xrange(i+1, num_events):
                    next_event = self._events[j]
                    if next_event._type == "SetState":
                        break
            elif next_event._type == "PopState":
                curr_event = self._stack.pop()

            # Compute duration with the next event.
            a = curr_event._start_time
            b = next_event._start_time

            found = False
            for j in xrange(len(self._stats)):
                if self._stats[j]._name == curr_event._name:
                    self._stats[j].aggregate(b - a)
                    found = True
                    break
            if not found == True:
                self._stats.append(EventStats(curr_event._name, b - a))

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

def insert_worker_event(workers, block):
    worker_id = -1
    name = None
    start_time = 0.0

    for line in block:
        if line[:2] == "E:": # EventType
            event_type = read_field(line, 2)
        elif line[:2] == "W:": # WorkerId
            worker_id = int(read_field(line, 2))
        elif line[:2] == "N:": # Name
            name = read_field(line, 2)
        elif line[:2] == "S:": # StartTime
            start_time = float(read_field(line, 2))

    for worker in workers:
        if worker._id == worker_id:
            worker.add_event(event_type, name, start_time)
            return
    worker = Worker(worker_id)
    worker.add_event(event_type, name, start_time)
    workers.append(worker)

def main():
    if len(sys.argv) != 2:
        print sys.argv[0] + " <trace.rec>"
	sys.exit(1)
    recfile = sys.argv[1]

    if not os.path.isfile(recfile):
        sys.exit("File does not exist!")

    # Declare a list for all workers.
    workers = []

    # Read the recutils file format per blocks.
    blocks = read_blocks(recfile)
    for block in blocks:
        if not len(block) == 0:
            first_line = block[0]
            if first_line[:2] == "E:":
                insert_worker_event(workers, block)

    # Compute worker statistics.
    stats = []
    for worker in workers:
        worker.calc_stats()
        for stat in worker._stats:
            found = False
            for s in stats:
                if stat._name == s._name:
                    found = True
                    break
            if not found == True:
                stats.append(EventStats(stat._name, 0.0, 0))

    # Compute global statistics for all workers.
    for i in xrange(0, len(workers)):
        for stat in stats:
            s = workers[i].get_event_stats(stat._name)
            if not s == None:
                # A task might not be executed on all workers.
                stat._duration_time += s._duration_time
                stat._count += s._count

    # Output statistics.
    print "\"Value\",\"Events\",\"Duration\""
    for stat in stats:
        stat.show()

if __name__ == "__main__":
    main()
