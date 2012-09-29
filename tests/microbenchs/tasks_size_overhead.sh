#!/bin/sh
./tasks_size_overhead > tasks_size_overhead.output
./tasks_size_overhead.gp
gv tasks_size_overhead.eps
