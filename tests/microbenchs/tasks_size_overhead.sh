#!/bin/sh
ROOT=${0%.sh}
$ROOT > tasks_size_overhead.output
$ROOT.gp
gv tasks_size_overhead.eps
