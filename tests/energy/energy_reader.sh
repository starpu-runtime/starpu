#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2024-2026   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# This drives tests/energy/energy_reader with FxT trace generation enabled,
# and checks the content of the energy.csv and topo.rec files produced by
# starpu_fxt_tool (see \ref HardwareEnergyMonitoring in the documentation),
#
DIR=$(realpath $(dirname $0))
ROOTDIR=$DIR/../..

TRACEDIR=$ROOTDIR/tests/energy/energy_reader.traces
rm -rf $TRACEDIR
mkdir -p $TRACEDIR
if test ! -f $ROOTDIR/tests/energy/energy_reader
then
    echo "Example not available"
    exit 77
fi

export STARPU_FXT_PREFIX=$TRACEDIR
export STARPU_FXT_TRACE=1
export STARPU_GENERATE_TRACE=1
export STARPU_ENERGY_READER=1
export STARPU_ENERGY_PKG_INTERVAL=10
$STARPU_MS_LAUNCHER $STARPU_LOADER $ROOTDIR/tests/energy/energy_reader

prof_file=prof_file_${USER}_0
if test -z "$USER"
then
    prof_file=prof_file_0
fi

if test ! -f $STARPU_FXT_PREFIX/$prof_file
then
    echo "FxT file not generated (FxT support probably not enabled)"
    exit 77
fi

energy_csv=$STARPU_FXT_PREFIX/energy.csv
topo_rec=$STARPU_FXT_PREFIX/topo.rec

if test ! -f $energy_csv
then
    echo "$energy_csv not generated (StarPU probably not configured with --enable-energyreader)"
    exit 77
fi

if test ! -f $topo_rec
then
    echo "$topo_rec not generated"
    exit 1
fi

header=$(head -n 1 $energy_csv)
expected_header="counter,domain,backend,scope,scope_id,worker,energy_j,delay_ns,timestamp"
if test "$header" != "$expected_header"
then
    echo "Unexpected header in $energy_csv: got '$header', expected '$expected_header'"
    exit 1
fi

# The topology and worker bindings are always recorded when
# STARPU_ENERGY_READER=1, regardless of which (if any) hardware backend is
# actually available on the machine.
if ! grep -q "^Type: WORKER$" $topo_rec
then
    echo "No worker registered in $topo_rec"
    exit 1
fi
if ! grep -q "^Type: PKG$" $topo_rec
then
    echo "No CPU package registered in $topo_rec"
    exit 1
fi

# Actual energy samples on the other hand depend on the backends detected
# at runtime (RAPL / Cray pm_counters / NVML / ROCm SMI), which themselves
# depend on the hardware and on the permissions available on the machine
# running the test, so their absence should not fail the test.
nb_samples=$(tail -n +2 $energy_csv | grep -c .)
if test "$nb_samples" -eq 0
then
    echo "No hardware energy counter available on this machine, only checked the trace format"
    rm -rf $TRACEDIR
    exit 0
fi

echo "Got $nb_samples hardware energy counter samples, checking their format"
tail -n +2 $energy_csv | while read -r line
do
    nb_fields=$(echo "$line" | awk -F, '{print NF}')
    if test "$nb_fields" != 9
    then
        echo "Malformed energy.csv line (expected 9 fields, got $nb_fields): $line"
        exit 1
    fi
    counter=$(echo "$line" | cut -d, -f1)
    energy_j=$(echo "$line" | cut -d, -f7)
    case "$counter" in
        PERF_CPU_PKG|PERF_CPU_CORES|PERF_CPU_DRAM|PERF_PSYS|PERF_CPU_IGPU|\
        POWERCAP_CPU_PKG|POWERCAP_CPU_CORES|POWERCAP_CPU_DRAM|POWERCAP_PSYS|POWERCAP_CPU_IGPU|\
        AMD_CPU_PKG|AMD_CPU_CORE|NVIDIA_GPU|AMD_GPU_ROCMSMI|\
        CRAY_PKG|CRAY_RAM|CRAY_GPU|CRAY_SYSTEM)
            ;;
        *)
            echo "Unknown counter name in energy.csv: $counter"
            exit 1
            ;;
    esac
    case "$energy_j" in
        ''|*[!0-9.]*)
            echo "Non-numeric energy value in energy.csv: $energy_j"
            exit 1
            ;;
    esac
done
ret=$?
if test "$ret" != 0
then
    exit 1
fi

rm -rf $TRACEDIR
exit 0
