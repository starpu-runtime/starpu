# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import starpu
from starpu import starpupy

import time
import sys
import getopt
import asyncio
import cProfile
import sys

mincpus = 1
maxcpus = starpupy.worker_get_count_by_type(starpu.STARPU_CPU_WORKER)
cpustep = 1

mintime = 128
maxtime = 128*1024
factortime = 2

ntasks = 64
nbuffers = 0
total_nbuffers = 0

#################parameters##############
try:
	opts, args = getopt.getopt(sys.argv[1:],"i:b:B:c:C:s:t:T:f:h")
except getopt.GetoptError:
	print("Usage:", sys.argv[0], "\n"\
	 "\t[-h help] \n "\
	 "\t[-i ntasks] [-b nbuffers] [-B total_nbuffers] \n"\
	 "\t[-c mincpus] [ -C maxcpus] [-s cpustep]\n"\
	 "\t[-t mintime] [-T maxtime] [-f factortime]")
	starpupy.shutdown()
	sys.exit(1)
for opt, arg in opts:
	if opt == '-i':
		ntasks = int(arg)
	elif opt == '-b':
		nbuffers = int(arg)
	elif opt == '-B':
		total_nbuffers = int(arg)
	elif opt == '-c':
		mincpus = int(arg)
	elif opt == '-C':
		maxcpus = int(arg)
	elif opt == '-s':
		cpustep = int(arg)
	elif opt == '-t':
		mintime = int(arg)
	elif opt == '-T':
		maxtime = int(arg)
	elif opt == '-f':
		factortime = int(arg)
	elif opt == '-h':
		print("Usage:", sys.argv[0], "[-h help] \n "\
		 "\t[-i ntasks] [-b nbuffers] [-B total_nbuffers] \n"\
		 "\t[-c mincpus] [ -C maxcpus] [-s cpustep]\n"\
		 "\t[-t mintime] [-T maxtime] [-f factortime]\n")
		print("runs \'ntasks\' tasks\n"\
		"- using \'nbuffers\' data each, randomly among \'total_nbuffers\' choices,\n"\
		"- with varying task durations, from \'mintime\' to \'maxtime\' (using \'factortime\')\n"\
		"- on varying numbers of cpus, from \'mincpus\' to \'maxcpus\' (using \'cpustep\')\n"\
		"\n"\
		"currently selected parameters: ", ntasks, " tasks using ", nbuffers, " buffers among ", total_nbuffers, \
		", from ", mintime, "us to ", maxtime, "us (factor ", factortime, "), from ", mincpus, " cpus to ", maxcpus, " cpus (step ", cpustep, ")", sep='')
		starpupy.shutdown()
		sys.exit(0)

########################################

# multiplication increment
def range_multi(start, end, factor):
	val_multi = []
	val = start
	while val <= end:
		val_multi.append(val)
		val = val * factor 
	return val_multi

# the test function
def func_test(t):
	time.sleep(t/1000000)

#pr = cProfile.Profile()

f = open("tasks_size_overhead.output",'w')

method="handle"
if len(sys.argv) > 1:
        method=sys.argv[1]

print("# tasks :", ntasks, "buffers :", nbuffers, "totoal_nbuffers :", total_nbuffers, file=f)
print("# ncups", end='\t', file=f)
for size in range_multi(mintime, maxtime, factortime):
	print(size, "iters(us)\ttotal(s)", end='\t', file=f)
print(end='\n', file=f)

print("\"seq\"\t", end=' ', file=f)
for size in range_multi(mintime, maxtime, factortime):
	#print("time size is", size)
	dstart=time.time()
	for i in range(ntasks):
		func_test(size)
	dend=time.time()
	print(int((dend-dstart)/ntasks*1000000), "\t", dend-dstart, end='\t', file=f)
	#print(size, "\t", dend-dstart, end='\t', file=f)
print(end='\n', file=f)

#pr.enable()

if method == "handle":
        # return value is handle
        for ncpus in range(mincpus, maxcpus+1, cpustep):
                starpupy.set_ncpu(ncpus)
                #print("ncpus is", ncpus)
                print(ncpus, end='\t', file=f)
                for size in range_multi(mintime, maxtime, factortime):
                        #print("time size is", size)
                        start=time.time()
                        for i in range(ntasks*ncpus):
                                res=starpu.task_submit(ret_handle=True)("func_test", size)
                        starpupy.task_wait_for_all()
                        end=time.time()
                        timing = end-start
                        print(size, "\t", timing/ncpus, end='\t', file=f)
                print(end='\n', file=f)

elif method == "futur":
        # return value is future
        async def main():
                for ncpus in range(mincpus, maxcpus+1, cpustep):
                        starpupy.set_ncpu(ncpus)
                        #print("ncpus is", ncpus)
                        print(ncpus, end='\t', file=f)
                        for size in range_multi(mintime, maxtime, factortime):
                                #print("time size is", size)
                                start=time.time()
                                for i in range(ntasks*ncpus):
                                        fut=starpu.task_submit(ret_fut=True)("func_test", size)
                                starpupy.task_wait_for_all()
                                end=time.time()
                                timing = end-start
                                print(size, "\t", timing/ncpus, end='\t', file=f)
                        print(end='\n', file=f)
        asyncio.run(main())

else:
        # return value is neither future nor handle
        for ncpus in range(mincpus, maxcpus+1, cpustep):
                starpupy.set_ncpu(ncpus)
                #print("ncpus is", ncpus)
                print(ncpus, end='\t', file=f)
                for size in range_multi(mintime, maxtime, factortime):
                        #print("time size is", size)
                        start=time.time()
                        for i in range(ntasks*ncpus):
                                fut=starpu.task_submit(ret_fut=False)("func_test", size)
                        starpupy.task_wait_for_all()
                        end=time.time()
                        timing = end-start
                        print(size, "\t", timing/ncpus, end='\t', file=f)
                print(end='\n', file=f)

#pr.disable()

f.close()
#pr.print_stats()
starpupy.shutdown()
