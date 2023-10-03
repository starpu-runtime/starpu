# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
from starpu import Handle
from starpu import HandleNumpy
try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
	print("\n\nCan't find \"Python3 NumPy\" module (consider running \"pip3 install numpy\" or refer to https://numpy.org/install/)\n\n")
	starpupy.shutdown()
	exit(77)
import asyncio
import time
import array
import struct
try:
    import nest_asyncio
except ModuleNotFoundError as e:
	print("\n\nCan't find \"Python3 nest_asyncio\" module (consider running \"pip3 install nest_asyncio\")\n\n")
	starpupy.shutdown()
	exit(77)
import json
import sys
import statistics
import test_handle_bench

#############################perf test####################################
# Numpy function
@starpu.access(a="RW", b="R")
def add_numpy(a,b):
	np.add(a,b,out=a)

# custom function
@starpu.access(a="RW", b="R")
def add_custom(a,b):
    for i in range(np.size(a)):
        a[i] = a[i] + b[i]

program_submit1=[]
program_await1=[]

program_submit2=[]
program_await2=[]

program_submit3=[]

num=20
# calculate the standard deviasion
list_std11 = []
list_std12 = []
list_std21 = []
list_std22 = []
list_std3 = []

# using handle return future
def test_comp_handle_ret_fut(a,b):
	async def asy_main():
		start_exec1=0
		end_exec1=0
		start_exec2=0
		end_exec2=0
		list_submit=[]
		list_await=[]
		for t in range(num):
			#print("loop", t)
			start_exec1=time.time()
			res_fut = starpu.task_submit()(add_custom, a, b)
			end_exec1=time.time()

			list_submit.append(end_exec1-start_exec1)

			start_exec2=time.time()
			res = await res_fut
			end_exec2=time.time()

			list_await.append(end_exec2-start_exec2)

		program_submit1.append(statistics.mean(list_submit))
		program_await1.append(statistics.mean(list_await))

		list_std11.append(statistics.stdev(list_submit))
		list_std12.append(statistics.stdev(list_await))

	loop=asyncio.get_event_loop()
	nest_asyncio.apply()
	loop.run_until_complete(asy_main())

# using handle return handle
def test_comp_handle_ret_handle(a,b):
	async def asy_main():
		start_exec1=0
		end_exec1=0
		start_exec2=0
		end_exec2=0
		list_submit=[]
		list_await=[]
		for t in range(num):
			#print("loop", t)
			start_exec1=time.time()
			res_handle = starpu.task_submit(ret_handle=True)(add_custom, a, b)
			end_exec1=time.time()

			list_submit.append(end_exec1-start_exec1)

			start_exec2=time.time()
			starpupy.task_wait_for_all()
			end_exec2=time.time()

			list_await.append(end_exec2-start_exec2)

		program_submit2.append(statistics.mean(list_submit))
		program_await2.append(statistics.mean(list_await))

		list_std21.append(statistics.stdev(list_submit))
		list_std22.append(statistics.stdev(list_await))
	loop=asyncio.get_event_loop()
	nest_asyncio.apply()
	loop.run_until_complete(asy_main())

#without using starpu
def test_numpy(a,b):
	async def asy_main():
		start_exec1=0
		end_exec1=0
		list_submit=[]
		for t in range(num):
			start_exec1=time.time()
			add_numpy(a, b)
			end_exec1=time.time()

			list_submit.append(end_exec1-start_exec1)

		program_submit3.append(statistics.mean(list_submit))

		list_std3.append(statistics.stdev(list_submit))
	loop=asyncio.get_event_loop()
	nest_asyncio.apply()
	loop.run_until_complete(asy_main())


#with handle return future
for i in test_handle_bench.list_size:
	#print("i with handle return future is", i)
	A = np.arange(i)
	test_comp_handle_ret_fut(A, A)
	starpu.unregister(A)

#with handle return handle
for i in test_handle_bench.list_size:
	#print("i with handle return handle is", i)
	A = np.arange(i)
	test_comp_handle_ret_handle(A, A)
	starpu.unregister(A)

#without starpu
for i in test_handle_bench.list_size:
	A = np.arange(i)
	test_numpy(A, A)

retfut_dict={'program_submit':program_submit1, 'program_await': program_await1}
rethandle_dict={'program_submit':program_submit2, 'program_await': program_await2}
nostarpu_dict={'program_submit':program_submit3}

# print(retfut_dict)
# print(rethandle_dict)
# print(nostarpu_dict)

dict_std={'list_std11':list_std11, 'list_std12':list_std12, 'list_std21':list_std21, 'list_std22':list_std22, 'list_std3':list_std3}

#####write the dict in file#####
js1 = json.dumps(retfut_dict)
file1 = open('handle_perf1.txt', 'w')
file1.write(js1)
file1.close()

js2 = json.dumps(rethandle_dict)
file2 = open('handle_perf2.txt', 'w')
file2.write(js2)
file2.close()

js3 = json.dumps(nostarpu_dict)
file3 = open('handle_perf3.txt', 'w')
file3.write(js3)
file3.close()

js_std = json.dumps(dict_std)
file_std = open('handle_perf_std.txt', 'w')
file_std.write(js_std)
file_std.close()

starpupy.shutdown()
