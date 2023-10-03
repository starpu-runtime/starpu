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
	print("Can't find \"Python3 NumPy\" module (consider running \"pip3 install numpy\" or refer to https://numpy.org/install/)")
	starpupy.shutdown()
	exit(77)
import asyncio
import time
import array
import struct
import nest_asyncio
import json
import sys
import statistics

#############################perf test####################################
# Numpy function
@starpu.access(a="RW", b="R")
def add(a,b):
	np.add(a,b,out=a)

# custom function
# @starpu.access(a="RW", b="R")
# def add(a,b):
# 	for i in range(np.size(a)):
# 		a[i] = a[i] + b[i]

listX = [10, 100, 1000, 10000, 100000, 1000000]
list_size = []
for x in listX:
	for X in range(x, x*10, x):
		list_size.append(X)
list_size.append(10000000)
list_size.append(20000000)
list_size.append(30000000)
list_size.append(40000000)
list_size.append(50000000)
#print("list of size is",list_size)

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

# using handle
def test_comp_handle(a,b):
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
			res_fut = starpu.task_submit()(add, a, b)
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

#without using handle
def test_comp(a,b):
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
			res_fut = starpu.task_submit(arg_handle=False)(add, a, b)
			end_exec1=time.time()

			list_submit.append(end_exec1-start_exec1)

			start_exec2=time.time()
			res = await res_fut
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
			add(a, b)
			end_exec1=time.time()
			
			list_submit.append(end_exec1-start_exec1)

		program_submit3.append(statistics.mean(list_submit))

		list_std3.append(statistics.stdev(list_submit))

	loop=asyncio.get_event_loop()
	nest_asyncio.apply()
	loop.run_until_complete(asy_main())


#with handle
for i in list_size:
	#print("i with handle is", i)
	A = np.arange(i)
	test_comp_handle(A, A)

	starpu.unregister(A)

#without handle	
for i in list_size:
	#print("i without handle is", i)
	A = np.arange(i)
	test_comp(A, A)

#without starpu
for i in list_size:
	A = np.arange(i)
	test_numpy(A, A)


withhandle_dict={'program_submit':program_submit1, 'program_await': program_await1}
nohandle_dict={'program_submit':program_submit2, 'program_await': program_await2}
nostarpu_dict={'program_submit':program_submit3}

# print(withhandle_dict)
# print(nohandle_dict)
# print(nostarpu_dict)

dict_std={'list_std11':list_std11, 'list_std12':list_std12, 'list_std21':list_std21, 'list_std22':list_std22, 'list_std3':list_std3}

#####write the dict in file#####
js1 = json.dumps(withhandle_dict)   
file1 = open('handle_perf1.txt', 'w')  
file1.write(js1)  
file1.close()

js2 = json.dumps(nohandle_dict)   
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
