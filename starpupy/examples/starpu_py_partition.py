# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
	print("Can't find \"Python3 NumPy\" module (consider running \"pip3 install numpy\" or refer to https://numpy.org/install/)")
	exit(77)

import starpu
from starpu import Handle
from starpu import HandleNumpy
import asyncio
import time
import array
import struct

try:
        starpu.init()
except Exception as e:
        print(e)
        exit(77)

def await_fut(fut):
    return fut.result()

# 1-dimension
# arr = np.arange(20)

# 2-dimension
# n, m = 20, 10
# arr = np.arange(n*m).reshape(n, m)

# 3-dimension
# x, y, z = 10, 15, 20
# arr = np.arange(x*y*z).reshape(x, y, z)

# 4-dimension
x, y, z, t = 10, 5, 10, 20
arr = np.arange(x*y*z*t).reshape(x, y, z, t)
print("input array is", arr)

arr_h = Handle(arr)

# split into split_num of sub handles
split_num = 3
#arr_h_list = arr_h.partition(split_num, 0, [6,6,8])
arr_h_list = arr_h.partition(split_num, 0, [3,2,5])

n_arr = arr_h.get_partition_size(arr_h_list)

print("partition size is", n_arr)

def show(x):
    print("Function printing:", x)

@starpu.access(a="RW")
def add(a,b):
	np.add(a,b,out=a)

for i in range(split_num):
	starpu.task_submit(ret_handle=False,ret_fut=False)(add, arr_h_list[i], arr_h_list[i])

# async def main():
# 	for i in range(split_num):
# 		res=starpu.task_submit()(add, arr_h_list[i], arr_h_list[i])
# 		res1=await(res)
# asyncio.run(main())

arr_r = arr_h.acquire(mode='RW')
print("output array is:", arr_r)
arr_h.release()

arr_h.unpartition(arr_h_list, split_num)

arr_h.unregister()

starpu.shutdown()
