# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Universit'e de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
except ModuleNotFoundError as e:
	print("Can't find \"Python3 NumPy\" module (consider running \"pip3 install numpy\" or refer to https://numpy.org/install/)")
	exit(77)

import starpu
from starpu import starpupy
import asyncio

starpu.init()

###############################################################################

def scal(x, t):
	for i in range(len(t)):
		t[i] = t[i] * x
	print ("Example scal(scalar, array):")
	return t

def add(x, y):
	print ("Example add(array, array):")
	return x + y

def multi(x, y):
	print ("Example multi(array, array):")
	return x * y

def matrix_multi(x, y):
	print ("Example matrix_multi(array, array):")
	return x @ y

t = np.arange(10)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.array([[1, 2], [3, 4]])
d = np.array([[2, 2], [2, 2]])

async def main():
    fut1 = starpu.task_submit()(scal, 2, t)
    res1 = await fut1
    print("The result is", res1)

    # two array element addition
    fut2 = starpu.task_submit()(add, a, b)
    res2 = await fut2
    print("The result is", res2)

    # two array element multiplication
    fut3 = starpu.task_submit()(multi, c, d)
    res3 = await fut3
    print("The result is", res3)

    # two array matrix multiplication
    fut4 = starpu.task_submit()(matrix_multi, c, d)
    res4 = await fut4
    print("The result is", res4)

    # two array matrix multiplication (inverse order)
    fut5 = starpu.task_submit()(matrix_multi, d, c)
    res5 = await fut5
    print("The result is", res5)


try:
        asyncio.run(main())
except starpupy.error as e:
        print("No worker to execute the job")
        starpupy.shutdown()
        exit(77)

starpu.shutdown()
