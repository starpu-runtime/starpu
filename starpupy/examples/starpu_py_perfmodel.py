# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2024   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import asyncio

try:
        starpu.init()
except Exception as e:
        print(e)
        exit(77)

@starpu.access(a="RW",b="R")
def arr_add(a,b):
	for i in range(np.size(a)):
		a[i] = a[i] + b[i]

a_h = Handle(np.arange(1000))
b_h = Handle(np.arange(1000))

starpu.task_submit(ret_fut=False, perfmodel="arr_add")(arr_add, a_h, b_h)

print("Array is", a_h.get())

a_h.unregister()
b_h.unregister()

starpu.shutdown()
