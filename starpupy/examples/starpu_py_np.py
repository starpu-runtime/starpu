# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import asyncio
import numpy as np


###############################################################################

def scal(a, t):
	for i in range(len(t)):
		t[i]=t[i]*a
	return t

t=np.array([1,2,3,4,5,6,7,8,9,10])

async def main():
    fut8 = starpu.task_submit()(scal, 2, t)
    res8 = await fut8
    print("The result of Example 10 is", res8)
    print("The return array is", t)
    #print("The result type is", type(res8))

asyncio.run(main())


#starpu.task_wait_for_all()
