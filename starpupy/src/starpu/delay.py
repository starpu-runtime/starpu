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
from starpu import starpupy
import starpu
import asyncio
from functools import partial

def delayed(f=None,*, name=None, synchronous=0, priority=0, color=None, flops=None, perfmodel=None):
	# add options of task_submit
	if f is None:
		return partial(delayed, name=name, synchronous=synchronous, priority=priority, color=color, flops=flops, perfmodel=perfmodel)
	def submit(*args):
		fut = starpu.task_submit(name=name, synchronous=synchronous, priority=priority,\
								 color=color, flops=flops, perfmodel=perfmodel)(f, *args)
		return fut
	return submit
