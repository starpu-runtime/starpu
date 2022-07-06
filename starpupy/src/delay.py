# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2022  Universit'e de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import inspect

def delayed(f=None, **kwargs):
	# add options of task_submit
	if f is None:
		return partial(delayed, **kwargs)
	def submit(*args):
		# set the access right
		access_mode={}
		f_args = inspect.getfullargspec(f).args
		# check the access right of argument is set in mode or not
		for i in range(len(f_args)):
			if f_args[i] in kwargs.keys():
				# write access modes in f.access attribute
				access_mode[f_args[i]]=kwargs[f_args[i]]
				setattr(f, "starpu_access", access_mode)

		fut = starpu.task_submit(**kwargs)(f, *args)
		return fut
	return submit
