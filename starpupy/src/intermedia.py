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

import starpu
from starpu import starpupy
import os
import inspect
import array

# define the different architecture
STARPU_CPU_WORKER = 0
STARPU_CUDA_WORKER = 1
STARPU_OPENCL_WORKER = 2
STARPU_MAX_FPGA_WORKER = 4
STARPU_MPI_MS_WORKER = 5
STARPU_TCPIP_MS_WORKER = 6
STARPU_HIP_WORKER = 7
STARPU_NARCH = 8
STARPU_ANY_WORKER = 255

#class perfmodel
class Perfmodel(object):
	def __init__(self, symbol):
		self.symbol=symbol
		self.pstruct=starpupy.init_perfmodel(self.symbol)

	def get_struct(self):
		return self.pstruct

	def __del__(self):
	#def free_struct(self):
		starpupy.free_perfmodel(self.pstruct)

# generate the dictionary which contains the perfmodel symbol and its struct pointer
dict_perf={}
def dict_perf_generator(perfsymbol):
	if dict_perf.get(perfsymbol)==None:
		p=Perfmodel(perfsymbol)
		dict_perf[perfsymbol]=p
	else:
		p=dict_perf[perfsymbol]
	return p

# add options in function task_submit
def task_submit(**kwargs):
	# set perfmodel
	perf=None
	if kwargs.__contains__("perfmodel") and kwargs['perfmodel']!=None:
		perf=dict_perf_generator(kwargs['perfmodel'])
	kwargs['perfmodel']=perf

	def call_task_submit(f, *args):
		modes={}
		# if there is access mode defined
		if hasattr(f,"starpu_access"):
			# the starpu_access attribute of f is the access mode
			access_mode=f.starpu_access
			# get the name of formal arguments of f
			f_args = inspect.getfullargspec(f).args
			# check the access right of argument is set in mode or not
			for i in range(len(f_args)):
				if f_args[i] in access_mode.keys():
					# set access mode in modes option
					modes[id(args[i])]=access_mode[f_args[i]]
		kwargs['modes']=modes

		res=starpupy._task_submit(f, *args, kwargs)
		return res
	return call_task_submit

# dump performance model and show the plot
def perfmodel_plot(perfmodel, view=True):
	p=dict_perf[perfmodel]
	starpupy.save_history_based_model(p)
	if view == True:
		os.system('starpu_perfmodel_plot -s "' + perfmodel +'"')
		os.system('gnuplot starpu_'+perfmodel+'.gp')
		os.system('gv starpu_'+perfmodel+'.eps')

# acquire object
def acquire(obj, mode='R'):
	return starpupy.starpupy_acquire_object(obj, mode)

# release object
def release(obj):
	return starpupy.starpupy_release_object(obj)

# acquire object
def unregister(obj):
	return starpupy.starpupy_data_unregister_object(obj)

# acquire object
def unregister_submit(obj):
	return starpupy.starpupy_data_unregister_submit_object(obj)

def init():
        return starpupy.init()

def shutdown():
        return starpupy.shutdown()
