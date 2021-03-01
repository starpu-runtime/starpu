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
import os

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

#add options in function task_submit
def task_submit(*, name=None, synchronous=0, priority=0, color=None, flops=None, perfmodel=None):
	if perfmodel==None:
		dict_option={'name': name, 'synchronous': synchronous, 'priority': priority, 'color': color, 'flops': flops, 'perfmodel': None}
	else:
		p=dict_perf_generator(perfmodel)
		dict_option={'name': name, 'synchronous': synchronous, 'priority': priority, 'color': color, 'flops': flops, 'perfmodel': p.get_struct()}

	def call_task_submit(f, *args):
		fut=starpupy._task_submit(f, *args, dict_option)
		return fut
	return call_task_submit

# dump performance model and show the plot
def perfmodel_plot(perfmodel, view=True):
	p=dict_perf[perfmodel]
	starpupy.save_history_based_model(p.get_struct())
	if view == True:
		os.system('starpu_perfmodel_plot -s "' + perfmodel +'"')
		os.system('gnuplot starpu_'+perfmodel+'.gp')
		os.system('gv starpu_'+perfmodel+'.eps')
