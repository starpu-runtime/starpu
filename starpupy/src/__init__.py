# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

from . import starpupy
from .delay import *
from .handle_access import *
#from . import joblib
from .intermedia import *

import asyncio
try:
    import numpy as np
    has_numpy=True
except:
    has_numpy=False

async def wait_for_fut(fut):
	return await fut

#class handle
class Handle(object):

	def __init__(self, obj):
		self.obj=obj
		self.handle_obj=starpupy.starpupy_data_register(self.obj)

	def get_capsule(self):
		return self.handle_obj

	# get PyObject
	def get(self):
		return starpupy.starpupy_get_object(self.handle_obj)

	# get array object
	def acquire(self, mode='R'):
		return starpupy.starpupy_acquire_handle(self.handle_obj, mode)

	# release
	def release(self):
		return starpupy.starpupy_release_handle(self.handle_obj)

	# unregister
	def unregister(self):
		return starpupy.starpupy_data_unregister(self.handle_obj)

	# unregister_submit
	def unregister_submit(self):
		return starpupy.starpupy_data_unregister_submit(self.handle_obj)

	# partition
	def partition(self, nchildren, dim, chunks_list=[]):
		return starpupy.starpupy_data_partition(self.handle_obj, nchildren, dim, chunks_list)

	# get partition size
	def get_partition_size(self, handle_list):
		return starpupy.starpupy_get_partition_size(self.handle_obj, handle_list)

	# unpartition
	def unpartition(self, handle_list, nchildren):
		return starpupy.starpupy_data_unpartition(self.handle_obj, handle_list, nchildren)

#class handle
class HandleNumpy(Handle):
	if has_numpy:
		def __init__(self, *args, dtype=np.dtype('float64')):
			#self.narray=narray
			self.dtype=dtype
			self.handle_obj=starpupy.starpupy_numpy_register(*args, self.dtype)


#detect class handle
class Handle_token(object):
	pass

handle_dict={}
def handle_dict_set_item(obj, handle):
	assert handle_dict.get(id(obj))==None
	handle_dict[id(obj)]=handle
	return handle_dict