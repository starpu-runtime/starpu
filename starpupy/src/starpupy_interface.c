/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#include "starpupy_interface.h"

static void pyobject_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) data_interface;

	int node;
	for (node =0; node < STARPU_MAXNODES; node++)
	{
		struct starpupyobject_interface *local_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->object = pyobject_interface->object;
		}
		else
		{
			local_interface->object = NULL;
		}
	}
}

static starpu_ssize_t pyobject_allocate_data_on_node(void *data_interface, unsigned node)
{
	return 0;
}

static void pyobject_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) data_interface;

	if (pyobject_interface->object != NULL)
	{
		Py_DECREF(pyobject_interface->object);
	}
	pyobject_interface->object = NULL;
}

static int pyobject_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, node);

	PyObject *obj = pyobject_interface->object;

	PyObject *cloudpickle_module = PyImport_ImportModule("cloudpickle");
	if (cloudpickle_module == NULL)
	{
		printf("can't find cloudpickle module\n");
		exit(1);
	}
	PyObject *dumps = PyObject_GetAttrString(cloudpickle_module, "dumps");
	PyObject *obj_bytes = PyObject_CallFunctionObjArgs(dumps, obj, NULL);

	char *obj_data;
	Py_ssize_t obj_data_size;
	PyBytes_AsStringAndSize(obj_bytes, &obj_data, &obj_data_size);

	char *data;
	data = (void*)starpu_malloc_on_node_flags(node, obj_data_size, 0);

	memcpy(data, obj_data, obj_data_size);

	*ptr = data;
	*count = obj_data_size;

	Py_DECREF(obj_bytes);
	return 0;
}

static int pyobject_peek_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	char *data = ptr;
	PyObject *pickle_module = PyImport_ImportModule("pickle");
	if (pickle_module == NULL)
	{
		printf("can't find pickle module\n");
		exit(1);
	}
	PyObject *loads = PyObject_GetAttrString(pickle_module, "loads");
	PyObject *obj_bytes_str = PyBytes_FromStringAndSize(data, count);
	PyObject *obj= PyObject_CallFunctionObjArgs(loads, obj_bytes_str, NULL);

	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, node);

	if (pyobject_interface->object != NULL)
	{
		Py_DECREF(pyobject_interface->object);
	}
	pyobject_interface->object = obj;

	return 0;
}

static int pyobject_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	pyobject_peek_data(handle, node, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t) ptr, count, 0);

	return 0;
}

static uint32_t starpupy_footprint(starpu_data_handle_t handle)
{
	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	PyObject *obj = pyobject_interface->object;

	/*fet obj.__class__*/
	PyObject *obj_class=PyObject_GetAttrString(obj,"__class__");
	// PyObject_Print(obj_class, stdout, 0);
	// printf("\n");

	uint32_t crc = 0;
	crc=starpu_hash_crc32c_be_ptr(obj_class, crc);

#ifdef STARPU_PYTHON_HAVE_NUMPY
	const char *tp = Py_TYPE(obj)->tp_name;
	/*if the object is a numpy array*/
	if (strcmp(tp, "numpy.ndarray")==0)
	{
		import_array1(0);
		/*get the array size*/
		int n1 = PyArray_SIZE(obj);
		/*get the item size*/
		int n2 = PyArray_ITEMSIZE(obj);

		crc=starpu_hash_crc32c_be(n1, crc);
		crc=starpu_hash_crc32c_be(n2, crc);
	}
	else
#endif
	{
		crc=starpu_hash_crc32c_be_ptr(obj, crc);
	}

	return crc;
}

static struct starpu_data_interface_ops interface_pyobject_ops =
{
	.register_data_handle = pyobject_register_data_handle,
	.allocate_data_on_node = pyobject_allocate_data_on_node,
	.free_data_on_node = pyobject_free_data_on_node,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpupyobject_interface),
	.footprint = starpupy_footprint,
	.pack_data = pyobject_pack_data,
	.peek_data = pyobject_peek_data,
	.unpack_data = pyobject_unpack_data,
	.dontcache = 0,
};

int starpupy_data_register(starpu_data_handle_t *handleptr, unsigned home_node, PyObject *obj)
{
	struct starpupyobject_interface pyobject_interface =
	{
	 .object = obj
	};

	if (interface_pyobject_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_pyobject_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &pyobject_interface, &interface_pyobject_ops);

	return interface_pyobject_ops.interfaceid;
}
