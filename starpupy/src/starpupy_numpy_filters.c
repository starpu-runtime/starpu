/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "starpupy_buffer_interface.h"
#include "starpupy_numpy_filters.h"

#define RETURN_EXCEPT(...) do{ \
		PyObject *starpupy_err = PyObject_GetAttrString(self, "error"); \
		PyErr_Format(starpupy_err, __VA_ARGS__);		\
		Py_DECREF(starpupy_err); \
		return NULL;\
}while(0)

static void starpupy_numpy_filter(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpupy_buffer_interface *buffer_father = (struct starpupy_buffer_interface *) father_interface;
	struct starpupy_buffer_interface *buffer_child = (struct starpupy_buffer_interface *) child_interface;

	size_t elemsize = buffer_father->item_size;

	/*get the ndim*/
	int ndim = buffer_father->dim_size;

#ifdef STARPU_PYTHON_HAVE_NUMPY
	Py_ssize_t nbuf = buffer_father->buffer_size;
	int narr = nbuf/elemsize;

	int child_narr;
	size_t offset;

	int dim = f->filter_arg;

	unsigned ni[ndim];
	int i;
	for (i=0; i<ndim; i++)
	{
		ni[i] = (unsigned)buffer_father->array_dim[i];
	}

	unsigned nn = ni[dim];
	unsigned ld;

	if (dim == 0 && ndim != 1)
	{
		ld = ni[1];
	}
	else if (dim == 1 || ndim == 1)
	{
		ld = 1;
	}
	else
	{
		ld = 1;
		for (i=0; i<dim; i++)
		{
			ld  = ld * ni[i];
		}
	}

	/*we will do the partition on ni*/
	unsigned child_nn;
	unsigned* chunks_list = (unsigned*) f->filter_arg_ptr;

	if (chunks_list != NULL)
	{
		child_nn = chunks_list[id];
		unsigned chunk_nn = 0;
		unsigned j = 0;
		while(j < id)
		{
			chunk_nn = chunk_nn + chunks_list[j];
			j++;
		}
		offset = chunk_nn * ld * elemsize;
	}
	else
	{
		starpu_filter_nparts_compute_chunk_size_and_offset(nn, nchunks, elemsize, id, ld, &child_nn, &offset);
	}

	child_narr = narr/nn*child_nn;

	buffer_child->py_buffer = buffer_father->py_buffer + offset;
	buffer_child->buffer_size = child_narr * elemsize;

	npy_intp *child_dim;
	child_dim = (npy_intp*)malloc(ndim*sizeof(npy_intp));
	for (i=0; i<ndim; i++)
	{
		if (i!=dim)
		{
			child_dim[i] = ni[i];
		}
		else
		{
			child_dim[i] = child_nn;
		}
	}
	buffer_child->array_dim = child_dim;
#endif
	buffer_child->buffer_type = buffer_father->buffer_type;
	buffer_child->dim_size = ndim;
	buffer_child->array_type = buffer_father->array_type;
	buffer_child->item_size = elemsize;

}

/*wrapper data partition*/
PyObject* starpu_data_partition_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;
	int nparts;
	int dim;
	PyObject *chunks_list;

	if (!PyArg_ParseTuple(args, "OIIO", &handle_obj, &nparts, &dim, &chunks_list))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	int node = starpu_data_get_home_node(handle);
	struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);

	int ndim = local_interface->dim_size;

	if (ndim <= 0)
	{
		RETURN_EXCEPT("Dimension size %d must be greater than 0.", ndim);
	}

	if (dim < 0)
	{
		RETURN_EXCEPT("The given dimension dim %d must not be less than 0.", dim);
	}

	if (dim >= ndim)
	{
		RETURN_EXCEPT("dim %d must be less than dimension size %d.", dim, ndim);
	}

	int i;
	int dim_len = 0;
	int nlist = PyList_Size(chunks_list);
	int nchunks[nparts];

	if(nlist != 0)
	{
		if (nlist != nparts)
		{
			RETURN_EXCEPT("The chunk list size %d does not correspond to the required split size %d.", nlist, nparts);
		}

		for (i=0; i<nparts; i++)
		{
			nchunks[i] = PyLong_AsLong(PyList_GetItem(chunks_list, i));
			dim_len += nchunks[i];
		}
#ifdef STARPU_PYTHON_HAVE_NUMPY
		if (dim_len != local_interface->array_dim[dim])
		{
			RETURN_EXCEPT("The total length of segments in chunk list %d must be equal to the length of selected dimension %d.", dim_len, local_interface->array_dim[dim]);
		}
#endif
	}

	/*filter func*/
	struct starpu_data_filter f;
	starpu_data_handle_t handles[nparts];

	f.filter_func = starpupy_numpy_filter;
	f.nchildren = nparts;
	f.get_nchildren = 0;
	f.get_child_ops = 0;
	f.filter_arg_ptr = (nlist==0) ? NULL : nchunks;
	/* partition along the given dimension */
	f.filter_arg = dim;

	Py_BEGIN_ALLOW_THREADS
	starpu_data_partition_plan(handle, &f, handles);
	Py_END_ALLOW_THREADS

	PyObject *handle_list = PyList_New(nparts);
	for(i=0; i<nparts; i++)
	{
		PyList_SetItem(handle_list, i, PyCapsule_New(handles[i], "Handle", NULL));
	}

	return handle_list;
}

/*get the partition size list*/
PyObject* starpupy_get_partition_size_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;
	PyObject *handle_list;
	int nparts;

	if (!PyArg_ParseTuple(args, "OO", &handle_obj, &handle_list))
		return NULL;

	nparts = PyList_Size(handle_list);

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	PyObject *arr_size = PyList_New(nparts);

	int i;
	for(i=0; i<nparts; i++)
	{
		PyObject *handles_cap = PyList_GetItem(handle_list, i);
		/*protect borrowed reference, decrement after using*/
		Py_INCREF(handles_cap);
		starpu_data_handle_t handle_tmp = (starpu_data_handle_t) PyCapsule_GetPointer(handles_cap, "Handle");

		int node = starpu_data_get_home_node(handle_tmp);
		struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle_tmp, node);
		int narr = local_interface->buffer_size/local_interface->item_size;

		PyList_SetItem(arr_size, i, Py_BuildValue("I", narr));

		Py_DECREF(handles_cap);
	}

	return arr_size;
}

/*wrapper data unpartition*/
PyObject* starpu_data_unpartition_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;
	PyObject *handle_list;
	int nparts;

	if (!PyArg_ParseTuple(args, "OOI", &handle_obj, &handle_list, &nparts))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	starpu_data_handle_t handles[nparts];

	int i;
	for(i=0; i<nparts; i++)
	{
		PyObject *handles_cap = PyList_GetItem(handle_list, i);
		/*protect borrowed reference, decrement rigth after*/
		Py_INCREF(handles_cap);
		handles[i] = (starpu_data_handle_t) PyCapsule_GetPointer(handles_cap, "Handle");
		Py_DECREF(handles_cap);
	}

	Py_BEGIN_ALLOW_THREADS
	starpu_data_partition_clean(handle, nparts, handles);
	Py_END_ALLOW_THREADS

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}
