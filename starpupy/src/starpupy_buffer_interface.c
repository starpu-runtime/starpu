/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdarg.h>
#include <starpu.h>
#include <common/utils.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#include "starpupy_buffer_interface.h"

PyObject* starpupy_buffer_get_numpy(struct starpupy_buffer_interface *pybuffer_interface)
{
#ifdef STARPU_PYTHON_HAVE_NUMPY
	char* pybuf = pybuffer_interface->py_buffer;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;
	int arr_type = pybuffer_interface->array_type;
	size_t nitem = pybuffer_interface->item_size;
	npy_intp narray = nbuf/nitem;
	npy_intp* get_dim = pybuffer_interface->array_dim;
	int get_ndim = pybuffer_interface->dim_size;

	/*store the dim array in a tuple*/
	PyObject* dim_tup = PyTuple_New(get_ndim);

	int i;
	for (i=0; i<get_ndim; i++)
	{
		PyTuple_SetItem(dim_tup, i, Py_BuildValue("i", get_dim[i]));
	}
	/*convert tuple to a sequence object*/
	PyObject* dim_seq = PySequence_Fast(dim_tup, "Can't generate dimension sequence.");

	/*get the buffer object*/
	PyObject* numpy_buf = PyMemoryView_FromMemory(pybuf, nbuf, PyBUF_WRITE);

	import_array();
	/*construct a one-dimensional Numpy array from buffer*/
	PyObject* numpy_arr1d = PyArray_FromBuffer(numpy_buf,PyArray_DescrFromType(arr_type),narray,0);

	/*reshape the one-dimensional Numpy array in case the original array is more than one dimension*/
	PyObject* numpy_arr = PyArray_Reshape((PyArrayObject*)numpy_arr1d, dim_seq);

	Py_DECREF(dim_tup);
	Py_DECREF(dim_seq);
	Py_DECREF(numpy_buf);
	Py_DECREF(numpy_arr1d);

	return numpy_arr;
#endif
	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* starpupy_buffer_get_arrarr(struct starpupy_buffer_interface *pybuffer_interface)
{
	char arr_typecode = pybuffer_interface->typecode;
	char* pybuf = pybuffer_interface->py_buffer;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;
	size_t nitem = pybuffer_interface->item_size;
	/*get size of array*/
	int narray = nbuf/nitem;

	/*create the new array.array*/
	PyObject *arr_module = PyImport_ImportModule("array");
	PyObject *arr_dict = PyModule_GetDict(arr_module);

	/*get array.array class*/
	PyObject *arr_class = PyDict_GetItemString(arr_dict, "array");

	/*create an instance of array.array, decrement in the end of the function*/
	PyObject *arr_instance = PyInstanceMethod_New(arr_class);

	/*get the buffer bytes, decrement in the end of the function*/
	PyObject *pybt=PyBytes_FromStringAndSize(pybuf, nbuf);

	/*get the array elements, reference is stolen by PyTuple_SetItem*/
	PyObject *arr_list = NULL;

	/*if the element is not unicode character*/
	if (arr_typecode!='u')
	{
		char type_str[narray+1];
		memset(type_str, arr_typecode, narray);
		type_str[narray] = 0;

		/*get the array element list using struct module*/
		PyObject *struct_module = PyImport_ImportModule("struct");
		arr_list = PyObject_CallMethod(struct_module, "unpack", "sO", type_str, pybt);

		Py_DECREF(struct_module);
	}
	/*if the element is unicode character*/
	else
	{
		/*decode buffer bytes to unicode*/
		PyObject* pyuni = PyUnicode_DecodeUTF32(PyBytes_AsString(pybt), PyBytes_Size(pybt), "can't decode", NULL);
		/*convert unicode to wide char*/
		wchar_t* uni_str = PyUnicode_AsWideCharString(pyuni, NULL);

		if(uni_str != NULL)
		{
			arr_list = Py_BuildValue("u", uni_str);
			PyMem_Free(uni_str);
		}

		Py_DECREF(pyuni);
	}

	/*initialize the instance*/
	PyObject *arr_args=PyTuple_New(2);

	char arr_type[]={arr_typecode, 0};
	PyTuple_SetItem(arr_args, 0, Py_BuildValue("s", arr_type));
	PyTuple_SetItem(arr_args, 1, arr_list);

	PyObject *arr_obj = PyObject_CallObject(arr_instance,arr_args);

	Py_DECREF(pybt);
	Py_DECREF(arr_module);
	Py_DECREF(arr_instance);
	Py_DECREF(arr_args);

	return arr_obj;
}

PyObject* starpupy_buffer_get_memview(struct starpupy_buffer_interface *pybuffer_interface)
{
	char* pybuf = pybuffer_interface->py_buffer;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;
	char mem_format = pybuffer_interface->typecode;
	size_t nitem = pybuffer_interface->item_size;
	int ndim = pybuffer_interface->dim_size;
	int* mem_shape = pybuffer_interface->shape;
	int narray = nbuf/nitem;

	/*decrement in each if{}*/
	PyObject *pybt=PyBytes_FromStringAndSize(pybuf, nbuf);

	/*return value of the function*/
	PyObject *memview_obj = NULL;
	if(mem_format=='B')
	{
		memview_obj = pybt;
	}
	/*if the element is not unicode character of array.array*/
	else if(mem_format!='w')
	{
		/* We have a flat array, split it into ndim-dimension lists of lists according to mem_shape */
		char type_str[narray+1];
		memset(type_str, mem_format, narray);
		type_str[narray] = 0;

		/*get the array element list using struct module, decrement after used*/
		PyObject *struct_module = PyImport_ImportModule("struct");
		PyObject *m_obj = PyObject_CallMethod(struct_module, "unpack", "sO", type_str, pybt);

		Py_DECREF(struct_module);
		Py_DECREF(pybt);
		/*reshape the list in case the original array is multi dimension*/
		/*get the index of each element in new multi dimension array*/
		int ind[narray][ndim];
		int d;
		int i;
		for (i = 0; i < narray; i++)
		{
			int n = narray;
			int ii = i;
			for (d = 0; d < ndim; d++)
			{
				n = n / mem_shape[d];
				ind[i][d] = ii / n;
				ii = ii % n;
			}
		}

		/*put the element of one dimension array into the multi dimension array according to the index*/
		PyObject* list_obj[ndim];
		memset(&list_obj, 0, sizeof(list_obj));
		for (i = 0; i < narray; i++)
		{
			for (d = ndim-1; d >=0; d--)
			{
				/*in the innermost nested list, we set the element in the current list*/
				if (d==ndim-1)
				{
					/*if i is the first element of this list, we need to initialize the list*/
					if(ind[i][d]==0)
					{
						if(list_obj[d] != NULL)
							Py_DECREF(list_obj[d]);
						list_obj[d] = PyList_New(mem_shape[d]);
					}

					PyObject *m_obj_item = PyTuple_GetItem(m_obj, i);
					/*protect borrowed reference, give it to PyList_SetItem*/
					Py_INCREF(m_obj_item);
					PyList_SetItem(list_obj[d], ind[i][d], m_obj_item);
				}
				/*in the rest of nested list, we set the inner list in the current list, once we have the nested list, one element of inner list is changed, current list is changes as well*/
				else
				{
					/*if the index of element in all inner list is 0, we are the first, we have to add this new list to the upper dimension list*/
					int flag=1;
					int dd;
					for(dd=ndim-1; dd>=d+1; dd--)
					{
						if(ind[i][dd]!=0)
							flag=0;
					}
					if(flag==1)
					{
						/*if i is the first element of this list and also the first element of all inner list, we need to initialize this list*/
						if (ind[i][d]==0)
						{
							if(list_obj[d] != NULL)
								Py_DECREF(list_obj[d]);
							list_obj[d] = PyList_New(mem_shape[d]);
						}
						/*if i is the first element of all inner list, we set the last inner list in the current list*/
						/*reference is stolen by PyList_SetItem*/
						Py_INCREF(list_obj[d+1]);
						PyList_SetItem(list_obj[d],ind[i][d],list_obj[d+1]);
					}
				}
			}
		}

		Py_DECREF(m_obj);

		memview_obj = list_obj[0];

		for(i=1; i<ndim; i++)
			Py_DECREF(list_obj[i]);
	}
	/*if the element is unicode character of array.array*/
	else
	{
		/*decode buffer bytes to unicode*/
		PyObject* pyuni = PyUnicode_DecodeUTF32(PyBytes_AsString(pybt), PyBytes_Size(pybt), "can't decode", NULL);
		/*convert unicode to wide char*/
		wchar_t* uni_str = PyUnicode_AsWideCharString(pyuni, NULL);

		if(uni_str != NULL)
		{
			memview_obj = Py_BuildValue("u", uni_str);
			PyMem_Free(uni_str);
		}

		Py_DECREF(pybt);
		Py_DECREF(pyuni);
	}

	return memview_obj;
}

static void pybuffer_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;

	int ndim = pybuffer_interface->dim_size;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);
		if (node == home_node)
		{
			if(pybuffer_interface->object != NULL)
			{
				Py_INCREF(pybuffer_interface->object);
				local_interface->object = pybuffer_interface->object;
			}
			else
			{
				local_interface->object = NULL;
			}
			local_interface->py_buffer = pybuffer_interface->py_buffer;
		}
		else
		{
			local_interface->object = NULL;
			local_interface->py_buffer = NULL;
		}
		local_interface->buffer_type = pybuffer_interface->buffer_type;
		local_interface->buffer_size = pybuffer_interface->buffer_size;
		local_interface->dim_size = pybuffer_interface->dim_size;

#ifdef STARPU_PYTHON_HAVE_NUMPY
		npy_intp* arr_dim = pybuffer_interface->array_dim;
		npy_intp* a_dim;
		if (arr_dim!=NULL)
		{
			a_dim = (npy_intp*)malloc(ndim*sizeof(npy_intp));
			memcpy(a_dim, arr_dim, ndim*sizeof(npy_intp));
		}
		else
			a_dim = NULL;

		local_interface->array_dim = a_dim;
#endif
		local_interface->array_type = pybuffer_interface->array_type;
		local_interface->item_size = pybuffer_interface->item_size;
		local_interface->typecode = pybuffer_interface->typecode;

		int* mem_shape = pybuffer_interface->shape;
		int* m_shape;
		if (mem_shape!=NULL)
		{
			m_shape = (int*)malloc(ndim*sizeof(int));
			memcpy(m_shape, mem_shape, ndim*sizeof(int));
		}
		else
			m_shape = NULL;

		local_interface->shape = m_shape;
	}
}

static void pybuffer_unregister_data_handle(starpu_data_handle_t handle)
{
	unsigned home_node = starpu_data_get_home_node(handle);
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);
		if(node == home_node)
		{
			if(local_interface->object!=NULL)
			{
				Py_DECREF(local_interface->object);
				local_interface->object = NULL;
				local_interface->py_buffer = NULL;
			}
		}
		else
		{
			STARPU_ASSERT(local_interface->object == NULL);
			STARPU_ASSERT(local_interface->py_buffer == NULL);
		}


#ifdef STARPU_PYTHON_HAVE_NUMPY
		free(local_interface->array_dim);
		local_interface->array_dim = NULL;
#endif
		free(local_interface->shape);
		local_interface->shape = NULL;
	}
}

static starpu_ssize_t pybuffer_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;
	starpu_ssize_t requested_memory = pybuffer_interface->buffer_size;

	pybuffer_interface->py_buffer = (char*)starpu_malloc_on_node(node, requested_memory);

	if (!pybuffer_interface->py_buffer)
		return -ENOMEM;

	return requested_memory;
}

static starpu_ssize_t pybuffer_allocate_bytes_data_on_node(void *data_interface, unsigned node)
{
	(void)node;
	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;
	char* pybuf = pybuffer_interface->py_buffer;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;

	STARPU_ASSERT(pybuf == NULL);
	PyObject *pybt=PyBytes_FromStringAndSize(NULL, nbuf);

	pybuffer_interface->object = pybt;

	pybuffer_interface->py_buffer = PyBytes_AsString(pybt);

	if (!pybuffer_interface->py_buffer)
		return -ENOMEM;

	/* release GIL */
	PyGILState_Release(state);

	return nbuf;
}

static void pybuffer_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;
	starpu_ssize_t requested_memory = pybuffer_interface->buffer_size;

	starpu_free_on_node(node, (uintptr_t) pybuffer_interface->py_buffer, requested_memory);

	pybuffer_interface->py_buffer = NULL;
}

static void pybuffer_free_bytes_data_on_node(void *data_interface, unsigned node)
{
	(void)node;
	/*make sure we own the GIL*/
	PyGILState_STATE state = PyGILState_Ensure();

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;

	if (pybuffer_interface->object != NULL)
	{
		Py_DECREF(pybuffer_interface->object);
	}

	pybuffer_interface->object = NULL;
	pybuffer_interface->py_buffer = NULL;

	/* release GIL */
	PyGILState_Release(state);
}

static void pybuffer_reuse_data_on_node(void *dst_data_interface, const void *cached_interface, unsigned node)
{
	(void)node;
	struct starpupy_buffer_interface *dst_pybuffer_interface = (struct starpupy_buffer_interface *) dst_data_interface;
	const struct starpupy_buffer_interface *cached_pybuffer_interface = (const struct starpupy_buffer_interface *) cached_interface;

	dst_pybuffer_interface->object = cached_pybuffer_interface->object;
	dst_pybuffer_interface->py_buffer = cached_pybuffer_interface->py_buffer;
}

static int pybuffer_map_data(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	struct starpupy_buffer_interface *src_pybuf = src_interface;
	struct starpupy_buffer_interface *dst_pybuf = dst_interface;
	int ret;
	uintptr_t mapped;

	mapped = starpu_interface_map((uintptr_t )src_pybuf->py_buffer, 0, src_node, dst_node, (size_t)src_pybuf->buffer_size, &ret);
	if (mapped)
	{
		dst_pybuf->py_buffer = (char*)mapped;
		return 0;
	}
	return ret;
}

static int pybuffer_unmap_data(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	struct starpupy_buffer_interface *src_pybuf = src_interface;
	struct starpupy_buffer_interface *dst_pybuf = dst_interface;

	int ret = starpu_interface_unmap((uintptr_t)src_pybuf->py_buffer, 0, src_node, (uintptr_t)dst_pybuf->py_buffer, dst_node, (size_t)src_pybuf->buffer_size);
	dst_pybuf->py_buffer = 0;

	return ret;
}

static int pybuffer_update_map(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	struct starpupy_buffer_interface *src_pybuf = src_interface;
	struct starpupy_buffer_interface *dst_pybuf = dst_interface;

	return starpu_interface_update_map((uintptr_t)src_pybuf->py_buffer, 0, src_node, (uintptr_t)dst_pybuf->py_buffer, 0, dst_node, (size_t)src_pybuf->buffer_size);
}

static size_t pybuffer_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	size = pybuffer_interface->buffer_size;
	return size;
}

static int pybuffer_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);

	char* pybuf = pybuffer_interface->py_buffer;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;

	char *data;
	data = (void*)starpu_malloc_on_node_flags(node, nbuf, 0);

	memcpy(data, pybuf, nbuf);

	*ptr = data;
	*count = nbuf;
	return 0;
}

static int pybuffer_peek_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)count;

	char *data = ptr;

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);

	memcpy(pybuffer_interface->py_buffer, data, pybuffer_interface->buffer_size);

	return 0;
}

static int pybuffer_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	pybuffer_peek_data(handle, node, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t) ptr, count, 0);

	return 0;
}

static int pybuffer_meta_size(struct starpupy_buffer_interface *pybuffer_interface)
{
	starpu_ssize_t count;

	count = sizeof(pybuffer_interface->buffer_type) +
		/* sizeof(pybuffer_interface->object) +    => built on the fly */
		sizeof(pybuffer_interface->py_buffer) +
		sizeof(pybuffer_interface->buffer_size) +
		sizeof(pybuffer_interface->dim_size) +
		sizeof(pybuffer_interface->array_type) +
		sizeof(pybuffer_interface->item_size) +
		sizeof(pybuffer_interface->typecode) + sizeof(int);
#ifdef STARPU_PYTHON_HAVE_NUMPY
	count += sizeof(int);
#endif
	count += pybuffer_interface->dim_size * (
#ifdef STARPU_PYTHON_HAVE_NUMPY
		sizeof(pybuffer_interface->array_dim[0]) +
#endif
		sizeof(pybuffer_interface->shape[0]));

	return count;
}

#define _pack(dst, src) do { memcpy(dst, &src, sizeof(src)); dst += sizeof(src); } while (0)
static int pybuffer_pack_meta(void *data_interface, void **ptr, starpu_ssize_t *count)
{
	struct starpupy_buffer_interface *pybuffer_interface = data_interface;
	*count = pybuffer_meta_size(pybuffer_interface);
	_STARPU_CALLOC(*ptr, *count, 1);
	char *cur = *ptr;

	_pack(cur, pybuffer_interface->buffer_type);
	_pack(cur, pybuffer_interface->py_buffer);
	_pack(cur, pybuffer_interface->buffer_size);
	_pack(cur, pybuffer_interface->dim_size);
	_pack(cur, pybuffer_interface->array_type);
	_pack(cur, pybuffer_interface->item_size);
	_pack(cur, pybuffer_interface->typecode);

#ifdef STARPU_PYTHON_HAVE_NUMPY
	int array_dim = pybuffer_interface->array_dim ? 1 : 0;
	_pack(cur, array_dim);
	if (pybuffer_interface->array_dim)
	{
		memcpy(cur, pybuffer_interface->array_dim,
		       pybuffer_interface->dim_size * sizeof(pybuffer_interface->array_dim[0]));
		cur += pybuffer_interface->dim_size * sizeof(pybuffer_interface->array_dim[0]);
	}
#endif
	int shape = pybuffer_interface->shape ? 1 : 0;
	_pack(cur, shape);
	if (pybuffer_interface->shape)
		memcpy(cur, pybuffer_interface->shape,
		       pybuffer_interface->dim_size * sizeof(pybuffer_interface->shape[0]));
	return 0;
}

#define _unpack(dst, src) do {	memcpy(&dst, src, sizeof(dst)); src += sizeof(dst); } while(0)
static int pybuffer_unpack_meta(void **data_interface, void *ptr, starpu_ssize_t *count)
{
	_STARPU_CALLOC(*data_interface, 1, sizeof(struct starpupy_buffer_interface));
	struct starpupy_buffer_interface *pybuffer_interface = (*data_interface);
	char *cur = ptr;

	_unpack(pybuffer_interface->buffer_type, cur);
	_unpack(pybuffer_interface->py_buffer, cur);
	_unpack(pybuffer_interface->buffer_size, cur);
	_unpack(pybuffer_interface->dim_size, cur);
	_unpack(pybuffer_interface->array_type, cur);
	_unpack(pybuffer_interface->item_size, cur);
	_unpack(pybuffer_interface->typecode, cur);

#ifdef STARPU_PYTHON_HAVE_NUMPY
	int array_dim;
	_unpack(array_dim, cur);
	if (array_dim)
	{
		_STARPU_MALLOC(pybuffer_interface->array_dim,
			       pybuffer_interface->dim_size * sizeof(pybuffer_interface->array_dim[0]));
		memcpy(pybuffer_interface->array_dim, cur,
		       pybuffer_interface->dim_size * sizeof(pybuffer_interface->array_dim[0]));
		cur += pybuffer_interface->dim_size * sizeof(pybuffer_interface->array_dim[0]);
	}
	else
		pybuffer_interface->array_dim = NULL;
#endif
	int shape;
	_unpack(shape, cur);
	if (shape)
	{
		_STARPU_MALLOC(pybuffer_interface->shape,
			       pybuffer_interface->dim_size * sizeof(pybuffer_interface->shape[0]));
		memcpy(pybuffer_interface->shape, cur,
		       pybuffer_interface->dim_size * sizeof(pybuffer_interface->shape[0]));
	}
	else
		pybuffer_interface->shape = NULL;

	*count = pybuffer_meta_size(pybuffer_interface);

	return 0;
}

static int pybuffer_free_meta(void *data_interface)
{
	struct starpupy_buffer_interface *pybuffer_interface = data_interface;

#ifdef STARPU_PYTHON_HAVE_NUMPY
	free(pybuffer_interface->array_dim);
	pybuffer_interface->array_dim = NULL;
#endif
	free(pybuffer_interface->shape);
	pybuffer_interface->shape = NULL;

	return 0;
}

static uint32_t starpupy_buffer_footprint(starpu_data_handle_t handle)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	int buf_type = pybuffer_interface->buffer_type;
	Py_ssize_t nbuf = pybuffer_interface->buffer_size;
	int ndim = pybuffer_interface->dim_size;
	int arr_type = pybuffer_interface->array_type;
	size_t nitem = pybuffer_interface->item_size;
	size_t narray = 0;
	if(pybuffer_interface->buffer_type != starpupy_bytes_interface && pybuffer_interface->buffer_type != starpupy_bytearray_interface)
	{
		narray = nbuf/nitem;
	}

	uint32_t crc = 0;

	crc=starpu_hash_crc32c_be(buf_type, crc);
	crc=starpu_hash_crc32c_be(nbuf, crc);
	crc=starpu_hash_crc32c_be(ndim, crc);
	crc=starpu_hash_crc32c_be(arr_type, crc);
	crc=starpu_hash_crc32c_be(narray, crc);
	crc=starpu_hash_crc32c_be(nitem, crc);

	return crc;
}

static void pybuffer_display(starpu_data_handle_t handle, FILE *f)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t", pybuffer_interface->dim_size);
}

static int pybuffer_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpupy_buffer_interface *a = (struct starpupy_buffer_interface *) data_interface_a;
	struct starpupy_buffer_interface *b = (struct starpupy_buffer_interface *) data_interface_b;

	return ((a->array_type == b->array_type) && (a->item_size == b->item_size) && (a->dim_size == b->dim_size));
}

static int pybuffer_copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpupy_buffer_interface *src = (struct starpupy_buffer_interface *) src_interface;
	struct starpupy_buffer_interface *dst = (struct starpupy_buffer_interface *) dst_interface;

	starpu_interface_copy((uintptr_t) src->py_buffer, 0, src_node,
			      (uintptr_t) dst->py_buffer, 0, dst_node,
			      src->buffer_size, async_data);
	starpu_interface_data_copy(src_node, dst_node, src->buffer_size);
	return 0;
}

static int pybuffer_copy_bytes_ram_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	struct starpupy_buffer_interface *src = (struct starpupy_buffer_interface *) src_interface;
	struct starpupy_buffer_interface *dst = (struct starpupy_buffer_interface *) dst_interface;

	starpu_interface_copy((uintptr_t) src->py_buffer, 0, src_node,
			      (uintptr_t) dst->py_buffer, 0, dst_node,
			      src->buffer_size, NULL);
	starpu_interface_data_copy(src_node, dst_node, src->buffer_size);
	return 0;
}

static const struct starpu_data_copy_methods pybuffer_copy_data_methods_s =
{
	.any_to_any = pybuffer_copy_any_to_any,
};

static const struct starpu_data_copy_methods pybuffer_bytes_copy_data_methods_s =
{
	.ram_to_ram = pybuffer_copy_bytes_ram_to_ram,
};

static struct starpu_data_interface_ops interface_pybuffer_ops =
{
	.register_data_handle = pybuffer_register_data_handle,
	.unregister_data_handle = pybuffer_unregister_data_handle,
	.allocate_data_on_node = pybuffer_allocate_data_on_node,
	.free_data_on_node = pybuffer_free_data_on_node,
	.reuse_data_on_node = pybuffer_reuse_data_on_node,
	.map_data = pybuffer_map_data,
	.unmap_data = pybuffer_unmap_data,
	.update_map = pybuffer_update_map,
	.get_size = pybuffer_get_size,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpupy_buffer_interface),
	.footprint = starpupy_buffer_footprint,
	.pack_data = pybuffer_pack_data,
	.peek_data = pybuffer_peek_data,
	.unpack_data = pybuffer_unpack_data,
	.pack_meta = pybuffer_pack_meta,
	.unpack_meta = pybuffer_unpack_meta,
	.free_meta = pybuffer_free_meta,
	.dontcache = 0,
	.display = pybuffer_display,
	.compare = pybuffer_compare,
	.name = "STARPUPY_BUFFER_INTERFACE",
	.copy_methods = &pybuffer_copy_data_methods_s,
};

/* we need another interface for bytes, bytearray, array.array, since we have to copy these objects between processes.
* some more explanations are here: https://discuss.python.org/t/adding-pybytes-frombuffer-and-similar-for-array-array/21717
*/
static struct starpu_data_interface_ops interface_pybuffer_bytes_ops =
{
	.register_data_handle = pybuffer_register_data_handle,
	.unregister_data_handle = pybuffer_unregister_data_handle,
	.allocate_data_on_node = pybuffer_allocate_bytes_data_on_node,
	.free_data_on_node = pybuffer_free_bytes_data_on_node,
	.reuse_data_on_node = pybuffer_reuse_data_on_node,
	.get_size = pybuffer_get_size,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpupy_buffer_interface),
	.footprint = starpupy_buffer_footprint,
	.pack_data = pybuffer_pack_data,
	.peek_data = pybuffer_peek_data,
	.unpack_data = pybuffer_unpack_data,
	.dontcache = 0,
	.display = pybuffer_display,
	.compare = pybuffer_compare,
	.name = "STARPUPY_BUFFER_BYTES_INTERFACE",
	.copy_methods = &pybuffer_bytes_copy_data_methods_s,
};

#ifdef STARPU_PYTHON_HAVE_NUMPY
void starpupy_buffer_numpy_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, int ndim, npy_intp* arr_dim, int arr_type, size_t nitem)
{
	struct starpupy_buffer_interface pybuffer_interface =
	{
		.buffer_type = buf_type,
		.py_buffer = pybuf,
		.buffer_size = nbuf,
		.dim_size = ndim,
		.array_dim = arr_dim,
		.array_type = arr_type,
		.item_size = nitem
	};

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);
}
#endif

void starpupy_buffer_bytes_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, PyObject *obj)
{
	struct starpupy_buffer_interface pybuffer_interface =
	{
		.object = obj,
		.buffer_type = buf_type,
		.py_buffer = pybuf,
		.buffer_size = nbuf
	};

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_bytes_ops);
}

void starpupy_buffer_array_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char arr_typecode, size_t nitem, PyObject *obj)
{
	struct starpupy_buffer_interface pybuffer_interface =
	{
		.object = obj,
		.buffer_type = buf_type,
		.py_buffer = pybuf,
		.buffer_size = nbuf,
		.typecode = arr_typecode,
		.item_size = nitem
	};

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_bytes_ops);
}

void starpupy_buffer_memview_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char mem_format, size_t nitem, int ndim, int* mem_shape)
{
	struct starpupy_buffer_interface pybuffer_interface =
	{
		.buffer_type = buf_type,
		.py_buffer = pybuf,
		.buffer_size = nbuf,
		.typecode = mem_format,
		.item_size = nitem,
		.dim_size = ndim,
		.shape = mem_shape
	};

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);
}

int starpupy_check_buffer_interface_id(starpu_data_handle_t handle)
{
	int interfaceid = (int)starpu_data_get_interface_id(handle);
	return (interfaceid == interface_pybuffer_ops.interfaceid || interfaceid == interface_pybuffer_bytes_ops.interfaceid);
}
