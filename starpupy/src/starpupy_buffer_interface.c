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

#include <stdarg.h>
#include <starpu.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
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
	/*create an instance of array.array*/
	PyObject *arr_instance = PyInstanceMethod_New(arr_class);

	/*get the buffer bytes*/
	PyObject *pybt=PyBytes_FromStringAndSize(pybuf, nbuf);

	/*get the array elements*/
	PyObject *arr_list;

	/*if the element is not unicode character*/
	if (arr_typecode!='u')
	{
		/*get the array type string*/
		char type_c[narray];
		int i;
		for (i=0; i<narray; i++)
		{
			type_c[i]=arr_typecode;
		}
		type_c[i] = '\0';

		char* type_str= strdup(type_c);

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

		arr_list = Py_BuildValue("u", uni_str);
	}

	/*initialize the instance*/
	PyObject *arr_args=PyTuple_New(2);

	char arr_type[]={arr_typecode};
	PyTuple_SetItem(arr_args, 0, Py_BuildValue("s", arr_type));
	PyTuple_SetItem(arr_args, 1, arr_list);

	PyObject *arr_obj = PyObject_CallObject(arr_instance,arr_args);

	Py_DECREF(pybt);
	Py_DECREF(arr_module);

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

	PyObject *pybt=PyBytes_FromStringAndSize(pybuf, nbuf);

	PyObject *memview_obj;
	if(mem_format=='B')
	{
		memview_obj = pybt;
	}
	/*if the element is not unicode character of array.array*/
	else if(mem_format!='w')
	{
		char type_c[narray];
		int i;
		for (i = 0; i<narray; i++)
		{
			type_c[i]=mem_format;
		}
		type_c[i] = '\0';

		char* type_str= strdup(type_c);

		/*get the array element list using struct module*/
		PyObject *struct_module = PyImport_ImportModule("struct");
		PyObject *m_obj = PyObject_CallMethod(struct_module, "unpack", "sO", type_str, pybt);

		/*reshape the list in case the original array is multi dimension*/
		/*get the index of each element in new multi dimension array*/
		int ind[narray][ndim];
		int d;
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
						list_obj[d] = PyList_New(mem_shape[d]);
					}

					PyList_SetItem(list_obj[d],ind[i][d],PyTuple_GetItem(m_obj, i));
				}
				/*in the rest of nested list, we set the inner list in the current list, once we have the nested list, one element of inner list is changed, current list is changes as well*/
				else
				{
					/*if the index of element in all inner list is 0, flag is 1*/
					int flag=1;
					for(int dd= ndim-1; dd>=d+1; dd--)
					{
						if(ind[i][dd]!=0)
							flag=0;
					}
					if(flag==1)
					{
						/*if i is the first element of this list and also the first element of all inner list, we need to initialize this list*/
						if (ind[i][d]==0)
						{
							list_obj[d] = PyList_New(mem_shape[d]);
						}
						/*if i is the first element of all inner list, we set the last inner list in the current list*/
						PyList_SetItem(list_obj[d],ind[i][d],list_obj[d+1]);
					}
				}
			}
		}

		memview_obj = list_obj[0];

	    Py_DECREF(struct_module);
	}
	/*if the element is unicode character of array.array*/
	else
	{
		/*decode buffer bytes to unicode*/
		PyObject* pyuni = PyUnicode_DecodeUTF32(PyBytes_AsString(pybt), PyBytes_Size(pybt), "can't decode", NULL);
		/*convert unicode to wide char*/
		wchar_t* uni_str = PyUnicode_AsWideCharString(pyuni, NULL);

		memview_obj = Py_BuildValue("u", uni_str);
	}

	return memview_obj;
}

static void pybuffer_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;

	int ndim = pybuffer_interface->dim_size;

#ifdef STARPU_PYTHON_HAVE_NUMPY
	npy_intp* arr_dim = pybuffer_interface->array_dim;
	npy_intp* a_dim = (npy_intp*)malloc(ndim*sizeof(npy_intp));
	if (arr_dim!=NULL)
		memcpy(a_dim, arr_dim, ndim*sizeof(npy_intp));
	else
		a_dim = NULL;
#endif

	int* mem_shape = pybuffer_interface->shape;
	int* m_shape = (int*)malloc(ndim*sizeof(int));
	if (mem_shape!=NULL)
		memcpy(m_shape, mem_shape, ndim*sizeof(int));
	else
		m_shape = NULL;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);
		if (node == home_node)
		{
			local_interface->py_buffer = pybuffer_interface->py_buffer;
		}
		else
		{
			local_interface->py_buffer = NULL;
		}
		local_interface->buffer_type = pybuffer_interface->buffer_type;
		local_interface->buffer_size = pybuffer_interface->buffer_size;
		local_interface->dim_size = pybuffer_interface->dim_size;
		//local_interface->array_dim = pybuffer_interface->array_dim;
#ifdef STARPU_PYTHON_HAVE_NUMPY
		local_interface->array_dim = a_dim;
#endif
		local_interface->array_type = pybuffer_interface->array_type;
		local_interface->item_size = pybuffer_interface->item_size;
		local_interface->typecode = pybuffer_interface->typecode;
		//local_interface->shape = pybuffer_interface->shape;
		local_interface->shape = m_shape;
	}
}

static void pybuffer_unregister_data_handle(starpu_data_handle_t handle)
{
	unsigned node=0;

	struct starpupy_buffer_interface *local_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_PYTHON_HAVE_NUMPY
	free(local_interface->array_dim);
#endif
	free(local_interface->shape);
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

static void pybuffer_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) data_interface;

	starpu_ssize_t requested_memory = pybuffer_interface->buffer_size;

	starpu_free_on_node(node, (uintptr_t) pybuffer_interface->py_buffer, requested_memory);
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

static struct starpu_data_interface_ops interface_pybuffer_ops =
{
	.register_data_handle = pybuffer_register_data_handle,
	.unregister_data_handle = pybuffer_unregister_data_handle,
	.allocate_data_on_node = pybuffer_allocate_data_on_node,
	.free_data_on_node = pybuffer_free_data_on_node,
	.get_size = pybuffer_get_size,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpupy_buffer_interface),
	.footprint = starpupy_buffer_footprint,
	.pack_data = pybuffer_pack_data,
	.peek_data = pybuffer_peek_data,
	.unpack_data = pybuffer_unpack_data,
	.dontcache = 0,
};

#ifdef STARPU_PYTHON_HAVE_NUMPY
int starpupy_buffer_numpy_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, int ndim, npy_intp* arr_dim, int arr_type, size_t nitem)
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

	if (interface_pybuffer_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_pybuffer_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);

	return interface_pybuffer_ops.interfaceid;

}
#endif

int starpupy_buffer_bytes_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf)
{

	struct starpupy_buffer_interface pybuffer_interface =
	{
	 .buffer_type = buf_type,
	 .py_buffer = pybuf,
	 .buffer_size = nbuf
	};

	if (interface_pybuffer_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_pybuffer_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);

	return interface_pybuffer_ops.interfaceid;
}

int starpupy_buffer_array_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char arr_typecode, size_t nitem)
{
	struct starpupy_buffer_interface pybuffer_interface =
	{
	 .buffer_type = buf_type,
	 .py_buffer = pybuf,
	 .buffer_size = nbuf,
	 .typecode = arr_typecode,
	 .item_size = nitem
	};

	if (interface_pybuffer_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_pybuffer_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);

	return interface_pybuffer_ops.interfaceid;
}

int starpupy_buffer_memview_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char mem_format, size_t nitem, int ndim, int* mem_shape)
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

	if (interface_pybuffer_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_pybuffer_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &pybuffer_interface, &interface_pybuffer_ops);

	return interface_pybuffer_ops.interfaceid;
}
