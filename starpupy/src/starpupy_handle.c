/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "starpupy_interface.h"
#include "starpupy_buffer_interface.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#include "starpupy_handle.h"

PyObject *starpu_module; /*starpu __init__ module*/
int buf_id;
int obj_id;

/*register buffer protocol PyObject*/
static PyObject* starpupy_object_register(PyObject *obj, char* mode)
{
	starpu_data_handle_t handle;
	int home_node = 0;

	Py_INCREF(obj);
	const char *tp = Py_TYPE(obj)->tp_name;
	//printf("the type of object is %s\n", tp);
	/*if the object is bytes*/
	if (strcmp(tp, "bytes")==0)
	{
		/*bytes size*/
		Py_ssize_t nbytes;
		char* buf_bytes;

		PyBytes_AsStringAndSize(obj, &buf_bytes, &nbytes);

		/*register the buffer*/
		buf_id = starpupy_buffer_bytes_register(&handle, home_node, starpupy_bytes_interface, buf_bytes, nbytes);
	}
#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*if the object is a numpy array*/
	else if (strcmp(tp, "numpy.ndarray")==0)
	{
		import_array();
		/*if array is not contiguous, treat it as a normal Python object*/
		if (!PyArray_IS_C_CONTIGUOUS(obj)&&!PyArray_IS_F_CONTIGUOUS(obj))
		{
			if(mode != NULL && strcmp(mode, "R")!=0)
			{
				PyObject *starpupy_module = PyObject_GetAttrString(starpu_module, "starpupy");
				PyErr_Format(PyObject_GetAttrString(starpupy_module, "error"), "The mode of object should not be other than R");
				return NULL;
			}
			else
			{
				obj_id = starpupy_data_register(&handle, home_node, obj);
			}
		}
		/*otherwise treat it as Python object supporting buffer protocol*/
		else
		{
			/*get number of dimension*/
			int ndim = PyArray_NDIM(obj);
			/*get array dim*/
			npy_intp* arr_dim = PyArray_DIMS(obj);
			/*get the item size*/
			int nitem = PyArray_ITEMSIZE(obj);
			/*get the array type*/
			int arr_type = PyArray_TYPE(obj);

			/*generate buffer of the array*/
			Py_buffer *view = (Py_buffer *) malloc(sizeof(*view));
			PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);

			/*register the buffer*/
			buf_id = starpupy_buffer_numpy_register(&handle, home_node, starpupy_numpy_interface, view->buf, view->len, ndim, arr_dim, arr_type, nitem);

			PyBuffer_Release(view);
			free(view);
		}
	}
#endif
	/*if the object is bytearray*/
	else if (strcmp(tp, "bytearray")==0)
	{
		/*generate buffer of the array*/
		Py_buffer *view = (Py_buffer *) malloc(sizeof(*view));
		PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);

		/*register the buffer*/
		buf_id = starpupy_buffer_bytes_register(&handle, home_node, starpupy_bytearray_interface, view->buf, view->len);

		PyBuffer_Release(view);
		free(view);
	}
	/*if the object is array.array*/
	else if (strcmp(tp, "array.array")==0)
	{
		/*get the arraytype*/
		PyObject* PyArrtype=PyObject_GetAttrString(obj,"typecode");

		const char* type_str = PyUnicode_AsUTF8(PyArrtype);
		char arr_type = type_str[0];

		/*generate buffer of the array*/
		Py_buffer *view = (Py_buffer *) malloc(sizeof(*view));
		PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);

		/*register the buffer*/
		buf_id = starpupy_buffer_array_register(&handle, home_node, starpupy_array_interface, view->buf, view->len, arr_type, view->itemsize);

		PyBuffer_Release(view);
		free(view);
	}
	/*if the object is memoryview*/
	else if (strcmp(tp, "memoryview")==0)
	{
		/*generate buffer of the memoryview*/
		Py_buffer *view = PyMemoryView_GET_BUFFER(obj);

		/*get the format of memoryview*/
		PyObject* PyFormat=PyObject_GetAttrString(obj,"format");

		const char* format_str = PyUnicode_AsUTF8(PyFormat);
		char mem_format = format_str[0];

		PyObject* PyShape=PyObject_GetAttrString(obj,"shape");

		int ndim = PyTuple_Size(PyShape);
		int* mem_shape;
		mem_shape = (int*)malloc(ndim*sizeof(int));
		int i;
		for(i=0; i<ndim; i++)
		{
			PyObject* shape_args = PyTuple_GetItem(PyShape, i);
			mem_shape[i] = PyLong_AsLong(shape_args);
		}

		/*register the buffer*/
		buf_id = starpupy_buffer_memview_register(&handle, home_node, starpupy_memoryview_interface, view->buf, view->len, mem_format, view->itemsize, ndim, mem_shape);

		free(mem_shape);
	}
	/*if the object is PyObject*/
	else
	{
		if(mode != NULL && strcmp(mode, "R")!=0)
		{
			PyObject *starpupy_module = PyObject_GetAttrString(starpu_module, "starpupy");
			PyErr_Format(PyObject_GetAttrString(starpupy_module, "error"), "The mode of object should not be other than R");
			return NULL;
		}
		else
		{
			obj_id = starpupy_data_register(&handle, home_node, obj);
		}
	}

	PyObject *handle_obj=PyCapsule_New(handle, "Handle", NULL);

	return handle_obj;
}

/*register PyObject in a handle*/
PyObject* starpupy_data_register_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	PyObject *handle_obj = starpupy_object_register(obj, NULL);

	return handle_obj;
}

/*register empty Numpy array in a handle*/
PyObject* starpupy_numpy_register_wrapper(PyObject *self, PyObject *args)
{
#ifdef STARPU_PYTHON_HAVE_NUMPY
	starpu_data_handle_t handle;
	int home_node = -1;
	PyObject* dtype; /*dtype of numpy array*/

	/*get the first argument*/
	PyObject* dimobj = PyTuple_GetItem(args, 0);
	/*detect whether user provides dtype or not*/
	int ndim;
	npy_intp *dim;
	/*if the first argument is integer, it's an array one dimension*/
	if(PyLong_Check(dimobj))
	{
		ndim = 1;
		dim = (npy_intp*)malloc(ndim*sizeof(npy_intp));
		dim[0] = PyLong_AsLong(dimobj);
	}
	/*if the first argument is a tuple, it contains information of dimension*/
	else if(PyTuple_Check(dimobj))
	{
		ndim = PyTuple_Size(dimobj);
		dim = (npy_intp*)malloc(ndim*sizeof(npy_intp));
		int i;
		for (i=0; i<ndim; i++)
		{
			dim[i] = PyLong_AsLong(PyTuple_GetItem(dimobj, i));
		}
	}
	else
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Please enter the shape of new array, e.g., (2, 3) or 2");
		return NULL;
	}

	/*the second argument is dtype*/
	dtype = PyTuple_GetItem(args, 1);

	/*get the size of array*/
	int narray = 1;
	int i;
	for (i=0; i<ndim; i++)
	{
		narray = narray*dim[i];
	}

	/*get the type of target array*/
	PyObject *type_obj = PyObject_GetAttrString(dtype, "num");
	int arr_type = PyLong_AsLong(type_obj);

	/*get the item size of target array*/
	PyObject *nitem_obj = PyObject_GetAttrString(dtype, "itemsize");
	int nitem = PyLong_AsLong(nitem_obj);

	import_array();
	/*generate a new empty array*/
	PyObject * new_array = PyArray_EMPTY(ndim, dim, arr_type, 0);

	npy_intp* arr_dim = PyArray_DIMS(new_array);

	/*register the buffer*/
	buf_id = starpupy_buffer_numpy_register(&handle, home_node, starpupy_numpy_interface, 0, narray*nitem, ndim, arr_dim, arr_type, nitem);

	/*handle->PyObject**/
	PyObject *handle_array=PyCapsule_New(handle, "Handle", NULL);

	free(dim);

	return handle_array;
#endif
	return NULL;
}

/*get PyObject from Handle*/
PyObject *starpupy_get_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "O", &handle_obj))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	if (obj_id!=starpu_data_get_interface_id(handle))
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Wrong interface is used");
		return NULL;
	}

	int ret;
	/*call starpu_data_acquire*/
	Py_BEGIN_ALLOW_THREADS
	ret= starpu_data_acquire(handle, STARPU_R);
	Py_END_ALLOW_THREADS
	if (ret!=0)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Unexpected value %d returned for starpu_data_acquire", ret);
		return NULL;
	}

	struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	PyObject *obj = STARPUPY_GET_PYOBJECT(pyobject_interface);

	/*call starpu_data_release method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_release(handle);
	Py_END_ALLOW_THREADS

	if(obj == NULL)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Unexpected PyObject value NULL returned for get()");
		return NULL;
	}

	return obj;
}

PyObject *handle_dict_check(PyObject *obj, char* mode, char* op)
{
	/*get handle_dict*/
	PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");
	/*get the arg id*/
	PyObject *obj_id = PyLong_FromVoidPtr(obj); //XXX in CPython, the pointer of object can be treated as it's id, in other implementation, it may be realised by other ways

	PyObject *handle_obj = NULL;
	if (strcmp(op, "register") == 0)
	{
		/*check whether the arg is already registed*/
		if(PyDict_GetItem(handle_dict, obj_id)==NULL)
		{
			/*get the handle of arg*/
			handle_obj = starpupy_object_register(obj, mode);
			/*set the arg_id and handle in handle_dict*/
			handle_dict = PyObject_CallMethod(starpu_module, "handle_dict_set_item", "OO", obj, handle_obj);
		}
		else
		{
			handle_obj = PyDict_GetItem(handle_dict, obj_id);
		}
	}
	else if (strcmp(op, "exception") == 0)
	{
		/*check in handle_dict whether this arg is already registed*/
		if(!PyDict_Contains(handle_dict, obj_id))
		{
			PyObject *starpupy_module = PyObject_GetAttrString(starpu_module, "starpupy");
			PyErr_Format(PyObject_GetAttrString(starpupy_module, "error"), "Argument does not have registered handle");
			return NULL;
		}

		/*get the corresponding handle of the obj*/
		handle_obj = PyDict_GetItem(handle_dict, obj_id);
	}

	Py_DECREF(handle_dict);
	Py_DECREF(obj_id);

	return handle_obj;

}
/*acquire Handle*/
PyObject *starpupy_acquire_handle_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;
	PyObject *pyMode;

	if (!PyArg_ParseTuple(args, "OO", &handle_obj, &pyMode))
		return NULL;

	const char* mode_str = PyUnicode_AsUTF8(pyMode);
	char* obj_mode = strdup(mode_str);

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	if (buf_id!=starpu_data_get_interface_id(handle))
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Wrong interface is used");
		return NULL;
	}

	int ret=0;
	if(strcmp(obj_mode, "R") == 0)
	{
		/*call starpu_data_acquire(STARPU_R)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_R);
		Py_END_ALLOW_THREADS
	}
	if(strcmp(obj_mode, "W") == 0)
	{
		/*call starpu_data_acquire(STARPU_W)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_W);
		Py_END_ALLOW_THREADS
	}
	if(strcmp(obj_mode, "RW") == 0)
	{
		/*call starpu_data_acquire(STARPU_RW)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_RW);
		Py_END_ALLOW_THREADS
	}

	free(obj_mode);

	if (ret!=0)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Unexpected value returned for starpu_data_acquire");
		return NULL;
	}

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	PyObject *obj = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);

	return obj;
}

/*acquire PyObject Handle*/
PyObject *starpupy_acquire_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;
	PyObject *pyMode;

	if (!PyArg_ParseTuple(args, "OO", &obj, &pyMode))
		return NULL;

	const char* mode_str = PyUnicode_AsUTF8(pyMode);
	char* obj_mode = strdup(mode_str);

	/*get the corresponding handle of the obj*/
	PyObject *handle_obj = handle_dict_check(obj, NULL, "register");

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (buf_id!=starpu_data_get_interface_id(handle))
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Wrong interface is used");
		return NULL;
	}

	int ret=0;
	if(strcmp(obj_mode, "R") == 0)
	{
		/*call starpu_data_acquire(STARPU_R)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_R);
		Py_END_ALLOW_THREADS
	}

	if(strcmp(obj_mode, "W") == 0)
	{
		/*call starpu_data_acquire(STARPU_W)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_W);
		Py_END_ALLOW_THREADS
	}

	if(strcmp(obj_mode, "RW") == 0)
	{
		/*call starpu_data_acquire(STARPU_RW)*/
		Py_BEGIN_ALLOW_THREADS
		ret= starpu_data_acquire(handle, STARPU_RW);
		Py_END_ALLOW_THREADS
	}

	free(obj_mode);

	if (ret!=0)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Unexpected value returned for starpu_data_acquire");
		return NULL;
	}

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	PyObject *obj_get = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);

	return obj_get;
}

/*release Handle*/
PyObject *starpupy_release_handle_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "O", &handle_obj))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	if (buf_id!=starpu_data_get_interface_id(handle))
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Wrong interface is used");
		return NULL;
	}

	/*call starpu_data_release method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_release(handle);
	Py_END_ALLOW_THREADS

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/*release PyObejct Handle*/
PyObject *starpupy_release_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	/*get the corresponding handle of the obj*/
	PyObject *handle_obj = handle_dict_check(obj, NULL, "exception");

	if(handle_obj == NULL)
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	/*call starpu_data_release method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_release(handle);
	Py_END_ALLOW_THREADS

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/* unregister handle*/
PyObject *starpupy_data_unregister_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "O", &handle_obj))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	PyObject *obj_get = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);
	Py_DECREF(obj_get);

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_obj, (void*)-1);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/* unregister PyObject handle*/
PyObject *starpupy_data_unregister_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	/*get the corresponding handle of the obj*/
	PyObject *handle_obj = handle_dict_check(obj, NULL, "exception");

	if(handle_obj == NULL)
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_obj, (void*)-1);

	/*delete object from handle_dict*/
	PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");
	/*get the id of arg*/
	PyObject *arg_id = PyLong_FromVoidPtr(obj);
	PyDict_DelItem(handle_dict, arg_id);

	Py_DECREF(obj);
	Py_DECREF(handle_dict);
	Py_DECREF(arg_id);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/* unregister_submit handle*/
PyObject *starpupy_data_unregister_submit_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "O", &handle_obj))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister_submit(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_obj, (void*)-1);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

/* unregister_submit PyObject handle*/
PyObject *starpupy_data_unregister_submit_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	/*get the corresponding handle of the obj*/
	PyObject *handle_obj = handle_dict_check(obj, NULL, "exception");

	if(handle_obj == NULL)
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_obj, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister_submit(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_obj, (void*)-1);
	/*delete object from handle_dict*/
	PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");
	/*get the id of arg*/
	PyObject *arg_id = PyLong_FromVoidPtr(obj);
	PyDict_DelItem(handle_dict, arg_id);

	Py_DECREF(handle_dict);
	Py_DECREF(arg_id);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}
