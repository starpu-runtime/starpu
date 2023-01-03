/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define RETURN_EXCEPT(...) do{ \
		PyObject *starpupy_err = PyObject_GetAttrString(self, "error"); \
		PyErr_Format(starpupy_err, __VA_ARGS__);		\
		Py_DECREF(starpupy_err); \
		return NULL;\
}while(0)

#define RETURN_EXCEPTION(...) do{ \
		PyObject *starpupy_module = PyObject_GetAttrString(starpu_module, "starpupy"); \
		PyObject *starpupy_err = PyObject_GetAttrString(starpupy_module, "error"); \
		PyErr_Format(starpupy_err, __VA_ARGS__); \
		Py_DECREF(starpupy_module); \
		Py_DECREF(starpupy_err); \
		return NULL;\
}while(0)

PyObject *starpu_module; /*starpu __init__ module*/
PyObject *starpu_dict;  /*starpu __init__ dictionary*/

/*register buffer protocol PyObject*/
static PyObject* starpupy_object_register(PyObject *obj, char* mode)
{
	starpu_data_handle_t handle;
	int home_node = 0;

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
		starpupy_buffer_bytes_register(&handle, home_node, starpupy_bytes_interface, buf_bytes, nbytes, obj);
	}
#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*if the object is a numpy array*/
	else if (strcmp(tp, "numpy.ndarray")==0)
	{
		import_array();
		/*if array is not contiguous, treat it as a normal Python object*/
		if (!PyArray_IS_C_CONTIGUOUS((const PyArrayObject *)obj)&&!PyArray_IS_F_CONTIGUOUS((const PyArrayObject *)obj))
		{
			if(mode != NULL && strcmp(mode, "R")!=0)
			{
				RETURN_EXCEPTION("The mode of object should not be other than R");
			}
			else
			{
				starpupy_data_register(&handle, home_node, obj);
			}
		}
		/*otherwise treat it as Python object supporting buffer protocol*/
		else
		{
			/*get number of dimension*/
			int ndim = PyArray_NDIM((const PyArrayObject *)obj);
			/*get array dim*/
			npy_intp* arr_dim = PyArray_DIMS((PyArrayObject *)obj);
			/*get the item size*/
			int nitem = PyArray_ITEMSIZE((const PyArrayObject *)obj);
			/*get the array type*/
			int arr_type = PyArray_TYPE((const PyArrayObject *)obj);

			/*generate buffer of the array*/
			Py_buffer *view = (Py_buffer *) malloc(sizeof(*view));
			PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);

			/*register the buffer*/
			starpupy_buffer_numpy_register(&handle, home_node, starpupy_numpy_interface, view->buf, view->len, ndim, arr_dim, arr_type, nitem);

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
		starpupy_buffer_bytes_register(&handle, home_node, starpupy_bytearray_interface, view->buf, view->len, obj);

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
		starpupy_buffer_array_register(&handle, home_node, starpupy_array_interface, view->buf, view->len, arr_type, view->itemsize, obj);

		Py_DECREF(PyArrtype);
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
			/*protect borrowed reference*/
			Py_INCREF(shape_args);
			mem_shape[i] = PyLong_AsLong(shape_args);
			Py_DECREF(shape_args);
		}

		/*register the buffer*/
		starpupy_buffer_memview_register(&handle, home_node, starpupy_memoryview_interface, view->buf, view->len, mem_format, view->itemsize, ndim, mem_shape);

		Py_DECREF(PyFormat);
		Py_DECREF(PyShape);
		free(mem_shape);
	}
	/*if the object is PyObject*/
	else
	{
		if(mode != NULL && strcmp(mode, "R")!=0)
		{
			RETURN_EXCEPTION("The mode of object should not be other than R");
		}
		else
		{
			starpupy_data_register(&handle, home_node, obj);
		}
	}

	PyObject *handle_cap=PyCapsule_New(handle, "Handle", NULL);

	return handle_cap;
}

/*register PyObject in a handle*/
PyObject* starpupy_data_register_wrapper(PyObject *self, PyObject *args)
{
	PyObject *obj;
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "OO", &obj, &handle_obj))
		return NULL;

	/*register the python object*/
	PyObject *handle_cap = starpupy_object_register(obj, NULL);

	const char *tp = Py_TYPE(obj)->tp_name;
	//printf("the type of object is %s\n", tp);
	/*if the object is immutable, store the obj_id and handle_obj in handle_set, and registering the same python object several times is authorised*/
	if (strcmp(tp, "int")==0 || strcmp(tp, "float")==0 || strcmp(tp, "str")==0 || strcmp(tp, "bool")==0 || strcmp(tp, "tuple")==0 || strcmp(tp, "range")==0 || strcmp(tp, "complex")==0 || strcmp(tp, "decimal.Decimal")==0 || strcmp(tp, "NoneType")==0)
	{
		/*set handle_obj in handle_set*/
		/*get handle_set*/
		PyObject *handle_set = PyObject_GetAttrString(starpu_module, "handle_set");

		/*add new handle object in set*/
		PySet_Add(handle_set, handle_obj);

		Py_DECREF(handle_set);
	}
	/*if the object is mutable, store the obj_id and handle_obj in handle_dict, and should not register the same python object more than twice*/
	else
	{
		/*set the obj_id and handle_obj in handle_dict*/
		/*get handle_dict*/
		PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");

		/*get object id*/
		PyObject *obj_id = PyObject_CallMethod(handle_obj, "get_obj_id", NULL);

		if(PyDict_GetItem(handle_dict, obj_id)!=NULL)
		{
			RETURN_EXCEPT("Should not register the same mutable python object once more.");
		}

		PyDict_SetItem(handle_dict, obj_id, handle_obj);

		Py_DECREF(handle_dict);
		Py_DECREF(obj_id);
	}

	return handle_cap;
}

/*generate empty Numpy array*/
PyObject* starpupy_numpy_register_wrapper(PyObject *self, PyObject *args)
{
#ifdef STARPU_PYTHON_HAVE_NUMPY
	/*get the first argument*/
	PyObject *dimobj = PyTuple_GetItem(args, 0);
	/*protect borrowed reference, decrement after check*/
	Py_INCREF(dimobj);
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
		RETURN_EXCEPT("Please enter the shape of new array, e.g., (2, 3) or 2");
	}

	Py_DECREF(dimobj);

	/*the second argument is dtype*/
	PyObject *dtype = PyTuple_GetItem(args, 1);
	/*protect borrowed reference*/
	Py_INCREF(dtype);

	/*get the type of target array*/
	PyObject *type_obj = PyObject_GetAttrString(dtype, "num");
	int arr_type = PyLong_AsLong(type_obj);

	Py_DECREF(dtype);
	Py_DECREF(type_obj);

	import_array();
	/*generate a new empty array, it's the return value*/
	PyObject * new_array = PyArray_EMPTY(ndim, dim, arr_type, 0);

	free(dim);
	return new_array;
#endif
	return NULL;
}

/*get PyObject from Handle*/
PyObject *starpupy_get_object_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_cap;

	if (!PyArg_ParseTuple(args, "O", &handle_cap))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	int ret;
	/*call starpu_data_acquire*/
	Py_BEGIN_ALLOW_THREADS
	ret= starpu_data_acquire(handle, STARPU_R);
	Py_END_ALLOW_THREADS
	if (ret!=0)
	{
		RETURN_EXCEPT("Unexpected value %d returned for starpu_data_acquire", ret);
	}

	PyObject *obj = NULL;
	if (STARPUPY_PYOBJ_CHECK(handle))
	{
		struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj = STARPUPY_GET_PYOBJECT(pyobject_interface);
	}

	if (STARPUPY_BUF_CHECK(handle))
	{
		struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);
	}

	/*call starpu_data_release method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_release(handle);
	Py_END_ALLOW_THREADS

	if(obj == NULL)
	{
		RETURN_EXCEPT("Unexpected PyObject value NULL returned for get()");
	}

	return obj;
}

PyObject *starpupy_handle_dict_check(PyObject *obj, char* mode, char* op)
{
	(void)mode;
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
			PyObject *Handle_class = PyDict_GetItemString(starpu_dict, "Handle");

			/*get the constructor, decremented after being called*/
			PyObject *pInstanceHandle = PyInstanceMethod_New(Handle_class);

			/*create a Null Handle object, decremented in the end of this if{}*/
			PyObject *handle_arg = PyTuple_New(1);
			/*obj is used for PyTuple_SetItem(handle_arg), once handle_arg is decremented, obj is decremented as well*/
			Py_INCREF(obj);
			PyTuple_SetItem(handle_arg, 0, obj);

			/*generate the handle object, decremented in the end of this function*/
			handle_obj = PyObject_CallObject(pInstanceHandle,handle_arg);

			/*set the arg_id and handle in handle_dict*/
			PyDict_SetItem(handle_dict, obj_id, handle_obj);

			Py_DECREF(pInstanceHandle);
			Py_DECREF(handle_arg);
		}
		else
		{
			handle_obj = PyDict_GetItem(handle_dict, obj_id);
			/*protect borrowed reference, decremented in the end of this function*/
			Py_INCREF(handle_obj);
		}
	}
	else if (strcmp(op, "exception") == 0)
	{
		/*check in handle_dict whether this arg is already registed*/
		if(!PyDict_Contains(handle_dict, obj_id))
		{
			RETURN_EXCEPTION("Argument does not have registered handle");
		}

		/*get the corresponding handle of the obj*/
		handle_obj = PyDict_GetItem(handle_dict, obj_id);
		/*protect borrowed reference, decremented in the end of this function*/
		Py_INCREF(handle_obj);
	}

	Py_DECREF(handle_dict);
	Py_DECREF(obj_id);

	/*get Handle capsule object, which is the return value of this function*/
	PyObject *handle_cap = PyObject_CallMethod(handle_obj, "get_capsule", NULL);

	Py_DECREF(handle_obj);
	return handle_cap;
}

/*acquire Handle*/
PyObject *starpupy_acquire_handle_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_cap;
	PyObject *pyMode;

	if (!PyArg_ParseTuple(args, "OO", &handle_cap, &pyMode))
		return NULL;

	const char* mode_str = PyUnicode_AsUTF8(pyMode);
	char* obj_mode = strdup(mode_str);

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
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
		RETURN_EXCEPT("Unexpected value returned for starpu_data_acquire");
	}

	PyObject *obj = NULL;
	if (STARPUPY_PYOBJ_CHECK(handle))
	{
		struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj = STARPUPY_GET_PYOBJECT(pyobject_interface);
	}

	if (STARPUPY_BUF_CHECK(handle))
	{
		struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);
	}

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

	/*get the corresponding handle capsule of the obj*/
	PyObject *handle_cap = starpupy_handle_dict_check(obj, NULL, "register");

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	Py_DECREF(handle_cap);

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
		RETURN_EXCEPT("Unexpected value returned for starpu_data_acquire");
	}

	PyObject *obj_get = NULL;
	if (STARPUPY_PYOBJ_CHECK(handle))
	{
		struct starpupyobject_interface *pyobject_interface = (struct starpupyobject_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj_get = STARPUPY_GET_PYOBJECT(pyobject_interface);
	}

	if (STARPUPY_BUF_CHECK(handle))
	{
		struct starpupy_buffer_interface *pybuffer_interface = (struct starpupy_buffer_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

		obj_get = STARPUPY_BUF_GET_PYOBJECT(pybuffer_interface);
	}

	return obj_get;
}

/*release Handle*/
PyObject *starpupy_release_handle_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_cap;

	if (!PyArg_ParseTuple(args, "O", &handle_cap))
		return NULL;

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	if (!STARPUPY_BUF_CHECK(handle))
	{
		RETURN_EXCEPT("Wrong interface is used");
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
	(void)self;
	PyObject *obj;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	/*get the corresponding handle capsule of the obj*/
	PyObject *handle_cap = starpupy_handle_dict_check(obj, NULL, "exception");

	if(handle_cap == NULL)
	{
		Py_XDECREF(handle_cap);
		return NULL;
	}

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	Py_DECREF(handle_cap);

	/*call starpu_data_release method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_release(handle);
	Py_END_ALLOW_THREADS

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}

static void starpupy_remove_handle_from_dict(PyObject *obj_id)
{
	/*delete object from handle_dict*/
	PyObject *handle_dict = PyObject_GetAttrString(starpu_module, "handle_dict");

	if(PyDict_GetItem(handle_dict, obj_id) != NULL)
	{
		PyDict_DelItem(handle_dict, obj_id);
	}

	Py_DECREF(handle_dict);
}

static void starpupy_remove_handle_from_set(PyObject *handle_obj)
{
	/*delete object from handle_set*/
	PyObject *handle_set = PyObject_GetAttrString(starpu_module, "handle_set");

	PySet_Discard(handle_set, handle_obj);

	Py_DECREF(handle_set);
}

/* unregister handle*/
PyObject *starpupy_data_unregister_wrapper(PyObject *self, PyObject *args)
{
	PyObject *handle_obj;

	if (!PyArg_ParseTuple(args, "O", &handle_obj))
		return NULL;

	/*get the handle capsule*/
	PyObject *handle_cap = PyObject_CallMethod(handle_obj, "get_capsule", NULL);
	/*get the id of arg*/
	PyObject *obj_id = PyObject_CallMethod(handle_obj, "get_obj_id", NULL);

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_cap, (void*)-1);

	starpupy_remove_handle_from_dict(obj_id);
	starpupy_remove_handle_from_set(handle_obj);

	Py_DECREF(handle_cap);
	Py_DECREF(obj_id);

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

	/*get the corresponding handle capsule of the obj*/
	PyObject *handle_cap = starpupy_handle_dict_check(obj, NULL, "exception");
	/*get the id of obj*/
	PyObject *obj_id = PyLong_FromVoidPtr(obj);

	if(handle_cap == NULL)
	{
		Py_XDECREF(handle_cap);
		return NULL;
	}

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_cap, (void*)-1);

	starpupy_remove_handle_from_dict(obj_id);

	Py_DECREF(handle_cap);
	Py_DECREF(obj_id);

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

	/*get the handle capsule*/
	PyObject *handle_cap = PyObject_CallMethod(handle_obj, "get_capsule", NULL);
	/*get the id of arg*/
	PyObject *obj_id = PyObject_CallMethod(handle_obj, "get_obj_id", NULL);

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		RETURN_EXCEPT("Handle has already been unregistered");
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister_submit(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_cap, (void*)-1);

	starpupy_remove_handle_from_dict(obj_id);
	starpupy_remove_handle_from_set(handle_obj);

	Py_DECREF(handle_cap);
	Py_DECREF(obj_id);

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

	/*get the corresponding handle capsule of the obj*/
	PyObject *handle_cap = starpupy_handle_dict_check(obj, NULL, "exception");
	/*get the id of obj*/
	PyObject *obj_id = PyLong_FromVoidPtr(obj);

	if(handle_cap == NULL)
	{
		Py_XDECREF(handle_cap);
		return NULL;
	}

	/*PyObject *->handle*/
	starpu_data_handle_t handle = (starpu_data_handle_t) PyCapsule_GetPointer(handle_cap, "Handle");

	if (handle == (void*)-1)
	{
		PyErr_Format(PyObject_GetAttrString(self, "error"), "Handle has already been unregistered");
		return NULL;
	}

	/*call starpu_data_unregister method*/
	Py_BEGIN_ALLOW_THREADS
	starpu_data_unregister_submit(handle);
	Py_END_ALLOW_THREADS

	PyCapsule_SetPointer(handle_cap, (void*)-1);

	starpupy_remove_handle_from_dict(obj_id);

	Py_DECREF(handle_cap);
	Py_DECREF(obj_id);

	/*return type is void*/
	Py_INCREF(Py_None);
	return Py_None;
}
