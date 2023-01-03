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
#include <stdarg.h>
#include <starpu.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef STARPU_PYTHON_HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

struct starpupy_buffer_interface
{
	enum BufType {starpupy_numpy_interface, starpupy_bytes_interface, starpupy_bytearray_interface, starpupy_array_interface, starpupy_memoryview_interface}buffer_type;
	PyObject* object; /* For bytes, bytearray, array.array, object corresponding py_buffer */
	char* py_buffer;	/* The buffer actually allocated to store the data */
	Py_ssize_t buffer_size;	/* The size of py_buffer */
	int dim_size;		/* For numpy objects, the dimension */
#ifdef STARPU_PYTHON_HAVE_NUMPY
	npy_intp* array_dim;	/* For numpy objects, the shapes of the different dimentions */
#endif
	int array_type;		/* The type of elements */
	size_t item_size;	/* The size of elements */
	char typecode;		/* For array.array, the type of elements */
	int* shape;		/* For memoryview, the shape of each dimension */
};

#ifdef STARPU_PYTHON_HAVE_NUMPY
void starpupy_buffer_numpy_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, int ndim, npy_intp* arr_dim, int arr_type, size_t nitem);
#endif

void starpupy_buffer_bytes_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, PyObject* obj);

void starpupy_buffer_array_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char arr_typecode, size_t nitem, PyObject* obj);

void starpupy_buffer_memview_register(starpu_data_handle_t *handleptr, int home_node, int buf_type, char* pybuf, Py_ssize_t nbuf, char mem_format, size_t nitem, int ndim, int* mem_shape);

int starpupy_check_buffer_interface_id(starpu_data_handle_t handle);

PyObject* starpupy_buffer_get_numpy(struct starpupy_buffer_interface *pybuffer_interface);

PyObject* starpupy_buffer_get_arrarr(struct starpupy_buffer_interface *pybuffer_interface);

PyObject* starpupy_buffer_get_memview(struct starpupy_buffer_interface *pybuffer_interface);

#define STARPUPY_BUF_CHECK(handle) (starpupy_check_buffer_interface_id(handle))
#define STARPUPY_BUF_GET_TYPE(interface) (((struct starpupy_buffer_interface *)(interface))->buffer_type)
#define STARPUPY_BUF_GET_OBJ(interface) (Py_INCREF(((struct starpupy_buffer_interface *)(interface))->object), ((struct starpupy_buffer_interface *)(interface))->object)
#define STARPUPY_BUF_GET_PYBUF(interface) (((struct starpupy_buffer_interface *)(interface))->py_buffer)
#define STARPUPY_BUF_GET_NBUF(interface) (((struct starpupy_buffer_interface *)(interface))->buffer_size)
#define STARPUPY_BUF_GET_NDIM(interface) (((struct starpupy_buffer_interface *)(interface))->dim_size)
#define STARPUPY_BUF_GET_DIM(interface) (((struct starpupy_buffer_interface *)(interface))->array_dim)
#define STARPUPY_BUF_GET_ARRTYPE(interface) (((struct starpupy_buffer_interface *)(interface))->array_type)
#define STARPUPY_BUF_GET_NITEM(interface) (((struct starpupy_buffer_interface *)(interface))->item_size)
#define STARPUPY_BUF_GET_TYPECODE(interface) (((struct starpupy_buffer_interface *)(interface))->typecode)
#define STARPUPY_BUF_GET_SHAPE(interface) (((struct starpupy_buffer_interface *)(interface))->shape)

#define STARPUPY_BUF_GET_PYNUMPY(interface) (starpupy_buffer_get_numpy(interface))

#define STARPUPY_BUF_GET_PYBYTES(interface) (PyBytes_FromStringAndSize(STARPUPY_BUF_GET_PYBUF(interface), STARPUPY_BUF_GET_NBUF(interface)))

#define STARPUPY_BUF_GET_PYARRAY(interface) (starpupy_buffer_get_arrarr(interface))

#define STARPUPY_BUF_GET_PYMEMVIEW(interface) (starpupy_buffer_get_memview(interface))

#define STARPUPY_BUF_GET_PYOBJECT(interface)\
	(STARPUPY_BUF_GET_TYPE(interface)==starpupy_numpy_interface ? STARPUPY_BUF_GET_PYNUMPY(interface) \
	 : STARPUPY_BUF_GET_TYPE(interface)==starpupy_bytes_interface || STARPUPY_BUF_GET_TYPE(interface)==starpupy_bytearray_interface || STARPUPY_BUF_GET_TYPE(interface)==starpupy_array_interface ?  STARPUPY_BUF_GET_OBJ(interface) \
	 : STARPUPY_BUF_GET_TYPE(interface)==starpupy_memoryview_interface ? STARPUPY_BUF_GET_PYMEMVIEW(interface) \
	 : NULL)
