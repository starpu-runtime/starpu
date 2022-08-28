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

PyObject *dumps; /*cloudpickle.dumps method*/
PyObject *loads; /*pickle.loads method*/

/*return the reference of PyBytes which must be kept while using obj_data. See documentation of PyBytes_AsStringAndSize()*/
static inline PyObject* starpu_cloudpickle_dumps(PyObject *obj, char **obj_data, Py_ssize_t *obj_data_size)
{
	PyObject *obj_bytes= PyObject_CallFunctionObjArgs(dumps, obj, NULL);

	PyBytes_AsStringAndSize(obj_bytes, obj_data, obj_data_size);

	return obj_bytes;
}

static inline PyObject* starpu_cloudpickle_loads(char* pyString, Py_ssize_t pyString_size)
{
	PyObject *obj_bytes_str = PyBytes_FromStringAndSize(pyString, pyString_size);
	PyObject *obj = PyObject_CallFunctionObjArgs(loads, obj_bytes_str, NULL);

	Py_DECREF(obj_bytes_str);

	return obj;
}
