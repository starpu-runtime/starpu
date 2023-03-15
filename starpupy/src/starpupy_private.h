/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPUPY__PRIVATE_H
#define __STARPUPY__PRIVATE_H

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

#endif // __STARPUPY__PRIVATE_H
