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

extern PyObject *starpu_module; /*starpu __init__ module*/
extern PyObject *starpu_dict;  /*starpu __init__ dictionary*/

PyObject *handle_dict_check(PyObject *obj, char* mode, char* op);

PyObject *starpupy_data_register_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_numpy_register_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_get_object_wrapper(PyObject *self, PyObject *args);

PyObject *starpupy_acquire_handle_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_acquire_object_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_release_handle_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_release_object_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_data_unregister_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_data_unregister_object_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_data_unregister_submit_wrapper(PyObject *self, PyObject *args);
PyObject *starpupy_data_unregister_submit_object_wrapper(PyObject *self, PyObject *args);

